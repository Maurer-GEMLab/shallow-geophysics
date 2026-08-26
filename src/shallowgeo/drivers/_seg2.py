"""Minimal SEG-2 reader (Pullan, 1990, *Geophysics* 55(9), 1260-1271).

Implemented here rather than delegated to ObsPy for two reasons. It keeps
``shallowgeo.core`` importable without ObsPy, and -- more importantly -- it
hands back the *raw* free-format header strings. SEG-2's header vocabulary is
only loosely standardised, every vendor writes its own keys, and the
Geometrics-specific interpretation is precisely the value this package adds.
A reader that normalises the headers before we see them is the wrong shape.

This module deliberately does no interpretation. It returns bytes and strings;
:mod:`shallowgeo.drivers.geode_seg2` and :mod:`shallowgeo.drivers.atom_seg2`
decide what they mean.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FILE_DESCRIPTOR_ID = 0x3A55
TRACE_DESCRIPTOR_ID = 0x4422

#: SEG-2 data format codes -> (numpy dtype string, bytes per sample)
_FORMATS = {
    1: ("i2", 2),   # 16-bit fixed point
    2: ("i4", 4),   # 32-bit fixed point
    4: ("f4", 4),   # 32-bit IEEE float
    5: ("f8", 8),   # 64-bit IEEE double
}
# Code 3 is 20-bit SEG-D floating point, which no instrument in scope emits.


class SEG2Error(ValueError):
    """Malformed or unsupported SEG-2 content."""


@dataclass
class SEG2Trace:
    """One trace: its free-format header strings and its samples."""

    header: dict[str, str]
    data: np.ndarray


@dataclass
class SEG2File:
    """A parsed SEG-2 file, uninterpreted."""

    header: dict[str, str]
    traces: list[SEG2Trace] = field(default_factory=list)
    endian: str = "<"
    revision: int = 0

    @property
    def n_traces(self) -> int:
        return len(self.traces)


def looks_like_seg2(data: bytes) -> bool:
    """Whether *data* starts with a SEG-2 file descriptor block.

    Cheap enough for ``can_open``; needs only the first two bytes.
    """
    if len(data) < 2:
        return False
    return (
        struct.unpack("<H", data[:2])[0] == FILE_DESCRIPTOR_ID
        or struct.unpack(">H", data[:2])[0] == FILE_DESCRIPTOR_ID
    )


def _parse_strings(block: bytes, endian: str, terminator: bytes) -> dict[str, str]:
    """Walk a free-format string sub-block.

    Each entry is a 2-byte offset to the next entry followed by
    ``KEY<space>VALUE``. An offset of zero ends the block. Malformed offsets
    are common in the wild, so anything that would not advance the cursor is
    treated as a terminator rather than an error -- refusing to read a whole
    field file over one bad byte is not useful behaviour.
    """
    out: dict[str, str] = {}
    pos = 0
    limit = len(block)
    while pos + 2 <= limit:
        (offset,) = struct.unpack(endian + "H", block[pos : pos + 2])
        if offset == 0:
            break
        if offset < 3 or pos + offset > limit:
            break
        raw = block[pos + 2 : pos + offset]
        if terminator:
            raw = raw.split(terminator)[0]
        text = raw.rstrip(b"\x00").decode("latin-1").strip()
        if text:
            key, _, value = text.partition(" ")
            out[key.strip().upper()] = value.strip()
        pos += offset
    return out


def read_seg2(path: str | Path) -> SEG2File:
    """Parse *path* into a :class:`SEG2File`."""
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 32:
        raise SEG2Error(f"{path.name}: too short to be SEG-2")

    if struct.unpack("<H", raw[:2])[0] == FILE_DESCRIPTOR_ID:
        endian = "<"
    elif struct.unpack(">H", raw[:2])[0] == FILE_DESCRIPTOR_ID:
        endian = ">"
    else:
        raise SEG2Error(
            f"{path.name}: bad file descriptor block id "
            f"(expected 0x{FILE_DESCRIPTOR_ID:04X})"
        )

    revision, pointer_bytes, n_traces = struct.unpack(endian + "HHH", raw[2:8])
    term_size = raw[8]
    terminator = bytes(raw[9 : 9 + term_size]) if term_size else b""

    if n_traces == 0:
        raise SEG2Error(f"{path.name}: file declares zero traces")
    if pointer_bytes < 4 * n_traces:
        raise SEG2Error(
            f"{path.name}: trace pointer sub-block is {pointer_bytes} bytes, "
            f"too small for {n_traces} traces"
        )

    pointers = struct.unpack(
        endian + f"{n_traces}I", raw[32 : 32 + 4 * n_traces]
    )
    file_header = _parse_strings(
        raw[32 + pointer_bytes : pointers[0]], endian, terminator
    )

    traces: list[SEG2Trace] = []
    for i, ptr in enumerate(pointers):
        if ptr + 32 > len(raw):
            raise SEG2Error(f"{path.name}: trace {i} pointer runs past end of file")
        (block_id,) = struct.unpack(endian + "H", raw[ptr : ptr + 2])
        if block_id != TRACE_DESCRIPTOR_ID:
            raise SEG2Error(
                f"{path.name}: trace {i} has bad descriptor id 0x{block_id:04X}"
            )
        desc_size, data_bytes, n_samples = struct.unpack(
            endian + "HII", raw[ptr + 2 : ptr + 12]
        )
        code = raw[ptr + 12]
        if code not in _FORMATS:
            raise SEG2Error(
                f"{path.name}: trace {i} uses unsupported data format code {code}"
            )
        dtype_str, width = _FORMATS[code]

        header = _parse_strings(raw[ptr + 32 : ptr + desc_size], endian, terminator)

        start = ptr + desc_size
        available = min(data_bytes, len(raw) - start)
        count = min(n_samples, available // width)
        if count < n_samples:
            # Truncated final record is common when acquisition is interrupted.
            header["_TRUNCATED"] = f"{count}/{n_samples}"
        samples = np.frombuffer(
            raw, dtype=np.dtype(endian + dtype_str), count=count, offset=start
        ).astype(np.float64)
        traces.append(SEG2Trace(header=header, data=samples))

    return SEG2File(
        header=file_header, traces=traces, endian=endian, revision=revision
    )


def header_float(header: dict[str, str], key: str, default: float | None = None):
    """First whitespace-separated token of ``header[key]`` as a float.

    SEG-2 location keys legitimately hold one to three numbers; this pulls the
    scalar case without the caller writing the same guard four times.
    """
    value = header.get(key)
    if value is None:
        return default
    try:
        return float(value.split()[0])
    except (ValueError, IndexError):
        return default


def header_vector(header: dict[str, str], key: str) -> list[float]:
    """All whitespace-separated numeric tokens of ``header[key]``."""
    value = header.get(key)
    if not value:
        return []
    out = []
    for token in value.replace(",", " ").split():
        try:
            out.append(float(token))
        except ValueError:
            break
    return out
