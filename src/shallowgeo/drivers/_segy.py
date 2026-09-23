"""Minimal SEG-Y reader (SEG Technical Standards Committee, rev 0 / rev 1).

Implemented here rather than delegated to ObsPy for the same reasons as
:mod:`._seg2`: it keeps the readers importable without ObsPy, and it hands
back the raw header fields so the Geometrics-specific interpretation --
which trace-header bytes actually carry the spread geometry, and in what
unit -- is done in one visible place.

Scope is deliberately narrow: fixed-length traces, one data format per file,
the standard 240-byte trace header. That covers every land-seismic engineering
export we have seen. Extended textual headers (rev 1) are skipped, not parsed.

Byte positions below are 1-based as in the standard's tables, converted to
0-based offsets in code.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TEXT_HEADER_BYTES = 3200
BINARY_HEADER_BYTES = 400
TRACE_HEADER_BYTES = 240

#: SEG-Y data sample format code -> (numpy dtype suffix or "ibm", bytes/sample)
_FORMATS = {
    1: ("ibm", 4),   # 4-byte IBM floating point
    2: ("i4", 4),    # 4-byte two's complement integer
    3: ("i2", 2),    # 2-byte two's complement integer
    5: ("f4", 4),    # 4-byte IEEE floating point
    6: ("f8", 8),    # 8-byte IEEE floating point (rev 2)
    8: ("i1", 1),    # 1-byte two's complement integer
}

#: Binary-header measurement system code (bytes 3255-3256).
MEASUREMENT_SYSTEM = {1: "m", 2: "ft"}

#: Trace-header fields we read: name -> (1-based byte, struct code).
_TRACE_FIELDS = {
    "trace_sequence_line": (1, "i"),
    "trace_sequence_file": (5, "i"),
    "field_record": (9, "i"),
    "trace_number": (13, "i"),
    "source_point": (17, "i"),
    "trace_id_code": (29, "h"),
    "offset": (37, "i"),
    "receiver_elevation": (41, "i"),
    "source_elevation": (45, "i"),
    "source_depth": (49, "i"),
    "elevation_scalar": (69, "h"),
    "coordinate_scalar": (71, "h"),
    "source_x": (73, "i"),
    "source_y": (77, "i"),
    "group_x": (81, "i"),
    "group_y": (85, "i"),
    "coordinate_units": (89, "h"),
    "delay_ms": (109, "h"),
    "n_samples": (115, "H"),
    "sample_interval_us": (117, "H"),
    "gain_type": (119, "h"),
    "instrument_gain_db": (121, "h"),
    "year": (157, "h"),
    "day_of_year": (159, "h"),
    "hour": (161, "h"),
    "minute": (163, "h"),
    "second": (165, "h"),
    "time_basis": (167, "h"),
}


class SEGYError(ValueError):
    """Malformed or unsupported SEG-Y content."""


@dataclass
class SEGYTrace:
    header: dict[str, int]
    data: np.ndarray


@dataclass
class SEGYFile:
    """A parsed SEG-Y file, uninterpreted."""

    text_header: str
    binary_header: dict[str, int]
    traces: list[SEGYTrace] = field(default_factory=list)
    endian: str = ">"

    @property
    def n_traces(self) -> int:
        return len(self.traces)

    @property
    def sample_interval(self) -> float:
        """Seconds, from the binary header."""
        return self.binary_header["sample_interval_us"] * 1e-6

    @property
    def measurement_unit(self) -> str | None:
        return MEASUREMENT_SYSTEM.get(self.binary_header.get("measurement_system", 0))


def _decode_text_header(raw: bytes) -> str:
    """EBCDIC unless it is plainly ASCII (rev 1 allows either)."""
    if raw[:1] in (b"C", b"c", b" "):
        return raw.decode("latin-1")
    try:
        return raw.decode("cp500")
    except UnicodeDecodeError:  # pragma: no cover - cp500 maps every byte
        return raw.decode("latin-1", errors="replace")


def _binary_header(raw: bytes, endian: str) -> dict[str, int]:
    def u(code: str, byte: int) -> int:
        pos = byte - 3201
        return struct.unpack(endian + code, raw[pos : pos + struct.calcsize(code)])[0]

    return {
        "job_id": u("i", 3201),
        "line_number": u("i", 3205),
        "reel_number": u("i", 3209),
        "traces_per_ensemble": u("h", 3213),
        "aux_traces_per_ensemble": u("h", 3215),
        "sample_interval_us": u("H", 3217),
        "n_samples": u("H", 3221),
        "format_code": u("h", 3225),
        "ensemble_fold": u("h", 3227),
        "trace_sorting": u("h", 3229),
        "measurement_system": u("h", 3255),
        "revision": u("H", 3501),
        "fixed_length_traces": u("h", 3503),
        "n_extended_text_headers": u("h", 3505),
    }


def _plausible(bh: dict[str, int]) -> bool:
    return bh["format_code"] in _FORMATS and 0 < bh["n_samples"] <= 65535


def looks_like_segy(data: bytes) -> bool:
    """Whether *data* (>= 3600 bytes) has a credible binary header.

    SEG-Y has no magic number, so this is a plausibility test: a known data
    format code and a sane sample count in either byte order.
    """
    if len(data) < TEXT_HEADER_BYTES + BINARY_HEADER_BYTES:
        return False
    bh_raw = data[TEXT_HEADER_BYTES : TEXT_HEADER_BYTES + BINARY_HEADER_BYTES]
    return any(_plausible(_binary_header(bh_raw, e)) for e in (">", "<"))


def ibm_to_float(raw: np.ndarray) -> np.ndarray:
    """Vectorised IBM System/360 single-precision -> float64.

    Layout: 1 sign bit, 7-bit excess-64 base-16 exponent, 24-bit fraction with
    no hidden bit. Zero is all-zero bits.
    """
    u = raw.astype(np.uint32)
    sign = np.where(u >> 31, -1.0, 1.0)
    exponent = ((u >> 24) & 0x7F).astype(np.int64) - 64
    fraction = (u & 0x00FFFFFF).astype(np.float64) / 16777216.0
    out = sign * fraction * np.power(16.0, exponent)
    return np.where(u == 0, 0.0, out)


def _trace_header(raw: bytes, endian: str) -> dict[str, int]:
    out = {}
    for name, (byte, code) in _TRACE_FIELDS.items():
        pos = byte - 1
        out[name] = struct.unpack(endian + code, raw[pos : pos + struct.calcsize(code)])[0]
    return out


def read_segy(path: str | Path) -> SEGYFile:
    """Parse *path* into a :class:`SEGYFile`."""
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < TEXT_HEADER_BYTES + BINARY_HEADER_BYTES + TRACE_HEADER_BYTES:
        raise SEGYError(f"{path.name}: too short to be SEG-Y")

    bh_raw = raw[TEXT_HEADER_BYTES : TEXT_HEADER_BYTES + BINARY_HEADER_BYTES]
    for endian in (">", "<"):
        bh = _binary_header(bh_raw, endian)
        if _plausible(bh):
            break
    else:
        raise SEGYError(
            f"{path.name}: binary header has no recognisable data format code "
            "in either byte order"
        )

    kind, width = _FORMATS[bh["format_code"]]
    ns_file = bh["n_samples"]
    cursor = TEXT_HEADER_BYTES + BINARY_HEADER_BYTES
    cursor += max(bh["n_extended_text_headers"], 0) * TEXT_HEADER_BYTES

    traces: list[SEGYTrace] = []
    while cursor + TRACE_HEADER_BYTES <= len(raw):
        header = _trace_header(raw[cursor : cursor + TRACE_HEADER_BYTES], endian)
        ns = header["n_samples"] or ns_file
        start = cursor + TRACE_HEADER_BYTES
        available = (len(raw) - start) // width
        count = min(ns, available)
        if count <= 0:
            break
        if count < ns:
            header["_truncated"] = count
        if kind == "ibm":
            samples = ibm_to_float(
                np.frombuffer(raw, dtype=endian + "u4", count=count, offset=start)
            )
        else:
            samples = np.frombuffer(
                raw, dtype=np.dtype(endian + kind), count=count, offset=start
            ).astype(np.float64)
        traces.append(SEGYTrace(header=header, data=samples))
        cursor = start + ns * width

    if not traces:
        raise SEGYError(f"{path.name}: no traces after the headers")

    return SEGYFile(
        text_header=_decode_text_header(raw[:TEXT_HEADER_BYTES]),
        binary_header=bh,
        traces=traces,
        endian=endian,
    )


def apply_scalar(value: int, scalar: int) -> float:
    """SEG-Y coordinate/elevation scalar: positive multiplies, negative divides."""
    if scalar == 0:
        return float(value)
    return float(value) * scalar if scalar > 0 else float(value) / -scalar
