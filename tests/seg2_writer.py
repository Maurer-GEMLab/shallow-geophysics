"""Minimal SEG-2 *writer*, for generating test fixtures.

Not part of the public package -- shallowgeo reads vendor formats, it does not
write them. This exists so the driver tests can exercise a byte-exact SEG-2
file without shipping proprietary sample data, and so a contributor can
reproduce a parser bug without sending a real survey.
"""

from __future__ import annotations

import struct

import numpy as np

TERMINATOR = b"\x00"


def _string_block(entries: dict[str, str]) -> bytes:
    out = b""
    for key, value in entries.items():
        payload = f"{key} {value}".encode("latin-1") + TERMINATOR
        size = 2 + len(payload)
        size += size % 2  # entries are word-aligned
        out += struct.pack("<H", size) + payload.ljust(size - 2, b"\x00")
    return out + struct.pack("<H", 0)


def write_seg2(
    path,
    traces: np.ndarray,
    sample_interval: float,
    *,
    file_header: dict[str, str] | None = None,
    trace_headers: list[dict[str, str]] | None = None,
) -> None:
    """Write ``traces`` (n_traces, n_samples) as 32-bit float SEG-2."""
    traces = np.atleast_2d(np.asarray(traces, dtype="<f4"))
    n_traces, n_samples = traces.shape
    trace_headers = trace_headers or [{} for _ in range(n_traces)]

    file_strings = _string_block(file_header or {})
    pointer_bytes = 4 * n_traces
    pointer_bytes += (4 - pointer_bytes % 4) % 4

    descriptor = bytearray(32 + pointer_bytes + len(file_strings))
    struct.pack_into("<HHHH", descriptor, 0, 0x3A55, 1, pointer_bytes, n_traces)
    descriptor[8] = 1                       # string terminator length
    descriptor[9] = 0                       # string terminator = NUL
    descriptor[32 + pointer_bytes :] = file_strings

    blocks, pointers, cursor = [], [], len(descriptor)
    for i in range(n_traces):
        headers = dict(trace_headers[i])
        headers.setdefault("SAMPLE_INTERVAL", f"{sample_interval:.9f}")
        headers.setdefault("CHANNEL_NUMBER", str(i + 1))
        strings = _string_block(headers)
        desc_size = 32 + len(strings)
        payload = traces[i].tobytes()

        block = bytearray(desc_size)
        struct.pack_into("<HHII", block, 0, 0x4422, desc_size, len(payload), n_samples)
        block[12] = 4                       # 32-bit IEEE float
        block[32:] = strings

        pointers.append(cursor)
        blocks.append(bytes(block) + payload)
        cursor += desc_size + len(payload)

    struct.pack_into(f"<{n_traces}I", descriptor, 32, *pointers)
    with open(path, "wb") as fh:
        fh.write(bytes(descriptor))
        for block in blocks:
            fh.write(block)
