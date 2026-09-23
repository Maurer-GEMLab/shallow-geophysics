"""Minimal SEG-Y *writer*, for generating test fixtures.

Like ``seg2_writer``, not part of the public package. Produces the shape of
file the Geometrics SeisModule Controller exports: rev 0, big-endian, IBM
float by default, along-line distances in ``source_x`` / ``group_x``, and a
measurement-system flag in the binary header.
"""

from __future__ import annotations

import struct

import numpy as np


def float_to_ibm(values: np.ndarray) -> np.ndarray:
    """float64 -> IBM single-precision bit patterns (uint32)."""
    v = np.asarray(values, dtype=np.float64)
    out = np.zeros(v.shape, dtype=np.uint32)
    nonzero = v != 0
    a = np.abs(v[nonzero])
    # a = fraction * 16**exponent with fraction in [1/16, 1)
    exponent = np.floor(np.log2(a) / 4.0).astype(np.int64) + 1
    fraction = a / np.power(16.0, exponent)
    mant = np.round(fraction * 16777216.0).astype(np.uint64)
    # rounding can push the fraction to exactly 1.0
    overflow = mant >= 16777216
    mant[overflow] //= 16
    exponent[overflow] += 1
    sign = (v[nonzero] < 0).astype(np.uint32) << 31
    out[nonzero] = sign | ((exponent + 64).astype(np.uint32) << 24) | mant.astype(np.uint32)
    return out


def write_segy(
    path,
    traces: np.ndarray,
    sample_interval: float,
    *,
    group_x: list[float] | None = None,
    source_x: float = 0.0,
    measurement_system: int = 2,
    format_code: int = 1,
    field_record: int = 1,
    delay_ms: int = 0,
    text: str = "GEOMETRICS SEISMODULE CONTROLLER",
    coordinate_scalar: int = 1,
    year: int = 26,
    day_of_year: int = 254,
    hms: tuple[int, int, int] = (13, 19, 34),
    endian: str = ">",
) -> None:
    traces = np.atleast_2d(np.asarray(traces, dtype=np.float64))
    n_traces, n_samples = traces.shape
    group_x = list(group_x) if group_x is not None else [4.0 * i for i in range(n_traces)]

    lines = [f"C{i + 1:2d} " + (text if i == 1 else "") for i in range(40)]
    text_header = "".join(line.ljust(80)[:80] for line in lines).encode("cp500")

    bh = bytearray(400)
    struct.pack_into(endian + "h", bh, 3213 - 3201, n_traces)
    struct.pack_into(endian + "H", bh, 3217 - 3201, int(round(sample_interval * 1e6)))
    struct.pack_into(endian + "H", bh, 3221 - 3201, n_samples)
    struct.pack_into(endian + "h", bh, 3225 - 3201, format_code)
    struct.pack_into(endian + "h", bh, 3255 - 3201, measurement_system)

    body = bytearray()
    for i in range(n_traces):
        th = bytearray(240)
        struct.pack_into(endian + "i", th, 8, field_record)
        struct.pack_into(endian + "i", th, 12, i + 1)
        struct.pack_into(endian + "h", th, 70, coordinate_scalar)
        struct.pack_into(endian + "i", th, 72, int(round(source_x)))
        struct.pack_into(endian + "i", th, 80, int(round(group_x[i])))
        struct.pack_into(endian + "h", th, 108, delay_ms)
        struct.pack_into(endian + "H", th, 114, n_samples)
        struct.pack_into(endian + "H", th, 116, int(round(sample_interval * 1e6)))
        struct.pack_into(endian + "h", th, 156, year)
        struct.pack_into(endian + "h", th, 158, day_of_year)
        struct.pack_into(endian + "hhh", th, 160, *hms)
        body += th
        if format_code == 1:
            body += float_to_ibm(traces[i]).astype(endian + "u4").tobytes()
        elif format_code == 5:
            body += traces[i].astype(endian + "f4").tobytes()
        elif format_code == 2:
            body += traces[i].astype(endian + "i4").tobytes()
        elif format_code == 3:
            body += traces[i].astype(endian + "i2").tobytes()
        else:
            raise ValueError(f"writer does not support format code {format_code}")

    with open(path, "wb") as fh:
        fh.write(text_header)
        fh.write(bytes(bh))
        fh.write(bytes(body))
