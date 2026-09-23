"""Minimal ATOM-1C ``.atm`` *writer*, for generating test fixtures.

Like ``seg2_writer`` and ``segy_writer``, this is a test helper and not part
of the public package. It reproduces the layout the node writes: a 512-byte
CRLF-delimited ASCII header, a GPS fix in degrees and decimal minutes, then
signed 24-bit little-endian samples each padded to four bytes with a leading
sign-extension byte.

Writing the fixture from an independent implementation of the spec is the
point. If the reader and the writer shared a codec, a test round-trip would
only prove they agree with each other, which is exactly the thing that was
uncertain while the format was being worked out.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HEADER_BYTES = 512


def degrees_to_dm(value: float, *, is_latitude: bool) -> str:
    """37.925017 -> ``N3755.5010``."""
    hemisphere = ("N" if value >= 0 else "S") if is_latitude else (
        "E" if value >= 0 else "W"
    )
    magnitude = abs(value)
    degrees = int(magnitude)
    minutes = (magnitude - degrees) * 60.0
    width = 2 if is_latitude else 3
    return f"{hemisphere}{degrees:0{width}d}{minutes:07.4f}"


def write_atm(
    path: str | Path,
    data: np.ndarray,
    *,
    unit_id: str = "101525",
    start: str = "2026/09/18 17:00:00",
    latitude: float = 37.925017,
    longitude: float = -91.776422,
    elevation: float = 333.29,
    adc_bits: int = 24,
    version: str = "2.00",
    corrupt_sign_byte: bool = False,
) -> Path:
    """Write ``data`` (integer counts) as one ``.atm`` minute-file."""
    path = Path(path)
    date, time = start.split()

    fields = [
        "Atom  ", unit_id, version, f"{HEADER_BYTES:04d}", "0016", "00",
        unit_id, "1", "1", f"{adc_bits:02d}", "00000",
        "004", "008", "008", "004", "OFF", date, time,
        " " * 62, " " * 62,
    ]
    ascii_block = "\r\n".join(fields).encode("latin1")

    gps = (
        degrees_to_dm(latitude, is_latitude=True).encode() + b" \r\n"
        + degrees_to_dm(longitude, is_latitude=False).encode() + b" \r\n"
        + f"{elevation:+08.2f}".encode() + b"\r\n"
    )

    header = bytearray(b"\x00" * HEADER_BYTES)
    header[: len(ascii_block)] = ascii_block
    # The real files place the GPS block just past the ASCII fields; the exact
    # offset does not matter to the reader, which searches for the pattern.
    header[0xEC : 0xEC + len(gps)] = gps
    header[HEADER_BYTES - 2 :] = b"\r\n"

    values = np.asarray(data, dtype=np.int64)
    if np.any(values >= 1 << 23) or np.any(values < -(1 << 23)):
        raise ValueError("sample outside the 24-bit range")

    unsigned = np.where(values < 0, values + (1 << 24), values).astype(np.uint32)
    packed = np.zeros((values.size, 4), dtype=np.uint8)
    packed[:, 0] = np.where(values < 0, 0xFF, 0x00)
    packed[:, 1] = unsigned & 0xFF
    packed[:, 2] = (unsigned >> 8) & 0xFF
    packed[:, 3] = (unsigned >> 16) & 0xFF
    if corrupt_sign_byte and values.size:
        packed[0, 0] ^= 0xFF

    path.write_bytes(bytes(header) + packed.tobytes())
    return path
