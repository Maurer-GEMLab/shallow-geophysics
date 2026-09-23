"""Geometrics ATOM-1C native ``.atm`` records (passive / ambient noise).

This is the format the node actually writes to its SD card. It is not SEG-2 --
:mod:`shallowgeo.drivers.atom_seg2` handles the SEG-2 *export*, which is a
different file with a different layout. A deployment that has never been
through the vendor's export step looks like this::

    ATOM-1C_PassiveShearData/
        26091816/          <- UTC YYMMDDHH, one folder per clock hour
            10152516.atm   <- <6-digit unit id><2-digit minute>
            10152517.atm
            ...
        26091817/
            10152500.atm

so one node-hour is 60 files and one deployment is several hundred. See
:func:`read_atm_deployment` for the assembly step.

Format
------
The layout below was reverse-engineered from a 1391-file corpus (three
deployments, June-September 2026); Geometrics publishes no specification.
Every claim here is checked in ``tests/test_atom_atm.py`` against that corpus.

A 512-byte header of CRLF-delimited ASCII fields, then the samples. The
header is fixed-width and ends with CRLF at bytes 510-511:

===== ==========================================================
Field Meaning
===== ==========================================================
0     ``Atom`` magic
1     unit id (6 digits), repeated in field 6
2     format version (``2.00`` throughout the corpus)
3     header size in bytes (``0512``)
9     ADC resolution in bits (``24``)
16-17 ``YYYY/MM/DD`` and ``HH:MM:SS`` of the first sample, UTC
===== ==========================================================

then, past the ASCII block, a GPS fix as ``N3755.5010`` / ``W09146.5853`` /
``+0333.29`` -- degrees and decimal *minutes*, not decimal degrees, and the
elevation in metres. Fields 4-5, 7-8 and 10-15 are constant across the whole
corpus, so their meaning could not be determined; they are preserved verbatim
in ``metadata['atm_header_fields']`` rather than guessed at.

Samples are 4 bytes each, but only three of them carry data:

    b0   0x00 or 0xff, the sign extension
    b1   least significant byte      | signed 24-bit,
    b2   middle byte                 | little-endian
    b3   most significant byte       |

The b0 byte is redundant with the sign bit of b3 and is verified on read --
it is the cheapest available check that the stream is aligned, and it catches
a truncated or mis-offset file immediately.

Reading the four bytes as a plain little-endian ``int32`` instead is wrong,
and wrong in a way that is easy to miss. The sign byte lands in the *least*
significant position and pushes the real sample up a byte, so every count
comes out 256 times too large (plus 255 on the negatives, from the sign byte
itself). A uniform factor of 256 leaves waveforms and spectra looking exactly
as they should; only an absolute amplitude gives it away.

Sample rate
-----------
The header does not record it. It is 250 Hz, established from the data
rather than assumed: a full file is 15000 samples and consecutive files in a
node-hour are stamped exactly 60 s apart, and on that basis every partial
file in the corpus ends exactly on the minute boundary where the next one
begins. ``sample_rate`` is still a parameter, because a future firmware or a
differently configured node could change it and silently corrupt every
velocity derived downstream.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.crs import SpatialRef
from ..core.geometry import Geometry
from ..core.survey import SeismicSurvey
from .base import Driver, _sniff

WGS84 = "EPSG:4326"

HEADER_BYTES = 512
ASCII_BYTES = 0xEC
BYTES_PER_SAMPLE = 4
DEFAULT_SAMPLE_RATE = 250.0

_MAGIC = b"Atom"
# lat/lon are degrees + decimal minutes: DDMM.mmmm and DDDMM.mmmm.
_GPS_RE = re.compile(
    rb"([NS]\d{4}\.\d{4})\s*\r\n([EW]\d{5}\.\d{4})\s*\r\n([+-]\d+\.\d+)"
)
# <unit id><minute>.atm, e.g. 10152845.atm -> unit 101525, minute 45
_NAME_RE = re.compile(r"^(?P<unit>\d{6})(?P<minute>\d{2})$")


def _is_atm(path: Path) -> bool:
    return _sniff(path, 8).startswith(_MAGIC)


def _dm_to_degrees(token: str) -> float:
    """``N3755.5010`` -> 37.925017. Degrees and decimal minutes, not decimal degrees.

    Treating the field as decimal degrees is a silent 30-fold error in the
    minutes part that still lands in Missouri, so it would survive a
    plausibility check on the map.
    """
    hemisphere, rest = token[0], token[1:].strip()
    split = 2 if hemisphere in "NS" else 3
    value = float(rest[:split]) + float(rest[split:]) / 60.0
    return -value if hemisphere in "SW" else value


def _parse_header(raw: bytes) -> dict[str, Any]:
    fields = [f.strip() for f in raw[:ASCII_BYTES].decode("latin1").split("\r\n")]
    if not fields or not fields[0].startswith("Atom"):
        raise ValueError("not an ATOM-1C .atm file: missing 'Atom' magic")

    header: dict[str, Any] = {
        "unit_id": fields[1] if len(fields) > 1 else "",
        "format_version": fields[2] if len(fields) > 2 else "",
        "adc_bits": int(fields[9]) if len(fields) > 9 and fields[9].isdigit() else None,
        "atm_header_fields": fields,
    }

    stamp = f"{fields[16]} {fields[17]}" if len(fields) > 17 else ""
    header["start_time"] = pd.to_datetime(
        stamp, format="%Y/%m/%d %H:%M:%S", utc=True, errors="coerce"
    )

    match = _GPS_RE.search(raw)
    if match:
        header["latitude"] = _dm_to_degrees(match.group(1).decode())
        header["longitude"] = _dm_to_degrees(match.group(2).decode())
        header["elevation"] = float(match.group(3))
    else:
        header["latitude"] = header["longitude"] = header["elevation"] = np.nan
    return header


def _decode_samples(payload: bytes, *, check_sign_byte: bool = True) -> np.ndarray:
    """Signed 24-bit little-endian samples, one per 4 bytes.

    Trailing bytes that do not complete a sample are dropped: a node whose
    battery died mid-write leaves a partial sample, and that is not a reason
    to refuse the other 59.9 seconds.
    """
    n = len(payload) // BYTES_PER_SAMPLE
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    raw = np.frombuffer(payload[: n * BYTES_PER_SAMPLE], dtype=np.uint8)
    raw = raw.reshape(n, BYTES_PER_SAMPLE)

    value = (
        raw[:, 1].astype(np.int32)
        | raw[:, 2].astype(np.int32) << 8
        | raw[:, 3].astype(np.int32) << 16
    )
    value = np.where(value >= 1 << 23, value - (1 << 24), value)

    if check_sign_byte:
        expected = np.where(value < 0, 0xFF, 0x00).astype(np.uint8)
        bad = int(np.count_nonzero(raw[:, 0] != expected))
        if bad:
            raise ValueError(
                f"{bad} of {n} samples have a sign byte that disagrees with the "
                "sample value. The file is truncated, mis-aligned, or not a "
                "24-bit ATOM-1C record."
            )
    return value.astype(np.float64)


def read_atm_file(path: str | Path, *, check_sign_byte: bool = True) -> dict[str, Any]:
    """Header fields and raw counts for one ``.atm`` file, without geometry.

    The low-level entry point. :func:`read_atm` wraps it in a
    :class:`~shallowgeo.core.survey.SeismicSurvey`; QC code that only wants
    headers is better served here, because building a survey per minute-file
    over a 600-file deployment is most of the cost of a scan.
    """
    path = Path(path)
    blob = path.read_bytes()
    if len(blob) < HEADER_BYTES:
        raise ValueError(f"{path.name}: shorter than a {HEADER_BYTES}-byte header")

    header = _parse_header(blob[:HEADER_BYTES])
    header["data"] = _decode_samples(blob[HEADER_BYTES:], check_sign_byte=check_sign_byte)
    header["source_file"] = str(path)

    # The unit id is in the header, but the filename carries it too and the
    # two disagreeing means the file was renamed -- worth surfacing.
    name = _NAME_RE.match(path.stem)
    if name and header["unit_id"] and name.group("unit") != header["unit_id"]:
        header["unit_id_from_filename"] = name.group("unit")
    return header


def read_atm(
    path: str | Path,
    *,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    position: tuple[float, float, float] | None = None,
    spatial_ref: SpatialRef | None = None,
    check_sign_byte: bool = True,
) -> SeismicSurvey:
    """Read one ``.atm`` minute-file as a single-channel passive record."""
    path = Path(path)
    header = read_atm_file(path, check_sign_byte=check_sign_byte)
    node = header["unit_id"] or path.stem

    if position is not None:
        lon, lat, elev = position
    else:
        lon, lat, elev = header["longitude"], header["latitude"], header["elevation"]
    if not np.isfinite(lon) or not np.isfinite(lat):
        raise ValueError(
            f"{path.name}: no GPS fix in the header. Pass position=(lon, lat, elev)."
        )

    geometry = Geometry(
        ids=[node],
        x=[lon],
        y=[lat],
        z=[elev if np.isfinite(elev) else 0.0],
        roles=["receiver"],
        spatial_ref=SpatialRef(WGS84, vertical_datum="ellipsoidal"),
    )
    target = spatial_ref or geometry.spatial_ref
    if target.crs != geometry.spatial_ref.crs:
        geometry = geometry.to_crs(target)

    survey = SeismicSurvey(
        data=header["data"][np.newaxis, :],
        sample_interval=1.0 / sample_rate,
        geometry=geometry,
        # No source_id column: this is what marks the survey passive.
        trace_map=pd.DataFrame({"receiver_id": [node], "channel": ["Z"]}),
        start_time=header["start_time"],
        metadata={
            "instrument": "Geometrics ATOM-1C",
            "format": "ATOM native .atm",
            "node_id": node,
            "adc_bits": header["adc_bits"],
            "units": "counts",
            "atm_header_fields": header["atm_header_fields"],
            "source_file": str(path),
        },
    )
    survey.provenance.record("read", driver="atom-atm", path=str(path), node_id=node)
    return survey


# -- deployment assembly ----------------------------------------------------


def scan_atm(paths: Iterable[str | Path], *, sample_rate: float = DEFAULT_SAMPLE_RATE):
    """One row per file: node, time, position, length and amplitude.

    Headers and counts only -- no merging, no resampling. This is the table
    the QC step works from, and it is deliberately cheap enough to run over a
    whole zip before deciding which nodes and which hours are worth loading.
    """
    rows = []
    for path in sorted(Path(p) for p in paths):
        try:
            header = read_atm_file(path, check_sign_byte=False)
        except ValueError as exc:  # keep going; report at the end
            rows.append({"path": str(path), "node": "", "error": str(exc)})
            continue
        data = header["data"]
        rows.append(
            {
                "path": str(path),
                "folder": path.parent.name,
                "node": header["unit_id"],
                "start_time": header["start_time"],
                "n_samples": data.size,
                "duration": data.size / sample_rate,
                "latitude": header["latitude"],
                "longitude": header["longitude"],
                "elevation": header["elevation"],
                "rms": float(np.std(data)) if data.size else np.nan,
                "abs_max": float(np.abs(data).max()) if data.size else np.nan,
                "error": "",
            }
        )
    frame = pd.DataFrame(rows)
    if "start_time" in frame:
        frame["end_time"] = frame["start_time"] + pd.to_timedelta(
            frame["duration"], unit="s"
        )
    return frame


def group_deployments(scan: pd.DataFrame, *, max_gap: str | pd.Timedelta = "6h"):
    """Split a file inventory into separate field deployments.

    The hourly ``YYMMDDHH`` folders are a storage detail, not a survey
    boundary: one afternoon's recording routinely spans three of them, and
    merging per folder would cut a continuous record into hour-long pieces
    with nothing to gain. Equally, two folders named for the same hour on
    different months are not one survey.

    So the split is made on time rather than on folder names. Files are sorted
    by start time and a new deployment begins wherever the gap to the previous
    file exceeds ``max_gap``. Six hours is long enough to bridge a lunch break
    or a battery swap and far too short to bridge two field days.
    """
    if scan.empty:
        return pd.DataFrame(
            columns=["deployment", "date", "n_files", "n_nodes", "nodes", "start", "end"]
        )
    gap = pd.Timedelta(max_gap)
    frame = scan.dropna(subset=["start_time"]).sort_values("start_time").copy()
    frame["deployment"] = (frame["start_time"].diff() > gap).cumsum()

    rows = []
    for ident, part in frame.groupby("deployment"):
        nodes = sorted(part["node"].unique())
        rows.append(
            {
                "deployment": int(ident),
                "date": part["start_time"].min().strftime("%Y-%m-%d"),
                "n_files": len(part),
                "n_nodes": len(nodes),
                "nodes": nodes,
                "start": part["start_time"].min(),
                "end": part["end_time"].max(),
                "duration_min": float(
                    (part["end_time"].max() - part["start_time"].min()).total_seconds()
                    / 60.0
                ),
                "folders": sorted(part["folder"].unique()),
            }
        )
    return pd.DataFrame(rows)


def read_atm_deployment(
    source: str | Path | Iterable[str | Path],
    *,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    nodes: Iterable[str] | None = None,
    start: Any = None,
    end: Any = None,
    positions: dict[str, tuple[float, float, float]] | None = None,
    spatial_ref: SpatialRef | None = None,
    check_sign_byte: bool = True,
) -> SeismicSurvey:
    """Assemble a whole deployment into one time-aligned passive survey.

    ``source`` is a directory -- searched recursively, so the hourly
    ``YYMMDDHH`` folders are merged automatically -- or an explicit iterable
    of files.

    Records are placed on one absolute-time axis by GPS timestamp rather than
    concatenated in filename order. The two are not the same thing: nodes
    start at different minutes, a node that stops early leaves a hole, and a
    deployment can span two UTC hours and therefore two folders. Samples with
    no data are ``NaN``, never zero, so that a gap is visible to the
    processing rather than acting as a very loud impulse at every gap edge.

    The returned survey covers the window where *at least one* node has data;
    use :func:`shallowgeo.passive.common_window` to trim to the interval where
    all of them do, which is what dispersion processing needs.
    """
    if isinstance(source, (str, Path)) and Path(source).is_dir():
        paths = sorted(Path(source).rglob("*.atm"))
    elif isinstance(source, (str, Path)):
        paths = [Path(source)]
    else:
        paths = sorted(Path(p) for p in source)
    # __MACOSX/._name AppleDouble stubs unzip alongside the real files and
    # are 212-byte resource forks, not records.
    paths = [p for p in paths if not p.name.startswith("._")]
    if not paths:
        raise ValueError(f"no .atm files found under {source!r}")

    records = []
    for path in paths:
        header = read_atm_file(path, check_sign_byte=check_sign_byte)
        if nodes is not None and header["unit_id"] not in set(nodes):
            continue
        if pd.isna(header["start_time"]):
            raise ValueError(
                f"{path.name}: no ACQUISITION date/time, so it cannot be placed on "
                "the absolute time axis that passive processing requires."
            )
        records.append(header)
    if not records:
        raise ValueError("no .atm files matched the requested nodes")

    dt = 1.0 / sample_rate
    starts = [r["start_time"] for r in records]
    ends = [
        r["start_time"] + pd.Timedelta(seconds=r["data"].size * dt) for r in records
    ]
    t0 = pd.Timestamp(start, tz="UTC") if start is not None else min(starts)
    t1 = pd.Timestamp(end, tz="UTC") if end is not None else max(ends)
    n_samples = int(round((t1 - t0).total_seconds() * sample_rate))
    if n_samples <= 0:
        raise ValueError(f"empty time window {t0} to {t1}")

    node_ids = sorted({r["unit_id"] for r in records})
    index = {node: i for i, node in enumerate(node_ids)}
    data = np.full((len(node_ids), n_samples), np.nan)
    for r in records:
        offset = int(round((r["start_time"] - t0).total_seconds() * sample_rate))
        chunk = r["data"]
        lo, hi = max(offset, 0), min(offset + chunk.size, n_samples)
        if hi <= lo:
            continue
        data[index[r["unit_id"]], lo:hi] = chunk[lo - offset : hi - offset]

    positions = positions or {}
    lon, lat, elev = [], [], []
    for node in node_ids:
        if node in positions:
            x, y, z = positions[node]
        else:
            fixes = [r for r in records if r["unit_id"] == node]
            # Median, not mean: a handful of fixes per deployment land a
            # kilometre away and would drag a mean well off the station.
            x = float(np.nanmedian([f["longitude"] for f in fixes]))
            y = float(np.nanmedian([f["latitude"] for f in fixes]))
            z = float(np.nanmedian([f["elevation"] for f in fixes]))
        lon.append(x), lat.append(y), elev.append(z)

    geometry = Geometry(
        ids=node_ids,
        x=lon,
        y=lat,
        z=elev,
        roles=["receiver"] * len(node_ids),
        spatial_ref=SpatialRef(WGS84, vertical_datum="ellipsoidal"),
    )
    if spatial_ref is not None and spatial_ref.crs != geometry.spatial_ref.crs:
        geometry = geometry.to_crs(spatial_ref)

    survey = SeismicSurvey(
        data=data,
        sample_interval=dt,
        geometry=geometry,
        trace_map=pd.DataFrame({"receiver_id": node_ids, "channel": "Z"}),
        start_time=t0,
        metadata={
            "instrument": "Geometrics ATOM-1C",
            "format": "ATOM native .atm",
            "units": "counts",
            "n_nodes": len(node_ids),
            "n_files": len(records),
            "folders": sorted({Path(r["source_file"]).parent.name for r in records}),
            "coverage": {
                node: float(
                    np.isfinite(data[index[node]]).sum() / n_samples
                )
                for node in node_ids
            },
        },
    )
    survey.provenance.record(
        "read_deployment",
        driver="atom-atm",
        n_files=len(records),
        n_nodes=len(node_ids),
        window_start=str(t0),
        window_end=str(t1),
        sample_rate=sample_rate,
    )
    return survey


driver = Driver(
    name="atom-atm",
    description="Geometrics ATOM-1C native .atm (passive surface wave)",
    can_open=_is_atm,
    read=read_atm,
    extensions=(".atm",),
    methods=("passive_masw", "ambient_noise"),
    vendor="Geometrics",
    instrument="ATOM-1C",
    notes=(
        "One file per node per minute, in UTC YYMMDDHH folders. "
        "Use read_atm_deployment() to merge and time-align a deployment."
    ),
)
