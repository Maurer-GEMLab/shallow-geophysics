"""Geometrics ATOM-1C nodal recorder (passive surface-wave / ambient noise).

The ATOM-1C writes a proprietary ``.ATM``, ASCII, or SEG-2; this driver reads
the SEG-2 export, which is the one documented path off the instrument. Three
things differ from a Geode record and drive the code below.

Each node is an independent single-channel recorder with its own GPS, so a
deployment is *many files*, one per node, and geometry comes from real
coordinates rather than an assumed straight spread. :func:`read_atom_array`
assembles a deployment; the registry-facing :func:`read_atom` handles one file.

Recording is passive and continuous, so there is no source. The resulting
``SeismicSurvey`` has no ``source_id`` column and reports ``is_passive``.

Nodes free-run and are aligned by GPS time, so start times differ between
files by up to a sample. Alignment is done on absolute time, not by assuming a
common t=0 -- getting this wrong quietly destroys the cross-correlations that
passive surface-wave processing depends on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.crs import SpatialRef
from ..core.geometry import Geometry
from ..core.survey import SeismicSurvey
from ._seg2 import SEG2File, header_float, header_vector, looks_like_seg2, read_seg2
from .base import Driver, _sniff

_ATOM_MARKERS = (b"ATOM", b"ATOM-1C")

WGS84 = "EPSG:4326"


def _is_atom(path: Path) -> bool:
    data = _sniff(path, 4096)
    if not looks_like_seg2(data):
        return False
    return any(m in data.upper() for m in _ATOM_MARKERS)


def _node_id(seg2: SEG2File, path: Path) -> str:
    """Serial number if the headers carry one, else the filename stem."""
    for source in (seg2.header, seg2.traces[0].header):
        for key in ("UNIT_NUMBER", "SERIAL_NUMBER", "STATION_CODE", "TRACE_ID"):
            value = source.get(key)
            if value:
                return str(value).strip()
    return path.stem


def _node_position(seg2: SEG2File) -> tuple[float, float, float] | None:
    """``(lon, lat, elev)`` from GPS headers, or None if unpositioned.

    SEG-2 has no standard GPS key, so several spellings are tried. Note the
    ordering trap: ``RECEIVER_LOCATION`` on a GPS-equipped node is written
    ``lat lon elev``, which is the opposite of the ``x y z`` order everything
    downstream expects.
    """
    for source in (seg2.traces[0].header, seg2.header):
        for key in ("RECEIVER_GPS", "GPS_POSITION", "RECEIVER_LOCATION"):
            values = header_vector(source, key)
            if len(values) >= 2:
                lat, lon = values[0], values[1]
                elev = values[2] if len(values) > 2 else 0.0
                if abs(lat) <= 90 and abs(lon) <= 180 and (lat or lon):
                    return (lon, lat, elev)
    return None


def _start_time(seg2: SEG2File):
    date = seg2.header.get("ACQUISITION_DATE") or ""
    time = seg2.header.get("ACQUISITION_TIME") or ""
    stamp = f"{date} {time}".strip()
    if not stamp:
        return None
    return pd.to_datetime(stamp, errors="coerce", dayfirst=False)


def read_atom(
    path: str | Path,
    *,
    position: tuple[float, float, float] | None = None,
    spatial_ref: SpatialRef | None = None,
    node_id: str | None = None,
) -> SeismicSurvey:
    """Read one ATOM-1C node file as a single-channel passive record."""
    path = Path(path)
    seg2 = read_seg2(path)
    trace = seg2.traces[0]

    dt = header_float(trace.header, "SAMPLE_INTERVAL") or header_float(
        seg2.header, "SAMPLE_INTERVAL"
    )
    if not dt:
        raise ValueError(f"{path.name}: no SAMPLE_INTERVAL in headers")

    ident = node_id or _node_id(seg2, path)
    pos = position or _node_position(seg2)
    if pos is None:
        raise ValueError(
            f"{path.name}: no GPS position in headers. "
            "Pass position=(lon, lat, elevation)."
        )
    sref = spatial_ref or SpatialRef(WGS84, vertical_datum="ellipsoidal")

    geometry = Geometry(
        ids=[ident],
        x=[pos[0]],
        y=[pos[1]],
        z=[pos[2]],
        roles=["receiver"],
        spatial_ref=SpatialRef(WGS84, vertical_datum="ellipsoidal"),
    )
    if sref.crs != geometry.spatial_ref.crs:
        geometry = geometry.to_crs(sref)

    survey = SeismicSurvey(
        data=trace.data[np.newaxis, :],
        sample_interval=dt,
        geometry=geometry,
        # No source_id column: this is what marks the survey passive.
        trace_map=pd.DataFrame({"receiver_id": [ident], "channel": ["Z"]}),
        start_time=_start_time(seg2),
        metadata={
            "instrument": "Geometrics ATOM-1C",
            "format": "SEG-2",
            "node_id": ident,
            "seg2_file_header": seg2.header,
            "source_file": str(path),
        },
    )
    survey.provenance.record("read", driver="atom-seg2", path=str(path), node_id=ident)
    return survey


def read_atom_array(
    paths,
    *,
    spatial_ref: SpatialRef | None = None,
    positions: dict[str, tuple[float, float, float]] | None = None,
) -> SeismicSurvey:
    """Assemble many node files into one passive deployment.

    Records are trimmed to their common time window using GPS start times, so
    the returned array is genuinely synchronous. Nodes whose recording does not
    overlap the others are dropped with a warning rather than silently
    zero-padded.

    Parameters
    ----------
    positions
        Optional ``{node_id: (lon, lat, elev)}`` override for nodes whose GPS
        did not fix.
    """
    import warnings

    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("no files given")
    positions = positions or {}

    singles = []
    for p in paths:
        ident = _node_id(read_seg2(p), p)
        singles.append(read_atom(p, position=positions.get(ident)))

    dts = {s.sample_interval for s in singles}
    if len(dts) > 1:
        raise ValueError(f"nodes have mismatched sample intervals: {sorted(dts)}")
    dt = singles[0].sample_interval

    if any(s.start_time is None or pd.isna(s.start_time) for s in singles):
        raise ValueError(
            "at least one node has no ACQUISITION_DATE/TIME, so the records "
            "cannot be time-aligned. Passive processing requires absolute time."
        )

    starts = [s.start_time for s in singles]
    ends = [s.start_time + pd.Timedelta(seconds=s.duration) for s in singles]
    window_start, window_end = max(starts), min(ends)
    if window_end <= window_start:
        raise ValueError("node recordings do not share a common time window")

    n_samples = int((window_end - window_start).total_seconds() / dt)
    rows, keep = [], []
    for s in singles:
        offset = int(round((window_start - s.start_time).total_seconds() / dt))
        if offset + n_samples > s.n_samples:
            warnings.warn(
                f"node {s.metadata['node_id']} is shorter than the common "
                "window and was dropped",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        rows.append(s.data[0, offset : offset + n_samples])
        keep.append(s)

    if not rows:
        raise ValueError("no nodes survived time alignment")

    sref = spatial_ref or keep[0].spatial_ref
    geometry = keep[0].geometry.to_crs(sref)
    for s in keep[1:]:
        geometry = geometry.merge(s.geometry)

    survey = SeismicSurvey(
        data=np.vstack(rows),
        sample_interval=dt,
        geometry=geometry,
        trace_map=pd.DataFrame(
            {
                "receiver_id": [s.metadata["node_id"] for s in keep],
                "channel": "Z",
            }
        ),
        start_time=window_start,
        metadata={
            "instrument": "Geometrics ATOM-1C",
            "format": "SEG-2",
            "n_nodes": len(keep),
            "source_files": [str(p) for p in paths],
        },
    )
    survey.provenance.record(
        "read_array",
        driver="atom-seg2",
        n_files=len(paths),
        n_nodes=len(keep),
        window_start=str(window_start),
        window_end=str(window_end),
    )
    return survey


driver = Driver(
    name="atom-seg2",
    description="Geometrics ATOM-1C nodal SEG-2 (passive surface wave)",
    can_open=_is_atom,
    read=read_atom,
    extensions=(".dat", ".sg2", ".seg2", ".atm"),
    methods=("passive_masw", "ambient_noise"),
    vendor="Geometrics",
    instrument="ATOM-1C",
    notes=(
        "One file per node. Use read_atom_array() to assemble and "
        "GPS-time-align a deployment."
    ),
)
