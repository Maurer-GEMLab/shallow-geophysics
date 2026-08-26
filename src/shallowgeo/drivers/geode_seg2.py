"""Geometrics Geode / Geometrics-written SEG-2 (refraction, active MASW).

The Geode records SEG-2 with ``RECEIVER_LOCATION`` and ``SOURCE_LOCATION``
written as a *scalar distance along the spread*, not a coordinate triple, and
only if the operator entered geometry in the field software. In practice a
large fraction of teaching-lab files carry all-zero locations, so this driver
treats file geometry as a hint and lets the caller override it::

    read("LINE1.DAT", spacing=2.0, source_offset=-1.0)

The along-line distances are placed in a real CRS. With no ``spatial_ref``
given the driver builds a local metric grid, so downstream code can assume a
CRS always exists without pretending to know where on Earth the line was.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.crs import SpatialRef, local_grid
from ..core.geometry import Geometry
from ..core.survey import SeismicSurvey
from ._seg2 import SEG2File, header_float, looks_like_seg2, read_seg2
from .base import Driver, _sniff

#: Written by Geometrics field software into the SEG-2 file header.
_GEOMETRICS_MARKERS = ("GEOMETRICS", "GEODE", "STRATAVISUAL", "SEISMODULE")


def _is_geometrics(path: Path) -> bool:
    data = _sniff(path, 4096)
    if not looks_like_seg2(data):
        return False
    upper = data.upper()
    if any(m.encode() in upper for m in _GEOMETRICS_MARKERS):
        return True
    # An unbranded SEG-2 is still most likely a Geode in this context; claim it
    # weakly so a bare file is readable, and let atom_seg2 outrank us when its
    # own markers are present.
    return b"ATOM" not in upper


def _sample_interval(seg2: SEG2File) -> float:
    for source in (seg2.traces[0].header, seg2.header):
        dt = header_float(source, "SAMPLE_INTERVAL")
        if dt:
            return dt
    raise ValueError(
        "no SAMPLE_INTERVAL in file or trace headers; "
        "pass sample_interval=... explicitly"
    )


def _positions_from_headers(seg2: SEG2File, key: str) -> np.ndarray:
    """Scalar along-line position per trace, NaN where absent."""
    out = np.full(seg2.n_traces, np.nan)
    for i, tr in enumerate(seg2.traces):
        value = header_float(tr.header, key)
        if value is not None:
            out[i] = value
    return out


def _is_degenerate(positions: np.ndarray) -> bool:
    """True when headers carry no usable geometry.

    All-NaN, or every receiver at the same place -- both mean the operator did
    not enter the spread, and both must fall back to caller-supplied spacing.
    """
    finite = positions[np.isfinite(positions)]
    return finite.size == 0 or np.allclose(finite, finite[0])


def read_geode(
    path: str | Path,
    *,
    spacing: float | None = None,
    source_offset: float | None = None,
    spatial_ref: SpatialRef | None = None,
    azimuth: float = 90.0,
    sample_interval: float | None = None,
    elevations: np.ndarray | None = None,
) -> SeismicSurvey:
    """Read a Geometrics SEG-2 shot record.

    Parameters
    ----------
    spacing
        Geophone spacing in metres. Overrides header geometry, and is
        *required* when the headers carry none.
    source_offset
        Shot position as a distance along the line from the first geophone.
        Negative for the usual off-end shot.
    spatial_ref
        Where the line sits. Defaults to a local metric grid with the first
        geophone at the origin.
    elevations
        Per-receiver elevation, positive up. Flat spread assumed if omitted.
        Refraction inversion is sensitive to this; supply it whenever you have
        levelled the line.
    """
    path = Path(path)
    seg2 = read_seg2(path)
    n = seg2.n_traces

    dt = sample_interval or _sample_interval(seg2)
    sref = spatial_ref or local_grid(0.0, 0.0)

    rec_pos = _positions_from_headers(seg2, "RECEIVER_LOCATION")
    if spacing is not None:
        rec_pos = np.arange(n, dtype=float) * spacing
    elif _is_degenerate(rec_pos):
        raise ValueError(
            f"{path.name}: RECEIVER_LOCATION headers are empty or constant, so "
            "the spread geometry is unknown. Pass spacing=<metres>."
        )
    else:
        rec_pos = pd.Series(rec_pos).interpolate(limit_direction="both").to_numpy()

    if source_offset is not None:
        src_pos = float(source_offset)
    else:
        header_src = _positions_from_headers(seg2, "SOURCE_LOCATION")
        finite = header_src[np.isfinite(header_src)]
        if finite.size == 0:
            raise ValueError(
                f"{path.name}: no SOURCE_LOCATION in headers. "
                "Pass source_offset=<metres along line>."
            )
        src_pos = float(finite[0])

    if elevations is None:
        rec_z = np.zeros(n)
    else:
        rec_z = np.asarray(elevations, dtype=float)
        if rec_z.size != n:
            raise ValueError(
                f"elevations has {rec_z.size} entries but the file has {n} traces"
            )

    theta = np.deg2rad(azimuth)
    ux, uy = np.sin(theta), np.cos(theta)
    # Shot elevation interpolated onto the spread rather than assumed zero, so
    # an off-end shot on sloping ground does not sit underground.
    src_z = float(np.interp(src_pos, rec_pos, rec_z))

    geometry = Geometry(
        ids=[*range(1, n + 1), "S1"],
        x=[*(rec_pos * ux), src_pos * ux],
        y=[*(rec_pos * uy), src_pos * uy],
        z=[*rec_z, src_z],
        roles=[*["receiver"] * n, "source"],
        spatial_ref=sref,
    )

    trace_map = pd.DataFrame(
        {
            "receiver_id": range(1, n + 1),
            "source_id": "S1",
            "channel": [
                int(header_float(t.header, "CHANNEL_NUMBER", i + 1))
                for i, t in enumerate(seg2.traces)
            ],
            "delay": [header_float(t.header, "DELAY", 0.0) for t in seg2.traces],
        }
    )

    n_samples = max(t.data.size for t in seg2.traces)
    data = np.zeros((n, n_samples))
    for i, tr in enumerate(seg2.traces):
        data[i, : tr.data.size] = tr.data

    survey = SeismicSurvey(
        data=data,
        sample_interval=dt,
        geometry=geometry,
        trace_map=trace_map,
        metadata={
            "instrument": "Geometrics Geode",
            "format": "SEG-2",
            "seg2_file_header": seg2.header,
            "acquisition_date": seg2.header.get("ACQUISITION_DATE"),
            "acquisition_time": seg2.header.get("ACQUISITION_TIME"),
            "source_file": str(path),
        },
    )
    survey.provenance.record(
        "read",
        driver="geode-seg2",
        path=str(path),
        spacing=spacing,
        source_offset=source_offset,
        geometry_from="arguments" if spacing is not None else "headers",
    )
    return survey


driver = Driver(
    name="geode-seg2",
    description="Geometrics Geode SEG-2 shot records (refraction, active MASW)",
    can_open=_is_geometrics,
    read=read_geode,
    extensions=(".dat", ".sg2", ".seg2"),
    methods=("refraction", "masw"),
    vendor="Geometrics",
    instrument="Geode",
    notes=(
        "Header geometry is frequently absent; pass spacing= and "
        "source_offset= when RECEIVER_LOCATION is unset."
    ),
)
