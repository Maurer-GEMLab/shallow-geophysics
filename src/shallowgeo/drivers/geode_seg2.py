"""Geometrics Geode / SeisModule SEG-2 (refraction, active MASW).

The Geode records SEG-2 with ``RECEIVER_LOCATION`` and ``SOURCE_LOCATION``
written as a *scalar distance along the spread*, not a coordinate triple, in
the length unit declared by the file-header ``UNITS`` key -- and only if the
operator entered geometry in the field software. Files with all-zero
locations do occur, so this driver treats file geometry as a hint and lets
the caller override it::

    read("LINE1.DAT", spacing=2.0, source_offset=-1.0)

Caller-supplied geometry is always in metres. Header geometry is converted
from the file's unit (``UNITS FEET`` is common on North American Geodes) and
the conversion is recorded in provenance.

Confirmed against SeisModule Controller exports, September 2025 to September
2026: ``UNITS``, per-trace ``DELAY``, ``CHANNEL_NUMBER``, ``SAMPLE_INTERVAL``,
``DESCALING_FACTOR``, ``FIXED_GAIN``, ``STACK`` and a ``NOTE`` block carrying
``BASE_INTERVAL``/``SHOT_INCREMENT``/``PHONE_INCREMENT`` are all present. The
``INSTRUMENT`` string is ``GEOMETRICS SEISMODULES CONTROLLER``, not "GEODE".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.crs import SpatialRef
from ..core.survey import SeismicSurvey
from ._seg2 import SEG2File, header_float, looks_like_seg2, read_seg2
from ._spread import (
    build_shot_survey,
    length_unit_factor,
    resolve_spread,
    stack_traces,
    uniform_delay,
)
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


def _parse_note(note: str | None) -> dict[str, str]:
    """The Geode ``NOTE`` block is ``KEY value`` pairs separated by newlines."""
    out: dict[str, str] = {}
    for line in (note or "").replace("\\n", "\n").splitlines():
        key, _, value = line.strip().partition(" ")
        if key:
            out[key.upper()] = value.strip()
    return out


def read_geode(
    path: str | Path,
    *,
    spacing: float | None = None,
    source_offset: float | None = None,
    spatial_ref: SpatialRef | None = None,
    azimuth: float = 90.0,
    sample_interval: float | None = None,
    elevations: np.ndarray | None = None,
    units: str | None = None,
) -> SeismicSurvey:
    """Read a Geometrics SEG-2 shot record.

    Parameters
    ----------
    spacing
        Geophone spacing in **metres**. Overrides header geometry, and is
        *required* when the headers carry none.
    source_offset
        Shot position in **metres** along the line from the first geophone.
        Negative for the usual off-end shot.
    spatial_ref
        Where the line sits. Defaults to a local metric grid with the first
        geophone at the origin.
    elevations
        Per-receiver elevation in metres, positive up. Flat spread assumed if
        omitted. Refraction inversion is sensitive to this; supply it whenever
        you have levelled the line.
    units
        Override the file-header ``UNITS`` (``"feet"`` or ``"meters"``) when
        the operator set it wrong in the field software.
    """
    path = Path(path)
    seg2 = read_seg2(path)

    dt = sample_interval or _sample_interval(seg2)
    header_units = seg2.header.get("UNITS")
    unit_name = units or header_units
    factor = length_unit_factor(unit_name, default="m")

    receivers_m, source_m, geometry_from = resolve_spread(
        _positions_from_headers(seg2, "RECEIVER_LOCATION"),
        _positions_from_headers(seg2, "SOURCE_LOCATION"),
        unit_factor=factor,
        spacing=spacing,
        source_offset=source_offset,
        name=path.name,
    )
    delay = uniform_delay(
        [header_float(t.header, "DELAY") for t in seg2.traces], name=path.name
    )

    tr0 = seg2.traces[0].header
    survey = build_shot_survey(
        data=stack_traces([t.data for t in seg2.traces]),
        sample_interval=dt,
        delay=delay,
        receivers_m=receivers_m,
        source_m=source_m,
        channels=[
            int(header_float(t.header, "CHANNEL_NUMBER", i + 1))
            for i, t in enumerate(seg2.traces)
        ],
        elevations=elevations,
        azimuth=azimuth,
        spatial_ref=spatial_ref,
        metadata={
            "instrument": "Geometrics Geode",
            "format": "SEG-2",
            "seg2_file_header": seg2.header,
            "acquisition_date": seg2.header.get("ACQUISITION_DATE"),
            "acquisition_time": seg2.header.get("ACQUISITION_TIME"),
            "header_units": header_units,
            "shot_sequence_number": tr0.get("SHOT_SEQUENCE_NUMBER"),
            "stack": header_float(tr0, "STACK"),
            "fixed_gain_db": header_float(tr0, "FIXED_GAIN"),
            "descaling_factor": header_float(tr0, "DESCALING_FACTOR"),
            "amplitude_units": "counts",
            "note": _parse_note(seg2.header.get("NOTE")),
            "source_file": str(path),
        },
    )
    survey.provenance.record(
        "read",
        driver="geode-seg2",
        path=str(path),
        spacing=spacing,
        source_offset=source_offset,
        geometry_from=geometry_from,
        header_units=unit_name,
        unit_factor=factor,
        delay=delay,
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
        "Header geometry is converted from the file's UNITS to metres. Pass "
        "spacing= and source_offset= (metres) when RECEIVER_LOCATION is unset."
    ),
)
