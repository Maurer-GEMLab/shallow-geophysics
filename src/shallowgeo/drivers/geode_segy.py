"""Geometrics SeisModule Controller SEG-Y export (refraction, active MASW).

The same controller that writes the Geode's SEG-2 can export SEG-Y instead,
and it is a different animal from the marine-style SEG-Y that most readers
are tuned for. Observed on September 2026 exports:

* rev 0, big-endian, IBM floating point (format code 1), 240-byte trace
  headers, no extended textual headers;
* spread geometry as *along-line distances* in ``source_x`` (bytes 73-76) and
  ``group_x`` (bytes 81-84) with ``coordinate_scalar`` = 1, ``group_y`` and
  ``source_y`` zero -- not real coordinates;
* the unit given only by the binary-header measurement system flag
  (2 = feet), which the SEG-2 export spells out as ``UNITS FEET``;
* ``field_record`` carrying the shot number, ``trace_number`` the channel;
* the textual header naming ``GEOMETRICS SEISMODULE CONTROLLER``.

As with SEG-2, the header geometry is a hint: ``spacing`` / ``source_offset``
in metres override it, and are required when the headers are all zero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.crs import SpatialRef
from ..core.survey import SeismicSurvey
from ._segy import SEGYFile, apply_scalar, looks_like_segy, read_segy
from ._spread import (
    build_line_survey,
    length_unit_factor,
    resolve_spread,
    stack_traces,
    uniform_delay,
)
from .base import Driver, _sniff

_GEOMETRICS_MARKERS = ("GEOMETRICS", "SEISMODULE", "GEODE")
_SNIFF_BYTES = 3600


def _is_geometrics_segy(path: Path) -> bool:
    data = _sniff(path, _SNIFF_BYTES)
    if not looks_like_segy(data):
        return False
    from ._segy import _decode_text_header

    text = _decode_text_header(data[:3200]).upper()
    if any(m in text for m in _GEOMETRICS_MARKERS):
        return True
    # Unbranded SEG-Y with a plausible header: claim it weakly, by extension
    # only, so a renamed export is still readable but a random .bin is not.
    return path.suffix.lower() in driver.extensions


def _positions(traces, x_key: str, scalar_key: str) -> np.ndarray:
    out = np.full(len(traces), np.nan)
    for i, tr in enumerate(traces):
        out[i] = apply_scalar(tr.header[x_key], tr.header[scalar_key])
    return out


def _group_by_record(traces):
    """Split a file's traces into field records, in file order.

    The SeisModule writes several shots into one SEG-Y when the operator
    saves a group of records together; ``field_record`` is what separates
    them, and the textual header's ``TRACES/RECORD`` agrees. Concatenating
    them would silently attach every trace to the first record's source.
    """
    groups: list[tuple[int, list]] = []
    for tr in traces:
        record = tr.header["field_record"]
        if groups and groups[-1][0] == record:
            groups[-1][1].append(tr)
        else:
            groups.append((record, [tr]))
    return groups


def _start_time(segy: SEGYFile):
    h = segy.traces[0].header
    if not h["year"]:
        return None
    year = h["year"]
    if year < 100:  # two-digit year, as the SeisModule writes
        year += 2000 if year < 70 else 1900
    try:
        return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(
            days=h["day_of_year"] - 1,
            hours=h["hour"],
            minutes=h["minute"],
            seconds=h["second"],
        )
    except (ValueError, OverflowError):
        return None


def read_geode_segy(
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
    """Read a Geometrics SEG-Y shot record.

    Parameters are as for :func:`~shallowgeo.drivers.geode_seg2.read_geode`.
    ``units`` overrides the binary-header measurement system (``"feet"`` or
    ``"meters"``); when the flag is unset the file is assumed metric and a
    note is left in provenance.
    """
    path = Path(path)
    segy = read_segy(path)

    dt = sample_interval or segy.sample_interval
    if not dt:
        raise ValueError(f"{path.name}: no sample interval in binary header")

    header_units = segy.measurement_unit
    unit_name = units or header_units
    factor = length_unit_factor(unit_name, default="m")

    groups = _group_by_record(segy.traces)
    if len(groups) > 1 and source_offset is not None:
        raise ValueError(
            f"{path.name}: holds {len(groups)} field records "
            f"({', '.join(str(r) for r, _ in groups)}) at different source "
            "positions, so a single source_offset cannot describe them. Pass "
            "source_offset only for a single-record file."
        )

    records = []
    origins = set()
    for record, traces in groups:
        name = path.name if len(groups) == 1 else f"{path.name} record {record}"
        receivers_m, source_m, geometry_from = resolve_spread(
            _positions(traces, "group_x", "coordinate_scalar"),
            _positions(traces, "source_x", "coordinate_scalar"),
            unit_factor=factor,
            spacing=spacing,
            source_offset=source_offset,
            name=name,
        )
        origins.add(geometry_from)
        records.append(
            {
                "data": stack_traces([t.data for t in traces]),
                "receivers_m": receivers_m,
                "source_m": source_m,
                "channels": [
                    t.header["trace_number"] or i + 1 for i, t in enumerate(traces)
                ],
                "record": record,
            }
        )
    geometry_from = origins.pop() if len(origins) == 1 else "mixed"

    delay = uniform_delay(
        [t.header["delay_ms"] * 1e-3 for t in segy.traces], name=path.name
    )

    tr0 = segy.traces[0].header
    survey = build_line_survey(
        records=records,
        sample_interval=dt,
        delay=delay,
        elevations=elevations,
        azimuth=azimuth,
        spatial_ref=spatial_ref,
        start_time=_start_time(segy),
        metadata={
            "instrument": "Geometrics Geode",
            "format": "SEG-Y",
            "segy_text_header": segy.text_header.rstrip(),
            "segy_binary_header": segy.binary_header,
            "header_units": header_units,
            "field_record": tr0["field_record"],
            "field_records": [record for record, _ in groups],
            "instrument_gain_db": tr0["instrument_gain_db"] or None,
            "amplitude_units": "counts",
            "source_file": str(path),
        },
    )
    survey.provenance.record(
        "read",
        driver="geode-segy",
        path=str(path),
        spacing=spacing,
        source_offset=source_offset,
        geometry_from=geometry_from,
        field_records=[record for record, _ in groups],
        header_units=unit_name,
        unit_factor=factor,
        units_assumed=header_units is None and units is None,
        delay=delay,
    )
    return survey


driver = Driver(
    name="geode-segy",
    description="Geometrics SeisModule SEG-Y shot records (refraction, active MASW)",
    can_open=_is_geometrics_segy,
    read=read_geode_segy,
    extensions=(".sgy", ".segy"),
    methods=("refraction", "masw"),
    vendor="Geometrics",
    instrument="Geode",
    notes=(
        "Spread geometry is read from source_x/group_x as along-line "
        "distances in the binary-header measurement unit, converted to metres."
    ),
)
