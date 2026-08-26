"""Geometrics G-857 proton-precession magnetometer ASCII dump.

.. warning::

   **This driver's default column layout is provisional.** Geometrics does not
   publish the G-857 serial-dump column specification, and MagMap2000 accepts
   it only as a generic "G-857 / ASCII" import. The default below matches the
   layout most commonly reported in the field, but it has *not* been validated
   against a dump from a specific instrument and firmware.

   Validate before trusting: run :func:`inspect_g857` on a real dump, confirm
   the mapping, then either pass ``columns=`` explicitly or update
   :data:`DEFAULT_COLUMNS` and add the file to ``tests/data``. Until a file
   from your own G-857 is in the test corpus, treat readings from this driver
   as unverified.

The layout is therefore configurable rather than fixed::

    read("SURVEY.ASC", columns=["station", "field", "time", "line"])

Two G-857 workflows matter and are distinguished by ``role``: roving readings
over the survey grid, and repeat readings at a fixed base station used to
build the diurnal correction. Pass ``base_station=`` to tag the latter, since
a magnetics dataset without an identified base is not correctable.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.crs import SpatialRef, local_grid
from ..core.geometry import Geometry
from ..core.survey import PointSurvey
from .base import Driver, _sniff_text

#: Provisional -- see the module warning before relying on this.
DEFAULT_COLUMNS = ["station", "field", "time", "line"]

_TIME_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
#: Total field at Earth's surface, in nT. Used only to sanity-check that the
#: column identified as "field" is plausibly a magnetic reading.
_PLAUSIBLE_FIELD = (15_000.0, 90_000.0)


def _numeric_records(lines: list[str]) -> list[list[str]]:
    out = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("/", "#", ";", "*")):
            continue
        tokens = stripped.replace(",", " ").split()
        if len(tokens) < 2:
            continue
        if any(_looks_numeric(t) or _TIME_RE.match(t) for t in tokens[:3]):
            out.append(tokens)
    return out


def _looks_numeric(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _is_g857(path: Path) -> bool:
    lines = _sniff_text(path, 60)
    text = "".join(lines).upper()
    if "G-857" in text or "G857" in text:
        return True
    if "CG-5" in text:  # a CG-5 dump is also plain numeric ASCII
        return False
    records = _numeric_records(lines)
    if len(records) < 3:
        return False
    # Claim the file only if some column is consistently a plausible total
    # field. Weak, but it is what distinguishes a mag dump from arbitrary CSV.
    width = min(len(r) for r in records)
    for col in range(width):
        values = [float(r[col]) for r in records if _looks_numeric(r[col])]
        if len(values) == len(records) and all(
            _PLAUSIBLE_FIELD[0] <= v <= _PLAUSIBLE_FIELD[1] for v in values
        ):
            return True
    return False


def inspect_g857(path: str | Path, n: int = 10) -> pd.DataFrame:
    """Return the first *n* parsed records as bare positional columns.

    The intended first step with a new instrument: look at what is actually in
    the file, decide the mapping, then pass it as ``columns=``.
    """
    records = _numeric_records(_sniff_text(Path(path), n + 40))[:n]
    width = max((len(r) for r in records), default=0)
    return pd.DataFrame(
        [r + [None] * (width - len(r)) for r in records],
        columns=[f"col{i}" for i in range(width)],
    )


def read_g857(
    path: str | Path,
    *,
    columns: list[str] | None = None,
    date=None,
    base_station=None,
    coordinates: dict | pd.DataFrame | None = None,
    spatial_ref: SpatialRef | None = None,
) -> PointSurvey:
    """Read a G-857 ASCII dump.

    Parameters
    ----------
    columns
        Positional column names. Must include ``field``; should include
        ``station`` and ``time``. Defaults to :data:`DEFAULT_COLUMNS`.
    date
        Survey date. The G-857 dumps a time of day but not always a date, and
        diurnal correction needs absolute timestamps.
    base_station
        Station id occupied repeatedly for diurnal control. Tagged in the
        readings table as ``is_base``.
    coordinates
        ``{station_id: (x, y, z)}`` or a DataFrame. As with the CG-5, the
        instrument records no position.
    """
    path = Path(path)
    lines = path.read_text(encoding="latin-1", errors="replace").splitlines()
    records = _numeric_records(lines)
    if not records:
        raise ValueError(f"{path.name}: no readings found")

    names = list(columns or DEFAULT_COLUMNS)
    if "field" not in names:
        raise ValueError("columns must include 'field'")
    width = len(names)
    frame = pd.DataFrame(
        [r[:width] + [None] * (width - len(r)) for r in records], columns=names
    )

    field = pd.to_numeric(frame["field"], errors="coerce")
    plausible = field.between(*_PLAUSIBLE_FIELD)
    if plausible.mean() < 0.5:
        raise ValueError(
            f"{path.name}: the column mapped to 'field' holds values outside "
            f"{_PLAUSIBLE_FIELD} nT for most records, so the column mapping is "
            "probably wrong. Run inspect_g857() and pass columns=."
        )

    times = _resolve_times(frame, date, path)
    readings = pd.DataFrame(
        {
            "station_id": frame["station"] if "station" in frame else np.arange(len(frame)),
            "time": times,
            "value": field,
        }
    )
    if "line" in frame:
        readings["line"] = frame["line"]
    if base_station is not None:
        readings["is_base"] = readings["station_id"].astype(str) == str(base_station)
    readings = readings.dropna(subset=["value", "time"]).reset_index(drop=True)

    geometry = _build_geometry(readings, coordinates, spatial_ref)

    survey = PointSurvey(
        readings,
        geometry,
        quantity="total_field",
        units="nT",
        metadata={
            "instrument": "Geometrics G-857",
            "format": "G-857 ASCII dump",
            "base_station": base_station,
            "column_mapping": names,
            "column_mapping_verified": columns is not None,
            "source_file": str(path),
        },
    )
    survey.provenance.record(
        "read",
        driver="g857",
        path=str(path),
        columns=names,
        columns_from="argument" if columns else "provisional_default",
    )
    return survey


def _resolve_times(frame: pd.DataFrame, date, path: Path) -> pd.Series:
    if "time" not in frame:
        raise ValueError(
            f"{path.name}: no 'time' column mapped. Diurnal correction needs "
            "reading times; map one with columns=."
        )
    time_text = frame["time"].astype(str)
    if date is not None:
        stamp = pd.Timestamp(date).strftime("%Y-%m-%d") + " " + time_text
        times = pd.to_datetime(stamp, errors="coerce")
    else:
        times = pd.to_datetime(time_text, errors="coerce")
        if times.notna().any():
            # Times-only parse anchors everything to today, which silently
            # breaks any survey crossing midnight. Roll the day forward where
            # the clock goes backwards so the sequence stays monotonic.
            rollover = times.diff() < pd.Timedelta(0)
            times = times + pd.to_timedelta(rollover.cumsum(), unit="D")
    if times.isna().all():
        raise ValueError(
            f"{path.name}: could not parse any reading times from the 'time' "
            "column; pass date= or correct the column mapping."
        )
    return times


def _build_geometry(readings, coordinates, spatial_ref) -> Geometry:
    stations = pd.unique(readings["station_id"])
    if coordinates is None:
        return Geometry(
            ids=stations,
            x=np.arange(len(stations), dtype=float),
            y=np.zeros(len(stations)),
            z=np.zeros(len(stations)),
            roles=["station"] * len(stations),
            spatial_ref=spatial_ref or local_grid(0.0, 0.0),
        )
    if isinstance(coordinates, pd.DataFrame):
        table = coordinates.set_index("station_id")
    else:
        table = pd.DataFrame(
            [(k, *v) for k, v in coordinates.items()],
            columns=["station_id", "x", "y", "z"],
        ).set_index("station_id")
    if spatial_ref is None:
        raise ValueError("spatial_ref is required when coordinates are supplied")
    missing = set(stations) - set(table.index)
    if missing:
        raise ValueError(f"no coordinates given for station(s): {sorted(missing)}")
    table = table.reindex(stations)
    return Geometry(
        ids=stations,
        x=table["x"], y=table["y"], z=table["z"],
        roles=["station"] * len(stations),
        spatial_ref=spatial_ref,
    )


driver = Driver(
    name="g857",
    description="Geometrics G-857 proton magnetometer ASCII dump (PROVISIONAL)",
    can_open=_is_g857,
    read=read_g857,
    extensions=(".asc", ".txt", ".dat", ".mag"),
    methods=("magnetics",),
    vendor="Geometrics",
    instrument="G-857",
    notes=(
        "Column layout is UNVALIDATED -- Geometrics publishes no spec. "
        "Run inspect_g857() and pass columns= until a real dump is in the "
        "test corpus."
    ),
)
