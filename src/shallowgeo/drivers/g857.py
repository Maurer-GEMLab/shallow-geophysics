"""Geometrics G-857 proton-precession magnetometer ASCII dump.

Two layouts are supported, matching the two ways readings come off this
instrument in practice.

**Exported** -- five columns, with a commented header line::

    # Line  Station  Time(HH:MM:SS)  Total_Field(nT)  Quality/Signal
    L001    S001     10:30:15        52431.2          3.0

**Manual** -- three columns, hand-entered from the instrument display::

    station  time      value_nT
    S001     10:30:15  52431.2

The column mapping is resolved in three steps, most trustworthy first: an
explicit ``columns=`` argument, then a header line if the file has one, then
the column count (three means manual, five means exported). Which route was
taken is recorded in provenance as ``columns_from``, so a reading whose layout
was inferred rather than declared is always identifiable.

Geometrics publishes no column specification, so header labels may still vary
by firmware and by whatever wrote the export. Unrecognised labels are carried
through under their own names rather than silently dropped, and the column
mapped to ``field`` is checked against plausible total-field values -- a
mis-mapped file raises instead of producing quietly wrong magnetics.

Two G-857 workflows matter: roving readings over the survey grid, and repeat
readings at a fixed base station used to build the diurnal correction. Pass
``base_station=`` to tag the latter, since a magnetics dataset without an
identified base is not correctable.
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

#: Named column layouts, selectable as ``columns="manual"``.
LAYOUTS: dict[str, list[str]] = {
    "manual": ["station", "time", "field"],
    "exported": ["line", "station", "time", "field", "quality"],
}

#: Used only when a file has no header line, keyed by column count.
_LAYOUT_BY_WIDTH = {3: "manual", 5: "exported"}

#: Header labels seen in the wild, normalised to canonical names.
_COLUMN_ALIASES = {
    "line": "line", "l": "line", "lineid": "line", "line_id": "line",
    "station": "station", "stn": "station", "sta": "station",
    "station_id": "station", "stationid": "station", "point": "station",
    "time": "time", "timestamp": "time", "clock": "time",
    "date": "date",
    "field": "field", "total_field": "field", "totalfield": "field",
    "value_nt": "field", "value": "field", "nt": "field", "mag": "field",
    "magnetic_field": "field", "reading": "field", "tf": "field",
    "quality": "quality", "signal": "quality", "signal_strength": "quality",
    "sq": "quality", "qual": "quality",
}

_COMMENT_PREFIXES = ("#", "/", ";", "*")
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
#: Total field at Earth's surface, in nT. Used to sanity-check that the column
#: identified as "field" is plausibly a magnetic reading.
_PLAUSIBLE_FIELD = (15_000.0, 90_000.0)


def _looks_numeric(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _normalise_label(token: str) -> str:
    """Canonical column name for one header token.

    Strips the unit annotations the exported header carries -- ``Time(HH:MM:SS)``
    becomes ``time``, ``Total_Field(nT)`` becomes ``field`` -- and takes the
    first half of a slash-joined label, so ``Quality/Signal`` becomes
    ``quality``.
    """
    text = token.strip().lower()
    text = re.sub(r"\(.*?\)", "", text)
    text = text.split("/")[0]
    text = re.sub(r"[^a-z0-9_]", "", text).strip("_")
    return _COLUMN_ALIASES.get(text, text)


def _is_record(tokens: list[str]) -> bool:
    """Whether a token list is a reading rather than a header or note."""
    if len(tokens) < 2:
        return False
    return any(_looks_numeric(t) or _TIME_RE.match(t) for t in tokens[:3])


def _split(line: str) -> list[str]:
    return line.strip().lstrip("".join(_COMMENT_PREFIXES)).replace(",", " ").split()


def _records(lines: list[str]) -> list[list[str]]:
    """Reading lines, as raw token lists."""
    out = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(_COMMENT_PREFIXES):
            continue
        tokens = _split(line)
        if _is_record(tokens):
            out.append(tokens)
    return out


def _header_columns(lines: list[str]) -> list[str] | None:
    """Column names from a header line, commented or not.

    A header is any non-reading line whose tokens include something that maps
    to ``field`` -- the one column the file is meaningless without.
    """
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        tokens = _split(line)
        if not tokens or _is_record(tokens):
            continue
        names = [_normalise_label(t) for t in tokens]
        if "field" in names:
            return names
    return None


def _is_g857(path: Path) -> bool:
    lines = _sniff_text(path, 60)
    text = "".join(lines).upper()
    if "G-857" in text or "G857" in text:
        return True
    if "CG-5" in text:  # a CG-5 dump is also plain numeric ASCII
        return False
    if _header_columns(lines):
        return True
    records = _records(lines)
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
    """Show how *path* will be parsed: detected columns and the first *n* rows.

    The first thing to run against a dump from an unfamiliar firmware. If the
    columns come back wrong, pass the right ones as ``columns=``.
    """
    lines = _sniff_text(Path(path), n + 40)
    records = _records(lines)[:n]
    width = max((len(r) for r in records), default=0)
    names = _header_columns(lines)
    if names is None or len(names) != width:
        layout = _LAYOUT_BY_WIDTH.get(width)
        names = LAYOUTS[layout] if layout else [f"col{i}" for i in range(width)]
    return pd.DataFrame(
        [r + [None] * (width - len(r)) for r in records], columns=names[:width]
    )


def _resolve_columns(
    columns: list[str] | str | None, lines: list[str], width: int, path: Path
) -> tuple[list[str], str]:
    """Return ``(names, source)`` -- the mapping and how it was determined."""
    if isinstance(columns, str):
        if columns not in LAYOUTS:
            raise ValueError(
                f"unknown layout {columns!r}; expected one of "
                f"{sorted(LAYOUTS)} or an explicit list of column names"
            )
        return list(LAYOUTS[columns]), "layout_name"
    if columns is not None:
        return list(columns), "argument"

    from_header = _header_columns(lines)
    if from_header is not None and len(from_header) == width:
        return from_header, "header"

    layout = _LAYOUT_BY_WIDTH.get(width)
    if layout is None:
        raise ValueError(
            f"{path.name}: {width} columns and no usable header line, so the "
            f"layout cannot be inferred (recognised widths: "
            f"{sorted(_LAYOUT_BY_WIDTH)}). Pass columns= explicitly, or one of "
            f"{sorted(LAYOUTS)}."
        )
    return list(LAYOUTS[layout]), "column_count"


def read_g857(
    path: str | Path,
    *,
    columns: list[str] | str | None = None,
    date=None,
    base_station=None,
    min_quality: float | None = None,
    coordinates: dict | pd.DataFrame | None = None,
    spatial_ref: SpatialRef | None = None,
) -> PointSurvey:
    """Read a G-857 ASCII dump.

    Parameters
    ----------
    columns
        Explicit column names, or a layout name from :data:`LAYOUTS`
        (``"manual"`` or ``"exported"``). Omit to detect from the header line
        or the column count.
    date
        Survey date. The G-857 dumps a time of day but not a date, and diurnal
        correction needs absolute timestamps. Without it, times are anchored to
        today and only relative spacing is meaningful.
    base_station
        Station occupied repeatedly for diurnal control. Tagged as ``is_base``.
    min_quality
        Drop readings whose ``quality`` column falls below this. The G-857's
        signal-strength number is the instrument's own statement that a reading
        is untrustworthy; the count dropped is recorded in provenance.
    coordinates
        ``{station_id: (x, y, z)}`` or a DataFrame. The instrument records no
        position, so this is how a magnetics survey becomes georeferenced.
    """
    path = Path(path)
    lines = path.read_text(encoding="latin-1", errors="replace").splitlines()
    records = _records(lines)
    if not records:
        raise ValueError(f"{path.name}: no readings found")

    width = max(len(r) for r in records)
    names, source = _resolve_columns(columns, lines, width, path)
    if "field" not in names:
        raise ValueError(
            f"columns must include 'field'; got {names}. The total-field "
            "column is the one reading the file cannot be read without."
        )

    n = len(names)
    frame = pd.DataFrame(
        [r[:n] + [None] * (n - len(r)) for r in records], columns=names
    )

    field = pd.to_numeric(frame["field"], errors="coerce")
    if field.between(*_PLAUSIBLE_FIELD).mean() < 0.5:
        raise ValueError(
            f"{path.name}: the column mapped to 'field' holds values outside "
            f"{_PLAUSIBLE_FIELD} nT for most records, so the column mapping "
            f"(from {source}) is probably wrong: {names}. "
            "Run inspect_g857() and pass columns=."
        )

    times = _resolve_times(frame, date, path)
    readings = pd.DataFrame(
        {
            "station_id": (
                frame["station"] if "station" in frame else np.arange(len(frame))
            ),
            "time": times,
            "value": field,
        }
    )
    if "line" in frame:
        readings["line"] = frame["line"]
    if "quality" in frame:
        # Numeric where the whole column converts, raw otherwise -- some
        # firmwares write a letter grade rather than a signal number.
        numeric = pd.to_numeric(frame["quality"], errors="coerce")
        readings["quality"] = numeric if numeric.notna().all() else frame["quality"]
    if base_station is not None:
        readings["is_base"] = readings["station_id"].astype(str) == str(base_station)
    readings = readings.dropna(subset=["value", "time"]).reset_index(drop=True)

    dropped = 0
    if min_quality is not None:
        if "quality" not in readings:
            raise ValueError(
                f"{path.name}: min_quality was given but the file has no "
                f"quality column (columns: {names})"
            )
        quality = pd.to_numeric(readings["quality"], errors="coerce")
        keep = quality >= min_quality
        dropped = int((~keep).sum())
        readings = readings[keep].reset_index(drop=True)
        if readings.empty:
            raise ValueError(
                f"{path.name}: min_quality={min_quality} rejected every reading"
            )

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
            # Inferred from column count alone is the one route worth doubting.
            "column_mapping_verified": source != "column_count",
            "source_file": str(path),
        },
    )
    survey.provenance.record(
        "read", driver="g857", path=str(path), columns=names, columns_from=source
    )
    if dropped:
        survey.provenance.record(
            "quality_filter", min_quality=min_quality, dropped=dropped
        )
    return survey


def _resolve_times(frame: pd.DataFrame, date, path: Path) -> pd.Series:
    if "time" not in frame:
        raise ValueError(
            f"{path.name}: no 'time' column mapped. Diurnal correction needs "
            "reading times; map one with columns=."
        )
    time_text = frame["time"].astype(str)
    if date is None and "date" in frame:
        stamp = frame["date"].astype(str) + " " + time_text
        times = pd.to_datetime(stamp, errors="coerce")
    elif date is not None:
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
    description="Geometrics G-857 proton magnetometer ASCII dump",
    can_open=_is_g857,
    read=read_g857,
    extensions=(".asc", ".txt", ".dat", ".mag"),
    methods=("magnetics",),
    vendor="Geometrics",
    instrument="G-857",
    notes=(
        "Reads the 5-column exported layout and the 3-column manual layout; "
        "columns are taken from the header line when present."
    ),
)
