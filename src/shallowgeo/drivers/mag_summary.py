"""Magnetics station-summary table: one row per station, already reduced.

This is not an instrument dump. It is the table a class hands in after the
G-857 readings have been averaged per station and corrected, typically
written from a spreadsheet::

    Line_No,Station,Dist_from_S_ft,N_Readings,Mean_nT,StdDev_nT,Correction_nT,Leveled_Mean_nT
    1,1,0.0,2,49722.0,14.1,74.0,49796.3
    1,2,13.0,2,49918.0,5.7,74.0,49992.3

Columns, by canonical name (header labels are matched case-insensitively,
with unit suffixes such as ``_ft`` or ``(nT)`` stripped and remembered):

``line``
    Survey line number. Optional; a file without it is one line.
``station``
    Station number along the line. Station numbers restart on every line,
    so the station identifier is built from both (``L1-S3``).
``distance``
    Position along the line. Its unit is read from the label
    (``Dist_from_S_ft``); pass ``distance_units=`` when the label has none.
``n``
    Readings averaged into the station value.
``raw``
    Station mean of the readings as recorded (``Mean_nT``).
``repeat_sigma``
    Standard deviation of those readings (``StdDev_nT``). With two or three
    readings per station it is a rough number, but a large one is the
    clearest sign that a station is bad: the readings disagreed with each
    other, usually because of nearby metal or a weak signal.
``correction``
    What was added to ``raw`` to give ``leveled``. Labelled a diurnal or
    leveling correction, but nothing in the table says how it was made.
``leveled``
    ``raw + correction`` (``Leveled_Mean_nT``).
``time``
    Optional reading time (``Time``, ``HH:MM:SS``), with ``date=`` or a
    ``date`` column. Needed for a proper base-station correction.

**The diurnal correction is a choice made on read** (``diurnal=``), because
a correction column in a hand-in table cannot be verified:

``"file"``
    Use the table's leveled values (or ``raw + correction``) as they are.
``"remove"``
    Ignore the correction column and use the station means as recorded.
    The right choice when no base-station readings were taken: then any
    correction in the table did not come from the field.
``"base"``
    Use the station means, then compute the correction from repeat rows at
    ``base_station=`` -- see :func:`shallowgeo.magnetics.diurnal_correction`.
    The base station may appear on as many rows as it was read; every other
    station must appear once.

The default is ``"base"`` when ``base_station`` is given and ``"file"``
otherwise. Whatever is chosen, ``leveled - (raw + correction)`` is checked
and recorded in ``metadata["leveling_residual_max"]``, with a warning above
one nanotesla -- that catches a column pasted from the wrong row.

Positions are stored in metres, as everywhere in this package. The along-line
distance is the geometry's ``y`` and line number times ``line_spacing`` is its
``x``, so lines are drawn running up the page. Without ``line_spacing`` the
lines are not positioned (``x`` is NaN) and the map plots fall back to line
number on that axis.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.crs import SpatialRef, local_grid
from ..core.geometry import Geometry
from ..core.survey import PointSurvey
from .base import Driver, _sniff_text

#: Header labels, after unit stripping, mapped to canonical column names.
_ALIASES = {
    "line": "line", "line_no": "line", "lineno": "line", "line_number": "line",
    "line_id": "line", "l": "line",
    "station": "station", "station_no": "station", "stn": "station",
    "sta": "station", "station_id": "station", "point": "station",
    "distance": "distance", "dist": "distance", "position": "distance",
    "along_line": "distance", "offset": "distance",
    "n": "n", "n_readings": "n", "nreadings": "n", "readings": "n",
    "count": "n", "n_obs": "n",
    "mean": "raw", "raw": "raw", "raw_mean": "raw", "mean_field": "raw",
    "observed": "raw", "total_field": "raw", "field": "raw",
    "stddev": "repeat_sigma", "std_dev": "repeat_sigma", "std": "repeat_sigma",
    "stdev": "repeat_sigma", "sd": "repeat_sigma", "sigma": "repeat_sigma",
    "correction": "correction", "corr": "correction", "diurnal": "correction",
    "diurnal_correction": "correction", "level_correction": "correction",
    "leveled_mean": "leveled", "leveled": "leveled", "levelled": "leveled",
    "levelled_mean": "leveled", "corrected": "leveled",
    "corrected_mean": "leveled", "final": "leveled",
    "time": "time", "timestamp": "time", "clock": "time", "date": "date",
}

#: Columns only a reduced table has -- what separates it from an instrument dump.
_SUMMARY_ONLY = {"n", "repeat_sigma", "correction", "leveled"}

_UNIT_FACTORS = {"m": 1.0, "metre": 1.0, "meter": 1.0, "metres": 1.0,
                 "meters": 1.0, "ft": 0.3048, "feet": 0.3048, "foot": 0.3048}
_UNIT_NAMES = {"m": "m", "metre": "m", "meter": "m", "metres": "m", "meters": "m",
               "ft": "ft", "feet": "ft", "foot": "ft"}
_UNIT_SUFFIXES = set(_UNIT_FACTORS) | {"nt", "gamma"}
_DIST_FROM = re.compile(r"^dist(?:ance)?_from_(?P<end>[a-z]+)$")
_PLAUSIBLE_FIELD = (15_000.0, 90_000.0)
_LEVELING_TOLERANCE = 1.0  # nT


def _parse_label(label: str) -> tuple[str, str | None]:
    """``(canonical_name, unit)`` for one header label.

    ``Dist_from_S_ft`` -> ``("distance", "ft")``, ``Mean_nT`` -> ``("raw",
    "nt")``, ``Distance (m)`` -> ``("distance", "m")``. Unrecognised labels
    come back under their own normalised name.
    """
    text = str(label).strip().lower()
    unit = None
    paren = re.search(r"\(([^)]*)\)", text)
    if paren:
        unit = paren.group(1).strip() or None
        text = text[: paren.start()] + text[paren.end():]
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    parts = text.split("_")
    if len(parts) > 1 and parts[-1] in _UNIT_SUFFIXES:
        unit = unit or parts[-1]
        text = "_".join(parts[:-1])
    if _DIST_FROM.match(text):
        return "distance", unit
    return _ALIASES.get(text, text), unit


def _distance_from(label: str | None) -> str | None:
    """The line end distances are measured from: ``Dist_from_S_ft`` -> ``"S"``."""
    if label is None:
        return None
    text = re.sub(r"[^a-z0-9]+", "_", re.sub(r"\(.*?\)", "", label.lower())).strip("_")
    parts = text.split("_")
    if len(parts) > 1 and parts[-1] in _UNIT_SUFFIXES:
        text = "_".join(parts[:-1])
    match = _DIST_FROM.match(text)
    return match.group("end").upper() if match else None


def _header(lines: list[str]) -> list[str] | None:
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        return [t for t in re.split(r"[,\t;]|\s+", stripped) if t]
    return None


def _is_summary_header(tokens: list[str] | None) -> bool:
    if not tokens:
        return False
    names = {_parse_label(t)[0] for t in tokens}
    has_value = bool(names & {"raw", "leveled"})
    has_position = bool(names & {"distance", "station"})
    return has_value and has_position and bool(names & _SUMMARY_ONLY)


def _is_mag_summary(path: Path) -> bool:
    lines = _sniff_text(path, 5)
    if "\x00" in "".join(lines):
        return False
    return _is_summary_header(_header(lines))


def inspect_mag_summary(path: str | Path) -> pd.DataFrame:
    """How each column of *path* will be read: label, canonical name, unit.

    Run this first on a table from a different spreadsheet. A column whose
    ``name`` is not one of the canonical names above is carried through but
    not used.
    """
    tokens = _header(_sniff_text(Path(path), 5)) or []
    rows = [(t, *_parse_label(t)) for t in tokens]
    return pd.DataFrame(rows, columns=["label", "name", "unit"])


def read_mag_summary(
    path: str | Path,
    *,
    diurnal: str | None = None,
    base_station=None,
    line_spacing: float | None = None,
    distance_units: str | None = None,
    date=None,
    spatial_ref: SpatialRef | None = None,
) -> PointSurvey:
    """Read a per-station magnetics summary table.

    Parameters
    ----------
    diurnal
        ``"file"``, ``"remove"`` or ``"base"``; see the module docstring.
        Every column of the file is kept whichever is chosen; this decides
        what ``survey.readings["value"]`` holds.
    base_station
        The station re-read through the day, as ``"L1-S1"`` or ``(1, 1)``.
        Its readings are tagged ``is_base`` and may repeat.
    line_spacing
        Distance between adjacent line numbers, in the same unit as the
        file's distance column. Negative if line numbers increase to the
        left (west) of line 1. Without it the lines are not positioned
        across-line and maps use line number on that axis.
    distance_units
        ``"ft"`` or ``"m"``. Needed only when the distance label carries no
        unit; overrides the label when given.
    date
        Survey date, for a table whose time column has no date in it.
    spatial_ref
        CRS for the local grid. Default: a metre-based local grid.
    """
    path = Path(path)
    if diurnal is None:
        diurnal = "base" if base_station is not None else "file"
    if diurnal not in ("file", "remove", "base"):
        raise ValueError(f"diurnal must be 'file', 'remove' or 'base', got {diurnal!r}")
    if diurnal == "base" and base_station is None:
        raise ValueError("diurnal='base' needs base_station= (the station re-read through the day)")

    table = pd.read_csv(path, sep=None, engine="python", comment="#")
    table = table.dropna(how="all")
    labels = list(table.columns)
    parsed = [_parse_label(c) for c in labels]
    names = [p[0] for p in parsed]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(
            f"{path.name}: more than one column maps to {dupes} "
            f"(labels {labels}); rename one in the file"
        )
    table.columns = names
    units = {n: u for n, u in parsed}
    original = dict(zip(names, labels))

    if not {"raw", "leveled"} & set(names):
        raise ValueError(
            f"{path.name}: no field column found. Expected a station mean "
            f"(e.g. 'Mean_nT') and/or a leveled value; columns read as {names}. "
            "Run inspect_mag_summary() to see how each label was interpreted."
        )
    if not {"distance", "station"} & set(names):
        raise ValueError(f"{path.name}: needs a distance or station column; got {names}")

    if "line" not in table:
        table["line"] = 1
    if "station" not in table:
        table["station"] = table.groupby("line").cumcount() + 1
    for col in ("line", "station"):
        numeric = pd.to_numeric(table[col], errors="coerce")
        if numeric.notna().all() and (numeric == numeric.round()).all():
            table[col] = numeric.astype(int)
    for col in ("distance", "n", "raw", "repeat_sigma", "correction", "leveled"):
        if col in table:
            table[col] = pd.to_numeric(table[col], errors="coerce")

    # -- distance unit -------------------------------------------------------
    if "distance" in table:
        unit = (distance_units or units.get("distance") or "").lower() or None
        if unit not in _UNIT_FACTORS:
            raise ValueError(
                f"{path.name}: the distance column {original['distance']!r} "
                f"has no recognisable unit ({unit!r}). Pass distance_units='ft' "
                "or 'm'."
            )
        factor = _UNIT_FACTORS[unit]
        unit = _UNIT_NAMES[unit]
    else:
        factor, unit = 1.0, None

    # -- the leveling arithmetic --------------------------------------------
    residual = None
    if {"raw", "correction", "leveled"} <= set(table.columns):
        diff = table["leveled"] - (table["raw"] + table["correction"])
        residual = float(np.nanmax(np.abs(diff))) if diff.notna().any() else None
        if residual is not None and residual > _LEVELING_TOLERANCE:
            worst = table.loc[diff.abs().idxmax(), ["line", "station"]].tolist()
            warnings.warn(
                f"{path.name}: leveled differs from mean + correction by up to "
                f"{residual:.1f} nT (worst at line {worst[0]}, station {worst[1]}). "
                "Check the correction column against the leveled values.",
                stacklevel=2,
            )

    if "leveled" not in table and {"raw", "correction"} <= set(table.columns):
        table["leveled"] = table["raw"] + table["correction"]
    if "raw" not in table and {"leveled", "correction"} <= set(table.columns):
        table["raw"] = table["leveled"] - table["correction"]
    has_correction = "leveled" in table and ("correction" in table or "raw" not in table)
    if diurnal == "file":
        source = "leveled" if has_correction else "raw"
    else:
        if "raw" not in table:
            raise ValueError(
                f"{path.name}: diurnal={diurnal!r} needs the uncorrected station "
                "means, but the file has only a leveled column and no correction "
                "to take back out of it"
            )
        source = "raw"

    field = table[source]
    if field.between(*_PLAUSIBLE_FIELD).mean() < 0.5:
        raise ValueError(
            f"{path.name}: the {source!r} column ({original.get(source, source)!r}) "
            f"is mostly outside {_PLAUSIBLE_FIELD} nT, so it is probably not "
            "total field. Run inspect_mag_summary()."
        )

    readings = pd.DataFrame({
        "station_id": [f"L{l}-S{s}" for l, s in zip(table["line"], table["station"])],
        "time": _times(table, date, path),
        "value": field.to_numpy(float),
        "line": table["line"].to_numpy(),
        "station": table["station"].to_numpy(),
    })
    if "distance" in table:
        readings["distance"] = table["distance"].to_numpy(float) * factor
    for col in ("raw", "correction", "leveled", "repeat_sigma", "n"):
        if col in table:
            readings[col] = table[col].to_numpy()
    extra = [c for c in table.columns if c not in readings.columns
             and c not in ("distance", "date")]
    for col in extra:
        readings[col] = table[col].to_numpy()

    base_id = None
    if base_station is not None:
        base_id = (f"L{base_station[0]}-S{base_station[1]}"
                   if isinstance(base_station, tuple) else str(base_station))
        if base_id not in set(readings["station_id"]):
            raise KeyError(
                f"{path.name}: base station {base_station!r} is not in the table; "
                f"station ids look like {readings['station_id'].iloc[:3].tolist()}"
            )
        readings["is_base"] = readings["station_id"] == base_id
    repeated = readings["station_id"].duplicated(keep=False) & (readings["station_id"] != base_id)
    if repeated.any():
        dup = list(pd.unique(readings.loc[repeated, "station_id"]))
        hint = "" if base_id else " If one of them is a base station, pass base_station=."
        raise ValueError(f"{path.name}: station(s) listed more than once: {dup[:5]}.{hint}")

    # -- geometry ------------------------------------------------------------
    stations = readings.drop_duplicates("station_id")
    lines = stations["line"]
    if line_spacing is not None and pd.api.types.is_numeric_dtype(lines):
        x = (lines - lines.min()).to_numpy(float) * float(line_spacing) * factor
    else:
        x = np.full(len(stations), np.nan)
    y = stations["distance"].to_numpy(float) if "distance" in stations else np.full(
        len(stations), np.nan)
    geometry = Geometry(
        ids=stations["station_id"], x=x, y=y, z=np.zeros(len(stations)),
        roles=["station"] * len(stations),
        spatial_ref=spatial_ref or local_grid(0.0, 0.0),
    )

    distance_label = original.get("distance")
    survey = PointSurvey(
        readings,
        geometry,
        quantity="total_field",
        units="nT",
        metadata={
            "format": "magnetics station summary",
            "source_file": str(path),
            "diurnal": diurnal,
            "base_station": base_id,
            "column_mapping": dict(zip(labels, names)),
            "distance_units_in_file": unit,
            "distance_label": distance_label,
            "distance_from": _distance_from(distance_label),
            "line_spacing": line_spacing,
            "lines_positioned": line_spacing is not None,
            "leveling_residual_max": residual,
        },
    )
    survey.provenance.record(
        "read", driver="mag-summary", path=str(path), column=original.get(source, source),
        distance_factor=factor,
    )
    if source == "leveled":
        # Made before the file was written, by a method the table does not
        # record. Logged so a base-station correction refuses to stack on it.
        survey.provenance.record(
            "file_correction", source=original.get("correction") or original.get("leveled"),
        )
    elif diurnal == "remove" and has_correction:
        survey.provenance.record(
            "file_correction_removed", source=original.get("correction") or original.get("leveled"),
        )
    if diurnal == "base":
        from ..magnetics.diurnal import diurnal_correction

        survey = diurnal_correction(survey, base_id)
    return survey


def _times(table: pd.DataFrame, date, path: Path) -> pd.Series:
    """Reading times, or all NaT when the table has none."""
    if "time" not in table:
        return pd.Series(pd.NaT, index=table.index, dtype="datetime64[ns]")
    text = table["time"].astype(str).str.strip()
    dated = date is not None or "date" in table
    if date is not None:
        text = pd.Timestamp(date).strftime("%Y-%m-%d") + " " + text
    elif "date" in table:
        text = table["date"].astype(str).str.strip() + " " + text
    times = pd.Series(pd.NaT, index=table.index, dtype="datetime64[ns]")
    for fmt in (["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"] if dated else ["%H:%M:%S", "%H:%M"]):
        times = times.fillna(pd.to_datetime(text, format=fmt, errors="coerce"))
    if times.isna().all():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # per-element dateutil fallback
            times = pd.to_datetime(text, errors="coerce")
    if times.isna().all():
        raise ValueError(f"{path.name}: could not parse any times from the time column")
    if not dated:
        # A clock time alone has no day. Roll forward where the clock goes
        # backwards, so a session crossing midnight stays in order.
        rollover = times.diff() < pd.Timedelta(0)
        times = times + pd.to_timedelta(rollover.cumsum(), unit="D")
    return times


driver = Driver(
    name="mag-summary",
    description="Magnetics per-station summary table (mean, std, correction, leveled)",
    can_open=_is_mag_summary,
    read=read_mag_summary,
    extensions=(".csv", ".txt", ".tsv"),
    methods=("magnetics",),
    notes=(
        "Already-reduced data, one row per station. diurnal='file'|'remove'|'base' "
        "and base_station= choose the diurnal correction; line_spacing= positions "
        "the lines for a map."
    ),
)
