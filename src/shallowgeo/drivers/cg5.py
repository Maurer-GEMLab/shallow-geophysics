"""Scintrex CG-5 Autograv text dump (``.TXT`` / ``.XYZ``).

The CG-5 writes a ``/``-prefixed header block -- survey identification, setup
constants, and an *options* section -- followed by a column-header line and
fixed-column readings.

The options section is the important part and the reason this driver does more
than ``pandas.read_csv``. ``Tide Correction: YES`` means the ``Grav.`` column
already has the Longman tide removed; ``Terrain Corr.`` likewise. A CG-5
export therefore does not have one meaning, and a reading is not interpretable
without knowing which corrections the operator had switched on. Those switches
are parsed into provenance as ``instrument_tide_correction`` and
``instrument_terrain_correction`` steps, so the drift/tide routines can refuse
to apply a correction the meter already made.
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

#: Canonical order of the CG-5 reading columns, used when the header line is
#: absent (as in ``.XYZ`` exports).
_DEFAULT_COLUMNS = [
    "line", "station", "alt", "grav", "sd", "tiltx", "tilty", "temp",
    "tide", "dur", "rej", "time", "dectimedate", "terrain", "date",
]

_COLUMN_ALIASES = {
    "line": "line", "station": "station", "alt": "alt", "alt.": "alt",
    "grav": "grav", "grav.": "grav", "sd": "sd", "sd.": "sd",
    "tiltx": "tiltx", "tilt": "tiltx", "tilty": "tilty",
    "temp": "temp", "temp.": "temp", "tide": "tide", "dur": "dur",
    "rej": "rej", "time": "time", "dectimedate": "dectimedate",
    "terrain": "terrain", "date": "date",
}


def _is_cg5(path: Path) -> bool:
    text = "\n".join(_sniff_text(path, 60)).upper()
    return "CG-5" in text and ("SURVEY" in text or "SETUP PARAMETERS" in text)


def _parse_header(lines: list[str]) -> dict[str, str]:
    """Key/value pairs from the ``/``-prefixed preamble."""
    out: dict[str, str] = {}
    for line in lines:
        if not line.startswith("/"):
            continue
        body = line.lstrip("/").strip()
        if not body or set(body) <= {"-"}:
            continue
        if ":" in body:
            key, _, value = body.partition(":")
            key = key.strip().lower().replace(".", "")
            key = re.sub(r"\s+", "_", key).strip("_")
            if key:
                out[key] = value.strip()
    return out


def _yes(header: dict[str, str], key: str) -> bool:
    return header.get(key, "").strip().upper().startswith("Y")


#: Header labels the CG-5 writes with an embedded space. Splitting the header
#: line on whitespace would turn "Tilt x" into two names and shift every
#: column after it, so these are collapsed before the split.
_SPACED_LABELS = ((r"tilt\s+x", "tiltx"), (r"tilt\s+y", "tilty"))


def _column_names(lines: list[str]) -> list[str] | None:
    """Column names from the ``/Line Station Alt. ...`` header line."""
    for line in lines:
        stripped = line.lstrip("/").strip()
        if not (stripped.lower().startswith("line") and "grav" in stripped.lower()):
            continue
        collapsed = stripped
        for pattern, replacement in _SPACED_LABELS:
            collapsed = re.sub(pattern, replacement, collapsed, flags=re.IGNORECASE)
        names = []
        for token in collapsed.split():
            key = token.strip().lower()
            names.append(_COLUMN_ALIASES.get(key, re.sub(r"\W+", "_", key)))
        return names
    return None


def read_cg5(
    path: str | Path,
    *,
    spatial_ref: SpatialRef | None = None,
    coordinates: dict | pd.DataFrame | None = None,
) -> PointSurvey:
    """Read a CG-5 dump into a :class:`~shallowgeo.core.survey.PointSurvey`.

    Parameters
    ----------
    coordinates
        Station positions, as ``{station_id: (x, y, z)}`` or a DataFrame with
        ``station_id, x, y, z``. The CG-5 records only a station *number* and
        an operator-entered altitude, never a position, so this is how a
        gravity survey becomes georeferenced. Without it, stations are laid
        out at unit spacing along a local line -- enough to inspect drift and
        repeatability, not enough to invert.
    """
    path = Path(path)
    lines = path.read_text(encoding="latin-1", errors="replace").splitlines()

    header = _parse_header(lines)
    names = _column_names(lines) or _DEFAULT_COLUMNS
    data_rows = [
        ln.split() for ln in lines if ln.strip() and not ln.lstrip().startswith("/")
    ]
    if not data_rows:
        raise ValueError(f"{path.name}: no reading lines found")

    width = len(names)
    frame = pd.DataFrame(
        [r[:width] + [None] * (width - len(r)) for r in data_rows], columns=names
    )
    for col in ("station", "alt", "grav", "sd", "tide", "terrain", "dur", "line"):
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    if "date" in frame and "time" in frame:
        # CG-5 writes "2010/ 5/ 4" with padding spaces that split() has already
        # collapsed inconsistently, so normalise before parsing.
        stamp = (
            frame["date"].astype(str).str.replace(r"\s+", "", regex=True)
            + " "
            + frame["time"].astype(str)
        )
        times = pd.to_datetime(stamp, errors="coerce", format="mixed")
    else:
        times = pd.to_datetime(frame.get("dectimedate"), errors="coerce")
    if times.isna().all():
        raise ValueError(
            f"{path.name}: could not parse reading times, which drift "
            "correction requires"
        )

    readings = pd.DataFrame(
        {
            "station_id": frame["station"],
            "time": times,
            "value": frame["grav"],
            "sigma": frame.get("sd"),
        }
    )
    for extra in ("alt", "tiltx", "tilty", "temp", "tide", "terrain", "dur", "rej"):
        if extra in frame:
            readings[extra] = frame[extra]
    readings = readings.dropna(subset=["value", "time"]).reset_index(drop=True)

    geometry = _build_geometry(readings, coordinates, spatial_ref, frame)

    survey = PointSurvey(
        readings,
        geometry,
        quantity="gravity",
        units="mGal",
        metadata={
            "instrument": "Scintrex CG-5",
            "format": "CG-5 text dump",
            "serial_number": header.get("instrument_s/n"),
            "survey_name": header.get("survey_name"),
            "operator": header.get("operator"),
            "cg5_header": header,
            "source_file": str(path),
        },
    )
    survey.provenance.record("read", driver="cg5", path=str(path))

    # Record what the meter already did, so later corrections can refuse to
    # double-apply. This is the whole point of parsing the options block.
    if _yes(header, "tide_correction"):
        survey.provenance.record(
            "tide_correction", applied_by="instrument", model="Longman"
        )
    if _yes(header, "terrain_corr"):
        survey.provenance.record("terrain_correction", applied_by="instrument")
    if _yes(header, "cont_tilt"):
        survey.provenance.record("tilt_correction", applied_by="instrument")

    return survey


def _build_geometry(readings, coordinates, spatial_ref, frame) -> Geometry:
    stations = pd.unique(readings["station_id"])

    if coordinates is None:
        sref = spatial_ref or local_grid(0.0, 0.0)
        alt = (
            readings.groupby("station_id")["alt"].first().reindex(stations).to_numpy()
            if "alt" in readings
            else np.zeros(len(stations))
        )
        return Geometry(
            ids=stations,
            x=np.arange(len(stations), dtype=float),
            y=np.zeros(len(stations)),
            z=np.nan_to_num(alt.astype(float)),
            roles=["station"] * len(stations),
            spatial_ref=sref,
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
        raise ValueError(
            f"no coordinates given for station(s): {sorted(missing)}"
        )
    table = table.reindex(stations)
    return Geometry(
        ids=stations,
        x=table["x"], y=table["y"], z=table["z"],
        roles=["station"] * len(stations),
        spatial_ref=spatial_ref,
    )


driver = Driver(
    name="cg5",
    description="Scintrex CG-5 Autograv text dump",
    can_open=_is_cg5,
    read=read_cg5,
    extensions=(".txt", ".xyz", ".sgd"),
    methods=("gravity",),
    vendor="Scintrex",
    instrument="CG-5",
    notes=(
        "Parses the options block so instrument-applied tide/terrain "
        "corrections are recorded in provenance and not double-applied."
    ),
)
