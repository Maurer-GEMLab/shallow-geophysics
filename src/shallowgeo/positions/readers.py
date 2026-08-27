"""Readers for corrected GNSS position exports.

This package does **not** process RINEX. RTKLIB (via Emlid Studio, RTKPOST, or
rtklib-py) and CSRS-PPP already do that well, and reimplementing either would
be a specialty project on its own -- besides requiring compiled code, which the
student install deliberately avoids. What is handled here is everything
downstream of the solution: antenna-height reduction, quality gating, absolute
time, duplicate detection, base shifts, and occupation matching. That is where
survey-killing errors actually live.

Formats are described by *profiles* rather than by separate readers, so a new
receiver is usually a dictionary entry. Column recognition is by normalised
header name, so a profile only needs to name the columns whose vendor spelling
is not already in :data:`_ALIASES`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.crs import SpatialRef
from .table import PositionTable

#: How the file's heights are referenced. No default anywhere -- see
#: :func:`read_positions`.
HEIGHT_REFERENCES = ("ground_mark", "antenna_phase_center")

#: Vendor header spellings -> canonical field names. Add to this rather than
#: writing a new reader when a receiver uses different wording.
_ALIASES = {
    "name": "name", "station": "name", "point": "name", "id": "name",
    "point_id": "name", "station_id": "name", "pointname": "name",
    "longitude": "longitude", "lon": "longitude", "long": "longitude",
    "latitude": "latitude", "lat": "latitude",
    "ellipsoidal_height": "height", "height": "height",
    "ellipsoid_height": "height", "hae": "height", "h_ell": "height",
    "easting": "easting", "east": "easting",
    "northing": "northing", "north": "northing",
    "elevation": "elevation", "ortho_height": "elevation",
    "orthometric_height": "elevation", "msl": "elevation",
    "solution_status": "status", "status": "status", "quality": "status",
    "fix_type": "status", "solution": "status",
    "antenna_height": "antenna_height", "ant_height": "antenna_height",
    "antenna_height_units": "antenna_height_units",
    "averaging_start": "start", "start_time": "start", "start": "start",
    "averaging_end": "end", "end_time": "end", "end": "end",
    "samples": "samples", "pdop": "pdop", "gdop": "gdop",
    "lateral_rms": "rms_h", "elevation_rms": "rms_v",
    "easting_rms": "rms_e", "northing_rms": "rms_n",
    "base_longitude": "base_longitude", "base_latitude": "base_latitude",
    "base_ellipsoidal_height": "base_height",
    "base_easting": "base_easting", "base_northing": "base_northing",
    "base_elevation": "base_elevation",
    "baseline": "baseline", "cs_name": "cs_name", "description": "description",
    "code": "code", "origin": "origin",
}

#: Per-vendor overrides layered on top of :data:`_ALIASES`.
PROFILES: dict[str, dict] = {
    "emlid-reach": {
        "description": "Emlid Reach RS2/RS3 point export (Emlid Studio / Flow)",
        "aliases": {},
        "status_ok": ("FIX",),
    },
    "generic-csv": {
        "description": "Any CSV whose headers match the standard aliases",
        "aliases": {},
        "status_ok": (),
    },
}


def _normalise(label: str) -> str:
    text = label.strip().lower()
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def _map_columns(columns, profile: dict) -> dict[str, str]:
    """``{original header: canonical name}`` for headers we recognise."""
    aliases = {**_ALIASES, **profile.get("aliases", {})}
    out = {}
    for col in columns:
        canonical = aliases.get(_normalise(col))
        if canonical and canonical not in out.values():
            out[col] = canonical
    return out


def parse_offset_times(values: pd.Series) -> pd.Series:
    """Parse timestamps carrying an explicit UTC offset, returning true UTC.

    Emlid writes ``2024-08-30 12:07:32.6 UTC-05:00``. Two traps: the literal
    ``UTC`` before the sign is not ISO-8601 and must be stripped, and the
    offset can legitimately *differ* between the start and end of a single
    occupation, so discarding it and reading the local clock turns a 35-second
    occupation into a 35-minute or one-hour one.
    """
    text = values.astype(str).str.replace(
        r"\s*UTC(?=[+-])", "", regex=True
    ).str.strip()
    parsed = pd.to_datetime(text, utc=True, format="mixed", errors="coerce")
    if parsed.isna().all() and len(text):
        parsed = pd.to_datetime(text, utc=True, errors="coerce")
    return parsed


def read_positions(
    path: str | Path,
    *,
    heights_are: str,
    profile: str = "emlid-reach",
    antenna_height: float | None = None,
    on_duplicate: str = "error",
    spatial_ref: SpatialRef | None = None,
) -> PositionTable:
    """Read a corrected GNSS position export.

    Parameters
    ----------
    heights_are
        **Required**, one of :data:`HEIGHT_REFERENCES`. Whether the file's
        heights are already at the ground mark or at the antenna phase centre.
        There is deliberately no default: exports rarely record which, the
        difference is a constant offset the size of the antenna pole, and it
        propagates directly into the free-air correction. Guessing here can
        exceed the entire gravity signal on a low-relief line.
    profile
        Key into :data:`PROFILES`. ``"generic-csv"`` relies on header aliases
        alone and is the starting point for an unfamiliar receiver.
    antenna_height
        Metres, overriding the file's own column. Required if
        ``heights_are="antenna_phase_center"`` and the file has no such column.
    on_duplicate
        ``"error"`` (default), ``"keep"``, or ``"first"``. A repeated name is
        an ordinary repeat occupation only if the positions agree; when they do
        not, the name means two different marks and any keyed join is wrong.
    """
    path = Path(path)
    if heights_are not in HEIGHT_REFERENCES:
        raise ValueError(
            f"heights_are must be one of {HEIGHT_REFERENCES}, got "
            f"{heights_are!r}. This cannot be defaulted: it decides whether "
            "the antenna height is still in the elevations."
        )
    if profile not in PROFILES:
        raise ValueError(
            f"unknown profile {profile!r}; known: {sorted(PROFILES)}"
        )
    spec = PROFILES[profile]

    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    mapping = _map_columns(raw.columns, spec)
    frame = raw.rename(columns=mapping)[list(mapping.values())]

    # Vendor exports routinely ship every projected column empty when the job
    # was recorded in geographic coordinates. Drop them so a caller reaching
    # for "elevation" gets an error rather than a column of NaN.
    empty = [c for c in frame.columns if frame[c].isna().all()]
    frame = frame.drop(columns=empty)

    if not {"longitude", "latitude", "height"} <= set(frame.columns):
        raise ValueError(
            f"{path.name}: need longitude, latitude and ellipsoidal height; "
            f"found {sorted(frame.columns)}"
            + (f" (all-empty columns dropped: {sorted(empty)})" if empty else "")
        )

    for col in ("longitude", "latitude", "height", "antenna_height", "samples",
                "pdop", "gdop", "rms_h", "rms_v", "rms_e", "rms_n", "baseline",
                "base_longitude", "base_latitude", "base_height"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in ("start", "end"):
        if col in frame.columns:
            frame[col] = parse_offset_times(frame[col])

    _check_units(frame, path)
    frame, reduction = _reduce_heights(frame, heights_are, antenna_height, path)

    table = PositionTable(
        frame,
        spatial_ref=spatial_ref,
        heights_reduced=True,
        metadata={
            "source_file": str(path),
            "profile": profile,
            "heights_are_input": heights_are,
            "antenna_height_m": reduction,
            "dropped_empty_columns": sorted(empty),
            "unmapped_columns": sorted(set(raw.columns) - set(mapping)),
        },
    )
    table.provenance.record(
        "read_positions",
        path=str(path),
        profile=profile,
        heights_are=heights_are,
        n=len(frame),
    )
    if reduction:
        table.provenance.record(
            "antenna_height_reduction", subtracted_m=reduction
        )
    else:
        table.provenance.record(
            "antenna_height_reduction", subtracted_m=0.0,
            note="file declared ground-mark heights",
        )

    _handle_duplicates(table, on_duplicate)
    return table


def _check_units(frame: pd.DataFrame, path: Path) -> None:
    if "antenna_height_units" not in frame.columns:
        return
    units = {str(u).strip().lower() for u in frame["antenna_height_units"].dropna()}
    unsupported = units - {"m", "meter", "meters", "metre", "metres"}
    if unsupported:
        raise ValueError(
            f"{path.name}: antenna height is recorded in {sorted(unsupported)}, "
            "not metres. Convert it, or pass antenna_height= in metres."
        )


def _reduce_heights(
    frame: pd.DataFrame, heights_are: str, override: float | None, path: Path
) -> tuple[pd.DataFrame, float]:
    """Bring heights to the ground mark. Returns the frame and what was removed."""
    if heights_are == "ground_mark":
        if override is not None:
            raise ValueError(
                "antenna_height was given but heights_are='ground_mark', so "
                "there is nothing to reduce. Pass heights_are="
                "'antenna_phase_center' if the antenna height is still in them."
            )
        return frame, 0.0

    if override is not None:
        heights = pd.Series(np.full(len(frame), float(override)))
    elif "antenna_height" in frame.columns and frame["antenna_height"].notna().any():
        heights = frame["antenna_height"]
        if heights.isna().any():
            raise ValueError(
                f"{path.name}: antenna height is missing for "
                f"{int(heights.isna().sum())} row(s). Pass antenna_height= to "
                "apply one value to every occupation."
            )
    else:
        raise ValueError(
            f"{path.name}: heights_are='antenna_phase_center' but the file has "
            "no antenna height column. Pass antenna_height= in metres."
        )

    frame = frame.copy()
    frame["height"] = frame["height"] - heights.to_numpy()
    # Plain floats, not numpy scalars: provenance gets serialized.
    distinct = sorted({float(v) for v in np.round(heights.to_numpy(), 6)})
    return frame, distinct[0] if len(distinct) == 1 else float("nan")


def _handle_duplicates(table: PositionTable, policy: str) -> None:
    if policy not in ("error", "keep", "first"):
        raise ValueError(
            f"on_duplicate must be 'error', 'keep' or 'first', got {policy!r}"
        )
    dupes = table.duplicates()
    if dupes.empty:
        return
    if policy == "error":
        worst = dupes.loc[dupes["separation_m"].idxmax()]
        raise ValueError(
            f"station name {worst['name']!r} is used for occupations "
            f"{worst['separation_m']:.2f} m apart "
            f"({worst['status_a']} and {worst['status_b']}). A repeated name "
            "at one mark is fine, but this is two different marks -- any "
            "join keyed on station name would silently pick one. Rename them, "
            "or pass on_duplicate='keep' to carry both."
        )
    if policy == "first":
        keep = ~table.frame.duplicated(subset="name", keep="first")
        table.frame = table.frame[keep].reset_index(drop=True)
        table.provenance.record(
            "drop_duplicate_names", dropped=int((~keep).sum())
        )
