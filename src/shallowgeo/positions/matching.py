"""Match instrument readings to GNSS occupations by absolute time.

The manual version of this is intersecting a corrected position file against a
spreadsheet of recording start/stop times. Done by hand it is tedious and, on
an export whose UTC offsets are inconsistent, quietly wrong.

Matching is on absolute UTC, never on a local clock or a station name, so it
also works as an *independent check*: if the operator mislabelled a station in
the meter but the GNSS occupation windows are right, the time match disagrees
with the name and says so.
"""

from __future__ import annotations

import pandas as pd

from ..core.survey import PointSurvey
from .table import PositionTable


def match_occupations(
    survey: PointSurvey,
    positions: PositionTable,
    *,
    tolerance_s: float = 0.0,
    unmatched: str = "error",
) -> pd.DataFrame:
    """Assign each reading the station whose occupation window contains it.

    Parameters
    ----------
    tolerance_s
        Seconds by which a reading may fall outside a window and still match.
        Useful when the meter's clock and the receiver's differ slightly.
    unmatched
        ``"error"`` (default), ``"warn"``, or ``"ignore"``. Readings outside
        every window usually mean a clock offset between instruments, so the
        default refuses rather than silently dropping data.

    Returns
    -------
    DataFrame
        The survey's readings with ``matched_station`` added, plus
        ``name_agrees`` where the survey's own ``station_id`` can be compared.
    """
    if unmatched not in ("error", "warn", "ignore"):
        raise ValueError(
            f"unmatched must be 'error', 'warn' or 'ignore', got {unmatched!r}"
        )
    if not {"start", "end"} <= set(positions.frame.columns):
        raise ValueError(
            "position table has no occupation windows (start/end columns), "
            "so readings cannot be matched by time"
        )

    windows = positions.frame[["name", "start", "end"]].dropna(
        subset=["start", "end"]
    )
    if windows.empty:
        raise ValueError("no occupation has both a start and an end time")

    pad = pd.Timedelta(seconds=tolerance_s)
    times = pd.to_datetime(survey.readings["time"], utc=True)

    matched = []
    for t in times:
        hits = windows[(windows["start"] - pad <= t) & (t <= windows["end"] + pad)]
        matched.append(hits["name"].iloc[0] if len(hits) else None)

    out = survey.readings.copy()
    out["matched_station"] = matched

    misses = out["matched_station"].isna()
    if misses.any():
        span = f"{windows['start'].min()} to {windows['end'].max()}"
        message = (
            f"{int(misses.sum())} of {len(out)} readings fall outside every "
            f"occupation window ({span}). This normally means the meter clock "
            "and the receiver disagree, or the survey times are not UTC. "
            "Raise tolerance_s, or pass unmatched='warn' to keep going."
        )
        if unmatched == "error":
            raise ValueError(message)
        if unmatched == "warn":
            import warnings

            warnings.warn(message, RuntimeWarning, stacklevel=2)

    if "station_id" in out.columns:
        # Nullable boolean: pd.NA where nothing matched, so a missing match is
        # never confused with a disagreement.
        out["name_agrees"] = pd.array(
            [
                None if m is None else str(m) == str(s)
                for m, s in zip(out["matched_station"], out["station_id"])
            ],
            dtype="boolean",
        )
    return out


def attach_positions(
    survey: PointSurvey,
    positions: PositionTable,
    *,
    spatial_ref=None,
    by: str = "time",
    tolerance_s: float = 0.0,
) -> PointSurvey:
    """Return *survey* georeferenced from *positions*.

    ``by="time"`` uses the occupation windows and is the trustworthy route --
    it does not depend on the operator having typed matching station labels
    into two different instruments. ``by="name"`` joins on station id instead,
    for surveys whose positions carry no windows.
    """
    if by == "time":
        matched = match_occupations(survey, positions, tolerance_s=tolerance_s)
        station_of = dict(zip(survey.readings.index, matched["matched_station"]))
        readings = survey.readings.copy()
        readings["station_id"] = [station_of[i] for i in readings.index]
    elif by == "name":
        readings = survey.readings.copy()
    else:
        raise ValueError(f"by must be 'time' or 'name', got {by!r}")

    coordinates = positions.to_coordinates(spatial_ref)
    missing = set(readings["station_id"].dropna()) - set(coordinates)
    if missing:
        raise ValueError(
            f"no GNSS position for station(s): {sorted(missing)}"
        )

    from ..core.geometry import Geometry

    stations = pd.unique(readings["station_id"].dropna())
    geometry = Geometry(
        ids=stations,
        x=[coordinates[s][0] for s in stations],
        y=[coordinates[s][1] for s in stations],
        z=[coordinates[s][2] for s in stations],
        roles=["station"] * len(stations),
        spatial_ref=spatial_ref or positions.spatial_ref,
    )
    out = PointSurvey(
        readings,
        geometry,
        quantity=survey.quantity,
        units=survey.units,
        metadata=dict(survey.metadata),
        provenance=survey.provenance.copy(),
    )
    out.provenance.record(
        "attach_positions",
        by=by,
        n_stations=len(stations),
        source=positions.metadata.get("source_file"),
    )
    return out
