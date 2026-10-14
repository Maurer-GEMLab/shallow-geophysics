"""Diurnal correction from repeat readings at a base station.

The Earth's field drifts through the day, by tens of nT on a quiet day and
by hundreds during a magnetic storm, and a roving magnetometer cannot tell
that drift from geology. The standard field fix is to come back to one
station -- the **base station** -- every half hour or so. Whatever change the
base readings show is the drift; it is interpolated to the time of every
other reading and subtracted.

Without base-station readings there is nothing to compute this from, and a
correction column in a hand-in table cannot be checked. That is why the
readers can drop such a column (``diurnal="remove"``) as well as compute the
correction (``diurnal="base"``).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..core.survey import PointSurvey


def base_station_id(survey: PointSurvey, base_station) -> str:
    """The ``station_id`` that *base_station* names in *survey*.

    Accepts the id itself (``"L1-S1"``, ``"S001"``) or, for a survey with
    line and station columns, a ``(line, station)`` pair such as ``(1, 1)``.
    """
    ids = survey.readings["station_id"].astype(str)
    if isinstance(base_station, tuple) and len(base_station) == 2:
        r = survey.readings
        if {"line", "station"} <= set(r.columns):
            match = r.loc[(r["line"].astype(str) == str(base_station[0]))
                          & (r["station"].astype(str) == str(base_station[1])), "station_id"]
            if len(match):
                return str(match.iloc[0])
    elif str(base_station) in set(ids):
        return str(base_station)
    raise KeyError(
        f"base station {base_station!r} is not in the survey. Station ids look "
        f"like {list(pd.unique(ids))[:4]}; a (line, station) pair also works."
    )


def _base_mask(survey: PointSurvey, base_station) -> tuple[np.ndarray, str | None]:
    if base_station is None:
        if "is_base" not in survey.readings:
            raise ValueError(
                "no base station given and no readings tagged is_base; pass "
                "base_station= (the station re-read through the day)"
            )
        mask = survey.readings["is_base"].fillna(False).to_numpy(bool)
        ids = pd.unique(survey.readings.loc[mask, "station_id"])
        return mask, (str(ids[0]) if len(ids) == 1 else None)
    sid = base_station_id(survey, base_station)
    return (survey.readings["station_id"].astype(str) == sid).to_numpy(), sid


def diurnal_correction(
    survey: PointSurvey,
    base_station=None,
    *,
    by: str = "auto",
    reference: str | float = "first",
) -> PointSurvey:
    """Remove the drift seen at the base station from every reading.

    Parameters
    ----------
    base_station
        The station re-read through the survey, as a ``station_id`` or a
        ``(line, station)`` pair. Omit if the reader already tagged the
        base readings (``is_base``).
    by
        What to interpolate the base drift against. ``"time"`` uses the
        reading times, the correct way. ``"order"`` uses the order of the
        rows, for a table with no times: it treats the readings as evenly
        spaced in time, which they are not, so the correction is only as
        good as the walking pace was steady. ``"auto"`` (default) uses time
        when every reading has one and falls back to order with a warning.
    reference
        The base value the survey is reduced to: ``"first"`` (the first base
        reading, the usual choice), ``"mean"`` of all base readings, or a
        number in nT.

    Returns a new survey. ``value`` is corrected; the correction added to
    each reading is in a new ``diurnal`` column and the uncorrected value is
    kept in ``raw``. Readings before the first or after the last base
    reading get the nearest base value (no extrapolation); how many there
    were is recorded in provenance, because those readings are not
    controlled.
    """
    if survey.provenance.applied("diurnal"):
        raise ValueError("a diurnal correction has already been applied to this survey")
    if survey.provenance.applied("file_correction"):
        raise ValueError(
            "the values already include the correction column from the file. "
            "Read the file again with diurnal='remove' (or diurnal='base') "
            "before computing a base-station correction."
        )
    mask, sid = _base_mask(survey, base_station)
    if mask.sum() < 2:
        raise ValueError(
            f"base station {sid or base_station!r} has {int(mask.sum())} reading(s); "
            "a drift needs at least two, at different times"
        )

    r = survey.readings
    times = pd.to_datetime(r["time"])
    if by == "auto":
        by = "time" if times.notna().all() else "order"
        if by == "order":
            warnings.warn(
                "no reading times, so the base-station drift is interpolated by "
                "reading order. That assumes the readings were evenly spaced in "
                "time; add a time column for a proper correction.",
                stacklevel=2,
            )
    if by == "time":
        if times.isna().any():
            raise ValueError(f"{int(times.isna().sum())} reading(s) have no time; use by='order'")
        x = (times - times.min()).dt.total_seconds().to_numpy(float)
    elif by == "order":
        x = np.arange(len(r), dtype=float)
    else:
        raise ValueError(f"by must be 'auto', 'time' or 'order', got {by!r}")

    values = r["value"].to_numpy(float)
    xb, vb = x[mask], values[mask]
    order = np.argsort(xb, kind="stable")
    xb, vb = xb[order], vb[order]
    if reference == "first":
        level = float(vb[0])
    elif reference == "mean":
        level = float(vb.mean())
    else:
        level = float(reference)
    drift = np.interp(x, xb, vb) - level
    outside = int(((x < xb[0]) | (x > xb[-1])).sum())

    out = survey.with_values(
        values - drift, "diurnal", base_station=sid, n_base=int(mask.sum()),
        by=by, reference=reference, reference_nT=level,
        max_drift_nT=float(np.abs(drift).max()), readings_outside_base_span=outside,
    )
    if "raw" not in out.readings:
        out.readings["raw"] = values
    out.readings["diurnal"] = -drift
    out.readings["is_base"] = mask
    out.metadata["base_station"] = sid
    return out


def plot_diurnal(survey: PointSurvey, *, ax=None):
    """The base-station readings and the drift curve interpolated through them.

    Needs a survey that has been through :func:`diurnal_correction`. The
    curve should be smooth; a jump between two base readings is a change of
    set-up (sensor height, a vehicle parked nearby), not the Sun.
    """
    import matplotlib.pyplot as plt

    if not survey.provenance.applied("diurnal") or "diurnal" not in survey.readings:
        raise ValueError("no base-station correction on this survey; run diurnal_correction first")
    step = next(s for s in reversed(list(survey.provenance)) if s.operation == "diurnal")
    r = survey.readings
    level = step.parameters["reference_nT"]
    if step.parameters["by"] == "time":
        x = pd.to_datetime(r["time"])
        xlabel = "Time"
    else:
        x = pd.Series(np.arange(len(r)))
        xlabel = "Reading number (no times: drift interpolated by order)"
    base = r["is_base"].to_numpy(bool)
    drift = -r["diurnal"].to_numpy(float)

    own = ax is None
    ax = ax or plt.subplots(figsize=(8, 3.4))[1]
    ax.plot(x, drift, "-", color="#8a8984", lw=1.2, label="Drift at each reading")
    ax.plot(x[base], drift[base], "o", color="#2a78d6", ms=7,
            label=f"Base station {step.parameters['base_station']}")
    ax.axhline(0, color="#3d3d3a", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Base field − {level:,.1f} nT")
    ax.set_title(f"Diurnal drift: largest {step.parameters['max_drift_nT']:.1f} nT",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, color="#e4e3df", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if own:
        ax.figure.autofmt_xdate() if step.parameters["by"] == "time" else None
        ax.figure.tight_layout()
    return ax
