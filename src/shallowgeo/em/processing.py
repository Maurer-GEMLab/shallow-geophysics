"""Processing an EM track: distance, spikes, offsets, heights, conductivity.

Every function returns a new :class:`~shallowgeo.em.survey.FDEMSurvey` with
one provenance step appended, or a table; none changes its input.

The order that works for hand-carried or towed GEM-2 data:

1. :func:`add_distance` -- distance along each line and heading.
2. :func:`noise_table` -- which frequencies have any signal above the noise.
3. :func:`despike` -- remove transients (a passing car, the operator's
   phone, the start-of-line settle) before they reach a map.
4. :func:`heading_check` -- whether lines walked in opposite directions
   disagree; if they do, the difference is the operator, not the ground.
5. :func:`subtract_offsets` -- the in-phase and quadrature zero levels,
   from a free-air reading (:func:`free_air_offsets`) or a calibration.
6. :func:`add_apparent_conductivity` -- uniform-ground conductivity per
   frequency, using the deployment's height and coil orientation.

For a drone, :func:`heights_from_altitude` comes first; for a towed sled,
:func:`apply_gps_offset`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .forward import apparent_conductivity, halfspace_response
from .survey import FDEMSurvey, channel_name


def _line_groups(survey: FDEMSurvey):
    r = survey.readings
    if "line" in r:
        return [np.flatnonzero((r["line"] == ln).to_numpy()) for ln in survey.lines]
    return [np.arange(len(r))]


def _channels(survey: FDEMSurvey, channels) -> list[str]:
    if channels is None:
        return list(survey.channels.index)
    if isinstance(channels, str):
        channels = [channels]
    bad = [c for c in channels if c not in survey.readings]
    if bad:
        raise KeyError(f"no channel(s) {bad}; have {list(survey.channels.index)}")
    return list(channels)


def add_distance(survey: FDEMSurvey, *, straight: bool = True) -> FDEMSurvey:
    """Add ``distance``, ``offset`` and ``heading`` columns, per line.

    ``straight=True`` fits a straight line to each line's track and measures
    ``distance`` along it from the first sample, with ``offset`` the
    perpendicular distance off it (right of travel positive). GPS jitter of
    a metre would otherwise add to the path length -- a walked 15 m line
    measured sample to sample comes out several metres long. With
    ``straight=False`` distance is the cumulative path length and offset 0.

    ``heading`` is the line's direction of travel, degrees clockwise from
    grid north.
    """
    r = survey.readings.copy()
    dist = np.full(len(r), np.nan)
    off = np.zeros(len(r))
    head = np.full(len(r), np.nan)
    for idx in _line_groups(survey):
        xy = r[["x", "y"]].to_numpy(float)[idx]
        if len(idx) < 2:
            dist[idx] = 0.0
            continue
        if straight:
            c = np.nanmean(xy, axis=0)
            _, _, vt = np.linalg.svd(xy - c, full_matrices=False)
            u = vt[0]
            if np.dot(xy[-1] - xy[0], u) < 0:
                u = -u
            along = (xy - c) @ u
            dist[idx] = along - along[0]
            off[idx] = (xy - c) @ np.array([u[1], -u[0]])
            head[idx] = np.degrees(np.arctan2(u[0], u[1])) % 360
        else:
            step = np.hypot(*np.diff(xy, axis=0).T)
            dist[idx] = np.concatenate([[0], np.cumsum(step)])
            d = xy[-1] - xy[0]
            head[idx] = np.degrees(np.arctan2(d[0], d[1])) % 360
    r["distance"], r["offset"], r["heading"] = dist, off, head
    return survey.with_readings(r, "add_distance", straight=straight)


def noise_table(survey: FDEMSurvey, *, conductivity: float | None = None) -> pd.DataFrame:
    """Signal and noise per channel.

    ``noise`` is the sample-to-sample scatter, ``1.4826 * MAD(diff) / sqrt(2)``,
    which the slowly varying ground response hardly contributes to.
    ``median`` and ``spread`` (the 5-95 percentile range) describe the
    reading itself. Give a typical ``conductivity`` (S/m) and the table adds
    ``expected``, what uniform ground of that conductivity would read at the
    deployment's height, and ``snr = expected / noise``: a frequency whose
    expected ground signal is not well above its noise has nothing to say
    about the ground.
    """
    rows = []
    h = float(np.nanmedian(survey.heights()))
    for name, ch in survey.channels.iterrows():
        v = survey.readings[name].to_numpy(float)
        parts = [np.diff(v[i]) for i in _line_groups(survey)]
        d = np.concatenate(parts) if parts else np.diff(v)
        d = d[np.isfinite(d)]
        mad = np.median(np.abs(d - np.median(d))) if len(d) else np.nan
        row = {"channel": name, "frequency": ch["frequency"],
               "component": ch["component"],
               "median": np.nanmedian(v),
               "spread": np.nanpercentile(v, 95) - np.nanpercentile(v, 5),
               "noise": 1.4826 * mad / np.sqrt(2)}
        if conductivity is not None:
            model = halfspace_response(conductivity, ch["frequency"],
                                       survey.coil_separation, h, survey.orientation)
            row["expected"] = float(model.real if ch["component"] == "inphase"
                                    else model.imag)
            row["snr"] = abs(row["expected"]) / row["noise"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("channel")


def despike(survey: FDEMSurvey, channels=None, *, window: int = 9,
            threshold: float = 6.0, together: bool = True) -> FDEMSurvey:
    """Set to NaN readings far from their running median.

    A reading is a spike when it is more than ``threshold`` robust standard
    deviations (from the channel's own sample-to-sample noise) away from the
    median of the ``window`` samples around it, within its line. With
    ``together=True`` a sample spiking on any channel is removed from all of
    them, since a transient (metal passing the coils, the instrument
    settling) is rarely confined to one frequency. Adds a boolean ``spike``
    column; the count is recorded in provenance.
    """
    names = _channels(survey, channels)
    r = survey.readings.copy()
    flags = pd.DataFrame(False, index=r.index, columns=names)
    for name in names:
        v = r[name].to_numpy(float)
        d = np.diff(v)
        d = d[np.isfinite(d)]
        sd = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
        if not np.isfinite(sd) or sd == 0:
            continue
        for idx in _line_groups(survey):
            med = (pd.Series(v[idx]).rolling(window, center=True, min_periods=1)
                   .median().to_numpy())
            flags.loc[idx, name] = np.abs(v[idx] - med) > threshold * sd
    bad = flags.any(axis=1).to_numpy()
    if together:
        r.loc[bad, names] = np.nan
    else:
        for name in names:
            r.loc[flags[name].to_numpy(), name] = np.nan
    r["spike"] = bad
    return survey.with_readings(r, "despike", channels=names, window=window,
                                threshold=threshold, together=together,
                                removed=int(bad.sum()))


def free_air_offsets(survey: FDEMSurvey, mask, channels=None) -> dict[str, float]:
    """Zero levels from samples taken with the sensor lifted clear of the ground.

    Raising the ski above about 2 m (more than one coil separation) leaves
    almost no ground response, so whatever it then reads is the instrument's
    own offset. *mask* selects those samples.
    """
    mask = np.asarray(mask, bool)
    if not mask.any():
        raise ValueError("mask selects no samples")
    return {c: float(np.nanmedian(survey.readings.loc[mask, c]))
            for c in _channels(survey, channels)}


def subtract_offsets(survey: FDEMSurvey, offsets: dict[str, float], *,
                     source: str) -> FDEMSurvey:
    """Subtract a zero level from each named channel.

    *source* says where the numbers came from (``"free-air reading 14:02"``,
    ``"factory calibration"``) and is required, because an offset of unknown
    origin removed from the in-phase is indistinguishable from removing the
    geology. Refuses to run twice.
    """
    if survey.provenance.applied("subtract_offsets"):
        raise ValueError("offsets have already been subtracted from this survey")
    r = survey.readings.copy()
    for name, value in offsets.items():
        if name not in r:
            raise KeyError(f"no channel {name}")
        r[name] = r[name] - float(value)
    return survey.with_readings(r, "subtract_offsets", offsets=dict(offsets),
                                source=source)


def heading_check(survey: FDEMSurvey, channels=None) -> pd.DataFrame:
    """Compare lines walked one way with lines walked the other way.

    Lines are put in two groups by direction of travel (within 90 degrees of
    the first line's heading, or not). On a site with no trend across the
    lines the two groups' medians should agree to within the noise; a
    difference much larger than the noise is a heading error -- the
    operator's body, or a phone, on one side of the ski in one direction and
    the other side coming back. Needs :func:`add_distance` first.
    """
    if "heading" not in survey.readings:
        raise ValueError("run add_distance() first")
    r = survey.readings
    first = r["heading"].iloc[0]
    rel = (r["heading"] - first + 180) % 360 - 180
    forward = np.abs(rel) < 90
    noise = noise_table(survey)["noise"]
    rows = []
    for name in _channels(survey, channels):
        a = r.loc[forward, name].median()
        b = r.loc[~forward, name].median()
        rows.append({"channel": name, "forward": a, "reverse": b,
                     "difference": a - b, "noise": noise[name],
                     "ratio": abs(a - b) / noise[name]})
    return pd.DataFrame(rows).set_index("channel")


def level_lines(survey: FDEMSurvey, channels=None) -> FDEMSurvey:
    """Shift each line so its median matches the survey median.

    A last resort for line-to-line drift when there is no base-station
    re-occupation to measure it. It also removes any real difference
    between lines, so use it only where the lines cross the same ground.
    """
    names = _channels(survey, channels)
    r = survey.readings.copy()
    shifts = {}
    for idx, ln in zip(_line_groups(survey), survey.lines):
        for name in names:
            s = np.nanmedian(r[name].to_numpy(float)[idx]) - np.nanmedian(r[name])
            r.loc[idx, name] = r.loc[idx, name] - s
            shifts[f"{ln}:{name}"] = float(s)
    return survey.with_readings(r, "level_lines", shifts=shifts)


def apply_gps_offset(survey: FDEMSurvey) -> FDEMSurvey:
    """Move positions from the GPS antenna to the sensor centre.

    Uses the deployment's ``gps_offset = (forward, right, up)`` of the
    antenna relative to the sensor and each line's heading (from
    :func:`add_distance`). On a sled towed 4 m behind the antenna,
    uncorrected positions put every anomaly 4 m ahead in the direction of
    travel -- and in opposite places on lines walked in opposite directions,
    which turns one target into a herringbone.
    """
    if survey.provenance.applied("apply_gps_offset"):
        raise ValueError("the GPS offset has already been applied")
    fwd, right, _ = survey.deployment.gps_offset
    if "heading" not in survey.readings:
        survey = add_distance(survey)
    r = survey.readings.copy()
    th = np.radians(r["heading"].to_numpy(float))
    ux, uy = np.sin(th), np.cos(th)          # unit vector along travel
    rx, ry = uy, -ux                          # unit vector to the right
    r["x"] = r["x"] - fwd * ux - right * rx
    r["y"] = r["y"] - fwd * uy - right * ry
    return survey.with_readings(r, "apply_gps_offset", forward=fwd, right=right)


def heights_from_altitude(survey: FDEMSurvey, ground_elevation, *,
                          antenna_above_sensor: float | None = None) -> FDEMSurvey:
    """Sensor height above ground from the GPS altitude.

    For a drone: ``height = z - ground - antenna_above_sensor``, where ``z``
    is the antenna altitude, ``ground_elevation`` is a number, an array per
    sample, or a function ``f(x, y)`` (a DEM lookup) in the same vertical
    datum as ``z``, and ``antenna_above_sensor`` defaults to the
    deployment's ``gps_offset`` up component plus its ``sling_length``.
    Heights below 0 are set to NaN: the GPS and the ground model disagree.
    """
    r = survey.readings.copy()
    if "z" not in r:
        raise ValueError("readings have no 'z' (GPS altitude) column")
    if callable(ground_elevation):
        ground = np.asarray(ground_elevation(r["x"].to_numpy(), r["y"].to_numpy()), float)
    else:
        ground = np.broadcast_to(np.asarray(ground_elevation, float), (len(r),))
    if antenna_above_sensor is None:
        dep = survey.deployment
        antenna_above_sensor = dep.gps_offset[2] + (dep.sling_length or 0.0)
    h = r["z"].to_numpy(float) - ground - antenna_above_sensor
    bad = h < 0
    h[bad] = np.nan
    r["height"] = h
    out = survey.with_readings(r, "heights_from_altitude",
                               antenna_above_sensor=antenna_above_sensor,
                               negative=int(bad.sum()))
    return out.with_deployment(out.deployment.with_changes(
        height_source="gps_minus_ground"))


def add_apparent_conductivity(survey: FDEMSurvey, frequencies=None, *,
                              method: str = "halfspace") -> FDEMSurvey:
    """Add ``cond_<f>`` columns: apparent conductivity in mS/m.

    The uniform ground that would give each quadrature reading at the
    deployment's height (or the per-sample ``height``) and coil orientation.
    NaN where no uniform ground can: quadrature at or below zero, which on
    a GEM-2 at the low frequencies is usual -- the signal is below the noise.
    """
    freqs = survey.frequencies if frequencies is None else np.atleast_1d(frequencies)
    r = survey.readings.copy()
    h = survey.heights()
    for f in freqs:
        q = survey.quadrature(f)
        r[channel_name("Q", f).replace("Q_", "cond_")] = 1e3 * apparent_conductivity(
            q, f, survey.coil_separation, h, survey.orientation, method=method)
    return survey.with_readings(
        r, "apparent_conductivity", frequencies=[float(f) for f in freqs],
        method=method, orientation=survey.orientation,
        height=float(np.nanmedian(h)), height_source=survey.deployment.height_source)


def grid(survey: FDEMSurvey, values, *, cell: float = 0.5,
         max_distance: float = 2.0, method: str = "linear"):
    """Interpolate *values* (a column name or array) onto a regular grid.

    Returns ``(x, y, z)``: 1-D grid coordinates and a 2-D array with NaN
    farther than ``max_distance`` from any sample. Lines walked several
    metres apart leave nothing to interpolate between them, and a map that
    hides that is a map of the interpolator.
    """
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree

    v = (survey.readings[values] if isinstance(values, str) else
         pd.Series(np.asarray(values, float))).to_numpy(float)
    xy = survey.xy()
    ok = np.isfinite(v) & np.isfinite(xy).all(axis=1)
    if ok.sum() < 3:
        raise ValueError("fewer than three finite values to grid")
    xmin, ymin = np.floor(xy[ok].min(axis=0) / cell) * cell
    xmax, ymax = np.ceil(xy[ok].max(axis=0) / cell) * cell
    gx = np.arange(xmin, xmax + cell / 2, cell)
    gy = np.arange(ymin, ymax + cell / 2, cell)
    X, Y = np.meshgrid(gx, gy)
    Z = griddata(xy[ok], v[ok], (X, Y), method=method)
    dist, _ = cKDTree(xy[ok]).query(np.column_stack([X.ravel(), Y.ravel()]))
    Z[dist.reshape(X.shape) > max_distance] = np.nan
    return gx, gy, Z
