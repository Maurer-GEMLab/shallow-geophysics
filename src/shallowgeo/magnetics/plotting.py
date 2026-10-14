"""Station tables and plots for a magnetics :class:`PointSurvey`.

Distances are stored in metres but plotted in the unit the file was written
in (``units=None``), because a class that paced a line in feet reads its
stations in feet. Pass ``units="m"`` or ``"ft"`` to choose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.survey import PointSurvey

#: Default repeat-reading scatter above which a station is called noisy, nT.
#: Two readings a few seconds apart at one spot agree to a few nT on a quiet
#: site; tens of nT means a gradient or metal close to the sensor, hundreds
#: means the station value is not usable.
DEFAULT_MAX_SIGMA = 100.0

_FEET = 0.3048
_DATA = "#2a78d6"
_RAW = "#8a8984"
_FLAG = "#d03b3b"
_POS = "#e34948"
_NEG = "#2a78d6"
_GRID = "#e4e3df"
_DIVERGING = ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#f2a3a0", "#e34948", "#9c2424"]


# -- tables and selections ----------------------------------------------------

def _display_units(survey: PointSurvey, units: str | None) -> tuple[str, float]:
    """``(unit, metres_per_unit)`` for plot axes."""
    unit = units or survey.metadata.get("distance_units_in_file") or "m"
    if unit not in ("m", "ft"):
        raise ValueError(f"units must be 'm' or 'ft', got {unit!r}")
    return unit, (_FEET if unit == "ft" else 1.0)


def _require(survey: PointSurvey, *columns: str) -> None:
    missing = [c for c in columns if c not in survey.readings]
    if missing:
        raise ValueError(
            f"readings have no {missing} column(s); this plot needs a survey "
            "with line and along-line distance, as the mag-summary driver writes"
        )


def _per_station(survey: PointSurvey) -> PointSurvey:
    """One reading per station: repeat occupations (a base station) averaged.

    Plots and maps want one point per place; a base station read ten times
    would otherwise be drawn ten times over itself.
    """
    r = survey.readings
    if not r["station_id"].duplicated().any():
        return survey
    agg = {c: "first" for c in r.columns if c != "station_id"}
    for c in ("value", "raw", "diurnal"):
        if c in r:
            agg[c] = "mean"
    if "repeat_sigma" in r:
        agg["repeat_sigma"] = "max"
    collapsed = r.groupby("station_id", sort=False).agg(agg).reset_index()
    collapsed["occupations"] = r.groupby("station_id", sort=False).size().to_numpy()
    out = PointSurvey(collapsed, survey.geometry, quantity=survey.quantity,
                      units=survey.units, metadata=dict(survey.metadata),
                      provenance=survey.provenance.copy())
    return out


def flag_noisy(survey: PointSurvey, max_sigma: float | None = DEFAULT_MAX_SIGMA) -> pd.Series:
    """Boolean per reading: the repeat readings at that station disagreed.

    True where ``repeat_sigma`` exceeds *max_sigma* nT. All False when the
    survey has no ``repeat_sigma`` or *max_sigma* is None.
    """
    if max_sigma is None or "repeat_sigma" not in survey.readings:
        return pd.Series(False, index=survey.readings.index)
    return survey.readings["repeat_sigma"] > max_sigma


def station_table(
    survey: PointSurvey,
    *,
    units: str | None = None,
    max_sigma: float | None = DEFAULT_MAX_SIGMA,
) -> pd.DataFrame:
    """One row per station in display units, with a ``noisy`` flag.

    The table to print, sort, or save next to the plots.
    """
    unit, scale = _display_units(survey, units)
    survey = _per_station(survey)
    r = survey.readings
    out = pd.DataFrame({"station_id": r["station_id"]})
    for col in ("line", "station"):
        if col in r:
            out[col] = r[col]
    if "distance" in r:
        out[f"distance_{unit}"] = (r["distance"] / scale).round(2)
    out["value_nT"] = r["value"]
    for col in ("raw", "correction", "diurnal", "repeat_sigma", "n", "occupations"):
        if col in r:
            out[col] = r[col]
    out["noisy"] = flag_noisy(survey, max_sigma).to_numpy()
    return out


def remove_regional(
    survey: PointSurvey,
    reference: str | float = "median",
    *,
    exclude: pd.Series | np.ndarray | None = None,
) -> PointSurvey:
    """Subtract a constant regional field, leaving the anomaly in nT.

    Over a site a few hundred feet across the main field changes by well
    under a nanotesla per foot, so a constant is an adequate regional.

    Parameters
    ----------
    reference
        ``"median"`` (robust to a few large anomalies), ``"mean"``, or a
        number in nT -- an IGRF value for the site, for example.
    exclude
        Readings left out when computing the median or mean, typically
        :func:`flag_noisy`. They are still in the output.
    """
    if survey.provenance.applied("regional"):
        raise ValueError("a regional field has already been removed from this survey")
    values = survey.readings["value"]
    basis = values[~np.asarray(exclude, bool)] if exclude is not None else values
    if reference == "median":
        level = float(basis.median())
    elif reference == "mean":
        level = float(basis.mean())
    else:
        level = float(reference)
    out = survey.with_values(values - level, "regional", reference=reference, level_nT=level)
    out.quantity = "total_field_anomaly"
    return out


# -- plots --------------------------------------------------------------------

def _style(ax) -> None:
    ax.grid(True, color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _value_label(survey: PointSurvey) -> str:
    if survey.quantity == "total_field_anomaly":
        return "Anomaly (nT)"
    return "Total field (nT)"


def _distance_label(survey: PointSurvey, unit: str) -> str:
    end = survey.metadata.get("distance_from")
    return f"Distance from {end} end ({unit})" if end else f"Distance along line ({unit})"


def _lines(survey: PointSurvey, lines) -> list:
    available = list(pd.unique(survey.readings["line"]))
    if lines is None:
        return available
    missing = [l for l in lines if l not in available]
    if missing:
        raise KeyError(f"no line(s) {missing}; have {available}")
    return list(lines)


def plot_profiles(
    survey: PointSurvey,
    *,
    lines=None,
    units: str | None = None,
    show_raw: bool = False,
    error_bars: bool = True,
    max_sigma: float | None = DEFAULT_MAX_SIGMA,
    ncols: int = 4,
    sharey: bool = True,
    figsize=None,
):
    """Field against along-line distance, one panel per line.

    Parameters
    ----------
    show_raw
        Also draw the values before the diurnal correction (``raw``), dashed
        grey. The gap between the two curves is the correction.
    error_bars
        Draw +/- ``repeat_sigma``. The y-limits follow the values, not the
        bars, so one very noisy station cannot flatten the whole panel.
    max_sigma
        Stations with larger ``repeat_sigma`` (nT) are circled in red; None
        to switch off.
    sharey
        Same field scale on every panel, so anomaly sizes compare directly.
        False lets each line fill its panel.

    Returns the figure.
    """
    import matplotlib.pyplot as plt

    _require(survey, "line", "distance")
    survey = _per_station(survey)
    unit, scale = _display_units(survey, units)
    chosen = _lines(survey, lines)
    r = survey.readings
    noisy = flag_noisy(survey, max_sigma)
    ncols = max(1, min(ncols, len(chosen)))
    nrows = int(np.ceil(len(chosen) / ncols))
    figsize = figsize or (3.4 * ncols, 2.4 * nrows + 0.6)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True,
                             sharey=sharey, squeeze=False)

    shown = r[r["line"].isin(chosen)]
    columns = ["value"] + (["raw"] if show_raw and "raw" in r else [])
    lo = float(np.nanmin(shown[columns].to_numpy()))
    hi = float(np.nanmax(shown[columns].to_numpy()))
    pad = 0.06 * (hi - lo or 1.0)

    for ax, line in zip(axes.flat, chosen):
        rows = r["line"] == line
        d = r.loc[rows, "distance"].to_numpy() / scale
        v = r.loc[rows, "value"].to_numpy()
        order = np.argsort(d)
        if show_raw and "raw" in r:
            ax.plot(d[order], r.loc[rows, "raw"].to_numpy()[order], "--",
                    color=_RAW, lw=1.2, label="Before diurnal correction")
        if error_bars and "repeat_sigma" in r:
            ax.errorbar(d, v, yerr=r.loc[rows, "repeat_sigma"].to_numpy(), fmt="none",
                        ecolor=_DATA, elinewidth=0.8, alpha=0.45, capsize=0)
        ax.plot(d[order], v[order], "-o", color=_DATA, lw=1.5, ms=3.5,
                label=_value_label(survey).split(" (")[0])
        bad = noisy[rows].to_numpy()
        if bad.any():
            ax.plot(d[bad], v[bad], "o", ms=9, mfc="none", mec=_FLAG, mew=1.5,
                    label=f"Repeat σ > {max_sigma:g} nT")
        ax.set_title(f"Line {line}", fontsize=10, loc="left")
        if not sharey:
            sub = r.loc[rows, columns].to_numpy()
            s_lo, s_hi = np.nanmin(sub), np.nanmax(sub)
            s_pad = 0.06 * (s_hi - s_lo or 1.0)
            ax.set_ylim(s_lo - s_pad, s_hi + s_pad)
        _style(ax)
    if sharey:
        axes.flat[0].set_ylim(lo - pad, hi + pad)
    for ax in axes.flat[len(chosen):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(_value_label(survey))
    for ax in axes[-1, :]:
        ax.set_xlabel(_distance_label(survey, unit))
    # Panels above an empty slot in the last row need their own x labels.
    for ax in axes.flat[max(0, len(chosen) - ncols):len(chosen)]:
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_xlabel(_distance_label(survey, unit))

    handles = {}
    for ax in axes.flat[: len(chosen)]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="upper right", ncol=len(handles),
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def _across(survey: PointSurvey, scale: float) -> tuple[np.ndarray, str, bool]:
    """Across-line position per reading, its axis label, and whether to scale."""
    r = survey.readings
    x = survey.geometry.table.set_index("id")["x"].reindex(r["station_id"]).to_numpy(float)
    if np.isfinite(x).all():
        return x / scale, "Across-line position ({unit})", True
    return r["line"].to_numpy(float), "Line number (line spacing not given)", False


def plot_stacked_profiles(
    survey: PointSurvey,
    *,
    units: str | None = None,
    reference: str | float = "median",
    width: float = 0.8,
    max_sigma: float | None = DEFAULT_MAX_SIGMA,
    ax=None,
):
    """Every line on one plan view, its anomaly drawn as a wiggle.

    Each line is drawn at its across-line position; the curve swings right
    for a field above *reference* (shaded red) and left for one below
    (shaded blue). An anomaly from one buried source shows up on several
    neighbouring lines at the same distance -- this is the plot that shows
    whether it does.

    Parameters
    ----------
    width
        Largest swing, as a fraction of the gap between lines.
    reference
        As for :func:`remove_regional`; ignored if the survey already has
        the regional removed.
    """
    import matplotlib.pyplot as plt

    _require(survey, "line", "distance")
    survey = _per_station(survey)
    unit, scale = _display_units(survey, units)
    anomaly = survey if survey.provenance.applied("regional") else remove_regional(
        survey, reference, exclude=flag_noisy(survey, max_sigma))
    r = anomaly.readings
    across, xlabel, _ = _across(survey, scale)
    xlabel = xlabel.format(unit=unit)
    noisy = flag_noisy(survey, max_sigma).to_numpy()

    positions = pd.Series(across).groupby(r["line"].to_numpy()).first()
    gaps = np.diff(np.sort(positions.to_numpy()))
    gap = float(np.median(gaps)) if len(gaps) else 1.0
    good = r["value"].to_numpy()[~noisy]
    amp = float(np.nanpercentile(np.abs(good), 98)) if good.size else 1.0
    k = width * gap / (amp or 1.0)

    own = ax is None
    ax = ax or plt.subplots(figsize=(8, 7))[1]
    for line, x0 in positions.items():
        rows = (r["line"] == line).to_numpy()
        d = r.loc[rows, "distance"].to_numpy() / scale
        a = np.clip(r.loc[rows, "value"].to_numpy(), -1.5 * amp, 1.5 * amp)
        order = np.argsort(d)
        d, a = d[order], a[order]
        xs = x0 + k * a
        ax.fill_betweenx(d, x0, xs, where=a > 0, color=_POS, alpha=0.35, lw=0,
                         interpolate=True)
        ax.fill_betweenx(d, x0, xs, where=a < 0, color=_NEG, alpha=0.35, lw=0,
                         interpolate=True)
        ax.plot(xs, d, color="#3d3d3a", lw=1.0)
        ax.axvline(x0, color=_GRID, lw=0.8, zorder=0)
        bad = noisy[rows][order]
        if bad.any():
            ax.plot(xs[bad], d[bad], "o", ms=7, mfc="none", mec=_FLAG, mew=1.3)
        ax.text(x0, 1.01, str(line), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color="#52514e")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(_distance_label(survey, unit))
    level = anomaly.provenance[-1].parameters.get("level_nT")
    ref = f" about {level:,.0f} nT" if level is not None else ""
    ax.set_title(f"Anomaly{ref}; swing of one line gap = {amp / width:,.0f} nT",
                 fontsize=10, loc="left", pad=16)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if own:
        ax.figure.tight_layout()
    return ax


def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("mag_diverging", _DIVERGING)


def plot_map(
    survey: PointSurvey,
    *,
    units: str | None = None,
    anomaly: bool = True,
    reference: str | float = "median",
    max_sigma: float | None = DEFAULT_MAX_SIGMA,
    drop_noisy: bool = False,
    spacing: float | None = None,
    levels: int = 15,
    clip_percentile: float = 98.0,
    show_stations: bool = True,
    ax=None,
):
    """Contour map of the field, interpolated between stations.

    Linear interpolation on a Delaunay triangulation of the stations, so the
    map stops at the edge of the survey rather than extrapolating.

    Parameters
    ----------
    anomaly
        Map the field minus *reference* on a diverging colour scale centred
        on zero (default). False maps total field.
    drop_noisy
        Leave stations with ``repeat_sigma > max_sigma`` out of the gridding.
        They are still plotted, as red crosses.
    spacing
        Grid cell size in display units. Default: a quarter of the median
        station spacing.
    clip_percentile
        Colour scale spans this percentile of ``|anomaly|`` at the stations
        that are not noisy, so one spike does not wash out everything else.

    Needs the lines positioned: read the file with ``line_spacing=``.
    Interpolating between line numbers and distances in feet would put the
    contours wherever the mixed units happen to send them.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.ticker import MaxNLocator
    from scipy.interpolate import griddata

    _require(survey, "line", "distance")
    survey = _per_station(survey)
    unit, scale = _display_units(survey, units)
    noisy = flag_noisy(survey, max_sigma).to_numpy()
    shown = survey
    if anomaly and not survey.provenance.applied("regional"):
        shown = remove_regional(survey, reference, exclude=noisy)
    r = shown.readings
    x, xlabel, to_scale = _across(survey, scale)
    if not to_scale:
        raise ValueError(
            "the lines have no across-line positions, so there is no map to "
            "grid. Read the file with line_spacing= (in the file's distance "
            "unit), or use plot_stacked_profiles, which works on line number."
        )
    xlabel = xlabel.format(unit=unit)
    y = r["distance"].to_numpy() / scale
    v = r["value"].to_numpy(float)
    use = np.isfinite(v) & (~noisy if drop_noisy else True)

    if spacing is None:
        steps = np.diff(np.sort(y))
        spacing = float(np.median(steps[steps > 0])) / 4 if (steps > 0).any() else 1.0
    gx = np.arange(np.nanmin(x), np.nanmax(x) + spacing / 2, spacing)
    gy = np.arange(np.nanmin(y), np.nanmax(y) + spacing / 2, spacing)
    GX, GY = np.meshgrid(gx, gy)
    grid = griddata((x[use], y[use]), v[use], (GX, GY), method="linear")

    ax = ax or plt.subplots(figsize=(7, 7))[1]
    # The colour scale comes from the trusted stations either way, so a map
    # with the noisy stations and one without them share a scale.
    scaled = v[np.isfinite(v) & ~noisy] if (~noisy).any() else v[np.isfinite(v)]
    if anomaly:
        span = float(np.nanpercentile(np.abs(scaled), clip_percentile)) or 1.0
        lev = MaxNLocator(levels, symmetric=True).tick_values(-span, span)
        span = float(lev[-1])
        cmap, norm = _diverging_cmap(), TwoSlopeNorm(0.0, -span, span)
    else:
        lo, hi = np.nanpercentile(scaled, [100 - clip_percentile, clip_percentile])
        cmap, norm = "Blues", None
        lev = MaxNLocator(levels).tick_values(lo, hi)
    # Values beyond the colour scale fill the end bins rather than going blank.
    grid = np.clip(grid, lev[0], lev[-1])
    filled = ax.contourf(GX, GY, grid, levels=lev, cmap=cmap, norm=norm, extend="both")
    ax.contour(GX, GY, grid, levels=lev, colors="#3d3d3a", linewidths=0.3, alpha=0.5)
    cbar = ax.figure.colorbar(filled, ax=ax, shrink=0.85)
    cbar.set_label(_value_label(shown))

    if show_stations:
        ax.plot(x[~noisy], y[~noisy], ".", color="#3d3d3a", ms=3, label="Station")
        if noisy.any():
            ax.plot(x[noisy], y[noisy], "x", color=_FLAG, ms=6, mew=1.5,
                    label=f"Repeat σ > {max_sigma:g} nT"
                    + (" (not gridded)" if drop_noisy else ""))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2,
                  frameon=False, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(_distance_label(survey, unit))
    title = "Total-field anomaly" if anomaly else "Total field"
    if anomaly:
        level = shown.provenance[-1].parameters.get("level_nT")
        if level is not None:
            title += f" (minus {level:,.0f} nT)"
    ax.set_title(title, fontsize=10, loc="left")
    return ax


def plot_corrections(survey: PointSurvey, *, ax=None):
    """The correction added to each line, in nT.

    For a diurnal/leveling correction this is the drift of the field (and of
    the instrument) between the base-station ties. A line whose correction
    is far from its neighbours' is worth a second look at its tie readings.
    """
    import matplotlib.pyplot as plt

    _require(survey, "line", "correction")
    per_line = survey.readings.groupby("line", sort=False)["correction"]
    spread = (per_line.max() - per_line.min()).abs()
    values = per_line.mean()
    own = ax is None
    ax = ax or plt.subplots(figsize=(7, 3.4))[1]
    labels = [str(l) for l in values.index]
    ax.bar(labels, values.to_numpy(), color=_DATA, width=0.6)
    ax.axhline(0, color="#3d3d3a", lw=0.8)
    for i, v in enumerate(values.to_numpy()):
        ax.annotate(f"{v:+.0f}", (i, v), xytext=(0, 3 if v >= 0 else -3),
                    textcoords="offset points", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8, color="#52514e")
    ax.set_xlabel("Line")
    ax.set_ylabel("Correction (nT)")
    note = "" if (spread < 1e-6).all() else " (varies along some lines; mean shown)"
    ax.set_title("Correction applied to each line" + note, fontsize=10, loc="left")
    ax.margins(y=0.15)
    _style(ax)
    ax.grid(axis="x", visible=False)
    if own:
        ax.figure.tight_layout()
    return ax


def plot_repeatability(
    survey: PointSurvey,
    *,
    max_sigma: float | None = DEFAULT_MAX_SIGMA,
    ax=None,
):
    """Repeat-reading standard deviation at every station, by line, log scale.

    Most stations sit at a few nT. The ones above the line are where the
    readings at a single spot disagreed -- nearby steel, a strong gradient,
    or a poor signal -- and their station value should be trusted least.
    """
    import matplotlib.pyplot as plt

    _require(survey, "line", "repeat_sigma")
    r = survey.readings
    noisy = flag_noisy(survey, max_sigma).to_numpy()
    lines = list(pd.unique(r["line"]))
    pos = {l: i for i, l in enumerate(lines)}
    rng = np.random.default_rng(0)
    x = r["line"].map(pos).to_numpy(float) + rng.uniform(-0.18, 0.18, len(r))
    # A zero standard deviation (identical readings) cannot sit on a log axis.
    sigma = np.maximum(r["repeat_sigma"].to_numpy(float), 0.1)

    own = ax is None
    ax = ax or plt.subplots(figsize=(8, 3.6))[1]
    ax.plot(x[~noisy], sigma[~noisy], "o", color=_DATA, ms=4, alpha=0.8, label="Station")
    if noisy.any():
        ax.plot(x[noisy], sigma[noisy], "x", color=_FLAG, ms=7, mew=1.6,
                label=f"Repeat σ > {max_sigma:g} nT ({noisy.sum()})")
    if max_sigma is not None:
        ax.axhline(max_sigma, color=_FLAG, lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_xticks(range(len(lines)), [str(l) for l in lines])
    ax.set_xlabel("Line")
    ax.set_ylabel("Repeat σ (nT)")
    ax.set_title("Scatter of the repeat readings at each station", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    ax.grid(axis="x", visible=False)
    if own:
        ax.figure.tight_layout()
    return ax
