"""Plots for an EM track.

``plot_qc``
    Sample interval, walking speed, GPS fix and power-line noise against
    time: the four things that make a reading untrustworthy before the
    ground has anything to do with it.
``plot_track``
    The samples in plan view, coloured by line or by any value, with the
    direction each line was walked.
``plot_channels``
    Every frequency of one component against sample number, lines shaded:
    the raw data as the instrument saw them.
``plot_profiles``
    One or more columns against distance along each line.
``plot_map``
    A gridded map, masked away from the samples, with the track on top.
``plot_frequency_response``
    The measured response at each frequency next to what uniform ground of
    a few conductivities would give at this height.
``plot_depth_response``
    McNeill's cumulative response: how much of the reading comes from
    below each depth, for hand, sled and drone heights.
``plot_vendor_inversion``
    The instrument software's three-layer model along a line, with its
    misfit, so it can be judged rather than believed.

Plan coordinates are drawn relative to the south-west corner of the survey
(``origin`` in the returned axes title) because six-digit UTM eastings make
unreadable tick labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .forward import cumulative_response, halfspace_response
from .processing import grid
from .survey import FDEMSurvey, channel_name


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _origin(survey: FDEMSurvey) -> tuple[float, float]:
    xy = survey.xy()
    return float(np.nanmin(xy[:, 0])), float(np.nanmin(xy[:, 1]))


def _line_colors(lines):
    plt = _plt()
    cmap = plt.get_cmap("tab10")
    return {ln: cmap(i % 10) for i, ln in enumerate(lines)}


def _values(survey: FDEMSurvey, values):
    if values is None:
        return None
    if isinstance(values, str):
        return survey.readings[values].to_numpy(float)
    return np.asarray(values, float)


def plot_track(survey: FDEMSurvey, values=None, *, ax=None, cmap="viridis",
               label=None, vmin=None, vmax=None, arrows=True, size=12):
    """Samples in plan view, coloured by line (default) or by *values*."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5))
    x0, y0 = _origin(survey)
    r = survey.readings
    x, y = r["x"].to_numpy() - x0, r["y"].to_numpy() - y0
    v = _values(survey, values)
    if v is None:
        colors = _line_colors(survey.lines)
        for ln in survey.lines:
            m = (r["line"] == ln).to_numpy() if "line" in r else np.ones(len(r), bool)
            ax.plot(x[m], y[m], ".", ms=3, color=colors[ln], label=f"line {ln}")
        ax.legend(fontsize=8, loc="best")
    else:
        sc = ax.scatter(x, y, c=v, s=size, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax, label=label or (values if isinstance(values, str) else ""))
    if arrows and "line" in r:
        for ln in survey.lines:
            m = np.flatnonzero((r["line"] == ln).to_numpy())
            ok = m[np.isfinite(x[m]) & np.isfinite(y[m])]
            if len(ok) < 2:
                continue
            # Arrowhead on the last stretch of the line, in the direction walked.
            tail = ok[max(0, len(ok) - max(3, len(ok) // 6))]
            ax.annotate("", xy=(x[ok[-1]], y[ok[-1]]), xytext=(x[tail], y[tail]),
                        arrowprops={"arrowstyle": "-|>", "color": "0.25", "lw": 1.2})
            ax.text(x[ok[0]], y[ok[0]], f" {ln}", fontsize=9, color="0.2",
                    ha="right", va="center")
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"origin E {x0:.1f}, N {y0:.1f}", fontsize=8, loc="right")
    return ax


def plot_qc(survey: FDEMSurvey, *, axes=None):
    """Sample interval, speed, GPS fix quality and power-line field vs time."""
    plt = _plt()
    r = survey.readings
    if axes is None:
        _, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    t = (r["time"] - r["time"].iloc[0]).dt.total_seconds().to_numpy()
    dt = np.diff(t)
    step = np.hypot(np.diff(r["x"]), np.diff(r["y"]))
    same_line = (r["line"].to_numpy()[1:] == r["line"].to_numpy()[:-1]) \
        if "line" in r else np.ones(len(dt), bool)
    axes[0].plot(t[1:][same_line], 1e3 * dt[same_line], ".", ms=3)
    axes[0].set_ylabel("sample\ninterval (ms)")
    speed = np.where(same_line & (dt > 0), step / np.where(dt > 0, dt, 1), np.nan)
    axes[1].plot(t[1:], speed, ".", ms=3)
    # Running median within each line only, so it never bridges a gap.
    ids = np.cumsum(np.concatenate([[0], ~same_line[1:]])) if len(dt) else []
    run = pd.Series(speed).groupby(ids).transform(
        lambda v: v.rolling(15, center=True, min_periods=3).median())
    axes[1].plot(t[1:], np.where(same_line, run, np.nan), "k-", lw=1)
    axes[1].set_ylabel("speed (m/s)\n(GPS is 1 Hz)")
    if "gps_quality" in r:
        axes[2].step(t, r["gps_quality"], where="post")
        axes[2].set_yticks([0, 1, 2, 4, 5])
        axes[2].set_yticklabels(["none", "GPS", "DGPS", "RTK fix", "RTK float"], fontsize=7)
    axes[2].set_ylabel("GPS fix")
    if "powerline" in r:
        axes[3].plot(t, r["powerline"], ".", ms=3)
    axes[3].set_ylabel("power line\n(file units)")
    axes[3].set_xlabel("seconds since first sample")
    if "line" in r:
        for ax in axes:
            _shade_lines(ax, t, r["line"].to_numpy())
    return axes


def _shade_lines(ax, t, lines):
    edges = np.flatnonzero(lines[1:] != lines[:-1]) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(lines)]]) - 1
    for k, (a, b) in enumerate(zip(starts, ends)):
        if k % 2:
            ax.axvspan(t[a], t[b], color="0.92", zorder=0)


def plot_channels(survey: FDEMSurvey, component: str = "Q", frequencies=None,
                  *, axes=None, x: str = "index"):
    """Each frequency of one component, one panel each, against sample or time."""
    plt = _plt()
    freqs = survey.frequencies if frequencies is None else np.atleast_1d(frequencies)
    if axes is None:
        _, axes = plt.subplots(len(freqs), 1, figsize=(9, 1.5 * len(freqs) + 1),
                               sharex=True, squeeze=False)
        axes = axes[:, 0]
    r = survey.readings
    xs = (np.arange(len(r)) if x == "index"
          else (r["time"] - r["time"].iloc[0]).dt.total_seconds().to_numpy())
    for ax, f in zip(axes, freqs[::-1]):
        name = channel_name(component, f)
        ax.plot(xs, r[name], "-", lw=0.7, color="C0" if component.upper().startswith("Q") else "C3")
        ax.set_ylabel(f"{f/1e3:g} kHz", rotation=0, ha="right", va="center")
        if "line" in r:
            _shade_lines(ax, xs, r["line"].to_numpy())
    comp = "quadrature" if component.upper().startswith("Q") else "in-phase"
    axes[0].set_title(f"{comp} (ppm), highest frequency at top")
    axes[-1].set_xlabel("sample" if x == "index" else "seconds")
    return axes


def plot_profiles(survey: FDEMSurvey, columns, *, x: str = "distance", axes=None,
                  ylabel=None):
    """Each column against distance, one panel per column, every line overlaid."""
    plt = _plt()
    if isinstance(columns, str):
        columns = [columns]
    r = survey.readings
    if x not in r:
        raise ValueError(f"no {x!r} column; run add_distance() first")
    if axes is None:
        _, axes = plt.subplots(len(columns), 1, figsize=(9, 2.2 * len(columns) + 0.6),
                               sharex=True, squeeze=False)
        axes = axes[:, 0]
    colors = _line_colors(survey.lines)
    for ax, col in zip(axes, columns):
        for ln in survey.lines:
            m = (r["line"] == ln).to_numpy() if "line" in r else np.ones(len(r), bool)
            ax.plot(r.loc[m, x], r.loc[m, col], "-", lw=1, color=colors[ln],
                    label=f"line {ln}")
        ax.set_ylabel(ylabel or col)
    axes[0].legend(fontsize=7, ncol=min(len(survey.lines), 6), loc="best")
    axes[-1].set_xlabel(f"{x} along line (m)")
    return axes


def plot_map(survey: FDEMSurvey, values, *, ax=None, cell: float = 0.5,
             max_distance: float = 2.0, cmap="viridis", label=None,
             vmin=None, vmax=None, track=True):
    """Gridded map of *values*, blank farther than ``max_distance`` from data."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5))
    x0, y0 = _origin(survey)
    gx, gy, Z = grid(survey, values, cell=cell, max_distance=max_distance)
    im = ax.pcolormesh(gx - x0, gy - y0, Z, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="nearest")
    plt.colorbar(im, ax=ax, label=label or (values if isinstance(values, str) else ""))
    if track:
        xy = survey.xy()
        ax.plot(xy[:, 0] - x0, xy[:, 1] - y0, ".", ms=1, color="k", alpha=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"origin E {x0:.1f}, N {y0:.1f}", fontsize=8, loc="right")
    return ax


def plot_frequency_response(survey: FDEMSurvey, conductivities=(0.005, 0.02, 0.05),
                            *, height=None, axes=None):
    """Measured median I and Q per frequency, with uniform-ground curves.

    Error bars span the 25th-75th percentile of the survey. If the
    measured quadrature follows a model curve, the frequencies agree about
    the ground; where they depart from every curve, the departure is the
    instrument (zero level, noise), not depth.
    """
    plt = _plt()
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 4))
    h = float(np.nanmedian(survey.heights())) if height is None else height
    freqs = survey.frequencies
    fm = np.logspace(np.log10(freqs.min() / 1.5), np.log10(freqs.max() * 1.5), 60)
    for sig in conductivities:
        m = halfspace_response(sig, fm, survey.coil_separation, h, survey.orientation)
        axes[0].plot(fm, m.real, "--", lw=1, label=f"{1e3*sig:g} mS/m")
        axes[1].plot(fm, m.imag, "--", lw=1, label=f"{1e3*sig:g} mS/m")
    for ax, comp in zip(axes, ("I", "Q")):
        vals = np.array([survey.channel(comp, f) for f in freqs])
        med = np.nanmedian(vals, axis=1)
        lo, hi = np.nanpercentile(vals, [25, 75], axis=1)
        ax.errorbar(freqs, med, yerr=[med - lo, hi - med], fmt="ko", ms=4,
                    capsize=3, label="measured")
        ax.set_xscale("log")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel(f"{'in-phase' if comp == 'I' else 'quadrature'} (ppm)")
        ax.legend(fontsize=7)
    axes[0].set_title(f"in-phase, h = {h:.2f} m {survey.orientation}", fontsize=9)
    axes[1].set_title(f"quadrature, h = {h:.2f} m {survey.orientation}", fontsize=9)
    return axes


def plot_depth_response(coil_separation: float = 1.66, heights=None, *, ax=None,
                        max_depth: float = 6.0):
    """Fraction of the LIN reading from below each depth, per platform height.

    ``heights`` maps a label to a height in metres; default hand (1 m),
    sled (0.25 m) and drone (3 m). Solid HCP, dashed VCP.
    """
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5.5))
    heights = heights or {"sled 0.25 m": 0.25, "handheld 1 m": 1.0, "drone 3 m": 3.0}
    d = np.linspace(0, max_depth, 300)
    for k, (name, h) in enumerate(heights.items()):
        for o, ls in (("HCP", "-"), ("VCP", "--")):
            ax.plot(cumulative_response(d, coil_separation, o, h), d, ls,
                    color=f"C{k}", label=f"{name} {o}")
    ax.axvline(0.3, color="0.6", lw=0.6)
    ax.text(0.31, max_depth * 0.97, "70 % of the reading\nfrom above here",
            fontsize=7, va="top")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of reading from below this depth")
    ax.set_ylabel("depth below ground (m)")
    ax.set_title(f"coil separation {coil_separation} m", fontsize=9)
    ax.legend(fontsize=7)
    return ax


def plot_vendor_inversion(survey: FDEMSurvey, line, *, ax=None, max_depth=None,
                          cmap="viridis_r"):
    """The instrument software's three-layer model along one line.

    Each sample is drawn as a column of three resistivities; the misfit the
    software itself reports is plotted above it.
    """
    plt = _plt()
    from matplotlib.colors import LogNorm

    v = survey.vendor
    need = ["r_layer01_ohm_m", "r_layer02_ohm_m", "r_layer03_ohm_m",
            "t_layer01_m", "t_layer02_m"]
    if not set(need) <= set(v.columns):
        raise ValueError("the export has no three-layer inversion columns")
    r = survey.readings
    if "distance" not in r:
        raise ValueError("run add_distance() first")
    m = (r["line"] == line).to_numpy()
    d = r.loc[m, "distance"].to_numpy()
    rho = v.loc[m, need[:3]].to_numpy()
    t1, t2 = v.loc[m, "t_layer01_m"].to_numpy(), v.loc[m, "t_layer02_m"].to_numpy()
    if ax is None:
        _, (ax0, ax) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 3]})
    else:
        ax0 = None
    max_depth = max_depth or float(np.nanmax(t1 + t2) * 1.2)
    edges = np.concatenate([[d[0]], (d[1:] + d[:-1]) / 2, [d[-1]]])
    norm = LogNorm(vmin=np.nanmin(rho), vmax=np.nanmax(rho))
    cm = plt.get_cmap(cmap)
    for i in range(len(d)):
        tops = [0, t1[i], t1[i] + t2[i], max_depth]
        for k in range(3):
            ax.fill_between([edges[i], edges[i + 1]], tops[k], tops[k + 1],
                            color=cm(norm(rho[i, k])), lw=0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
    plt.colorbar(sm, ax=ax, label="resistivity (ohm m)")
    ax.set_ylim(max_depth, 0)
    ax.set_ylabel("depth (m)")
    ax.set_xlabel("distance along line (m)")
    if ax0 is not None and "fiterror_pct" in v:
        ax0.plot(d, v.loc[m, "fiterror_pct"], "k.-", ms=3, lw=0.7)
        ax0.set_ylabel("fit error (%)")
        ax0.set_title(f"vendor three-layer inversion, line {line}", fontsize=9)
        plt.colorbar(sm, ax=ax0).remove()
    return ax
