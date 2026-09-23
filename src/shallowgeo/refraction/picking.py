"""First-break picking.

Automatic pickers exist to give students a starting point they then correct
by hand, not to replace the hand. Three are offered:

``"aic"``
    Akaike information criterion (Maeda, 1985): the sample that best splits
    the trace into two stationary segments, searched from the start of the
    record to the largest amplitude. It ignores how big the arrival is and
    asks only where the statistics change, which is why it is the default:
    on real hammer data the first break is often small and the later
    surface wave large, and amplitude-driven pickers jump to the latter.
``"mer"``
    Modified energy ratio (Wong et al., 2009): the leading-to-trailing
    energy ratio sharpened by the instantaneous amplitude. Excellent on
    impulsive, clean records; drawn to the strongest arrival on poor ones.
``"sta_lta"``
    Classic short-term over long-term average with a threshold crossing.
    Familiar from earthquake seismology; more parameters to tune.

Times are reported on the survey's own time axis, so a Geode ``DELAY`` is
already accounted for. :func:`plot_picks` draws the record with the picks
on it -- look at it before believing any of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.survey import SeismicSurvey


def _window_energy(trace: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Energy in the ``n`` samples ending at i (trailing) and starting after i (leading).

    Both are NaN where the window would run off the record, so a pick can
    never land in the edge region where one side has no data.
    """
    e = trace ** 2
    c = np.concatenate([[0.0], np.cumsum(e)])
    idx = np.arange(e.size)
    trailing = np.full(e.size, np.nan)
    leading = np.full(e.size, np.nan)
    ok = (idx >= n - 1) & (idx + n < e.size)
    i = idx[ok]
    trailing[ok] = c[i + 1] - c[i + 1 - n]
    leading[ok] = c[i + 1 + n] - c[i + 1]
    return trailing, leading


def _energy_ratio(trace: np.ndarray, window: int) -> np.ndarray:
    trailing, leading = _window_energy(trace, window)
    floor = 1e-9 * np.nanmax(trailing + leading) + 1e-30
    return leading / (trailing + floor)


def _mer(trace: np.ndarray, window: int) -> np.ndarray:
    """Modified energy ratio: ``(ER * |x|)^3`` (Wong et al., 2009)."""
    return (_energy_ratio(trace, window) * np.abs(trace)) ** 3


def _sta_lta(trace: np.ndarray, short: int, long: int) -> np.ndarray:
    """STA/LTA with the LTA allowed to grow from the record start.

    A fixed LTA would be undefined for the first ``long`` samples, which on a
    refraction record is exactly where the near-offset first breaks are.
    """
    e = trace ** 2
    c = np.concatenate([[0.0], np.cumsum(e)])
    i = np.arange(e.size)
    out = np.full(e.size, np.nan)
    ok = i >= short - 1
    ii = i[ok]
    sta = (c[ii + 1] - c[ii + 1 - short]) / short
    n_long = np.minimum(ii + 1, long)
    lta = (c[ii + 1] - c[ii + 1 - n_long]) / n_long
    out[ok] = sta / (lta + 1e-30)
    return out


def _aic(trace: np.ndarray) -> np.ndarray:
    """AIC of a two-segment stationary model with the break after sample k.

    NaN at the two ends, where one segment would be empty.
    """
    x = trace.astype(float)
    n = x.size
    out = np.full(n, np.nan)
    if n < 4:
        return out
    k = np.arange(1, n - 1)
    c1, c2 = np.cumsum(x), np.cumsum(x ** 2)
    var1 = c2[k - 1] / k - (c1[k - 1] / k) ** 2
    m = n - k
    var2 = (c2[-1] - c2[k - 1]) / m - ((c1[-1] - c1[k - 1]) / m) ** 2
    out[k] = k * np.log(np.maximum(var1, 1e-30)) + (m - 1) * np.log(np.maximum(var2, 1e-30))
    return out


def pick_first_breaks(
    survey: SeismicSurvey,
    *,
    method: str = "aic",
    window: float = 0.003,
    long_window: float = 0.02,
    threshold: float = 4.0,
    min_time: float | None = None,
    max_time: float | None = None,
) -> pd.DataFrame:
    """Pick the first arrival on every trace.

    Parameters
    ----------
    method
        ``"aic"`` (default), ``"mer"`` or ``"sta_lta"``; see the module notes.
    window
        Energy window in seconds (``"mer"`` window, or the STA of
        ``"sta_lta"``). A few dominant periods of the first break: 2-4 ms
        for hammer refraction. ``"aic"`` uses it only as the quality window.
    long_window
        LTA window, ``"sta_lta"`` only.
    threshold
        ``"sta_lta"`` trigger level.
    min_time, max_time
        Restrict the search, in seconds on the survey time axis. Use
        ``min_time`` to skip a trigger spike at time zero. For ``"aic"`` the
        default upper limit is each trace's own largest amplitude -- the
        first break is never later than that.

    Returns
    -------
    DataFrame with one row per trace: ``trace``, ``channel``,
    ``receiver_id``, ``source_id``, ``offset``, ``time`` (s), ``sample``, and
    ``quality`` (leading-to-trailing energy ratio across the pick: about 1
    for noise, tens to hundreds for a clean break).
    """
    data = survey.data
    times = survey.times()
    dt = survey.sample_interval
    w = max(int(round(window / dt)), 2)
    lw = max(int(round(long_window / dt)), w + 1)

    lo = 0 if min_time is None else int(np.searchsorted(times, min_time))
    hi = data.shape[1] if max_time is None else int(np.searchsorted(times, max_time))
    if hi - lo < 2 * w:
        raise ValueError("search window shorter than two energy windows")

    picks = np.empty(data.shape[0], dtype=int)
    quality = np.empty(data.shape[0])
    for i, tr in enumerate(data):
        seg = tr[lo:hi] - np.mean(tr[lo:lo + w])
        if not np.any(seg):
            picks[i], quality[i] = -1, 0.0
            continue
        if method == "aic":
            stop = seg.size if max_time is not None else int(np.argmax(np.abs(seg))) + 1
            stop = max(stop, 2 * w)
            cf = _aic(seg[:stop])
            j = int(np.nanargmin(cf)) + 1  # break is *after* sample k
        elif method == "mer":
            cf = _mer(seg, w)
            j = int(np.nanargmax(cf))
        elif method == "sta_lta":
            cf = _sta_lta(seg, w, lw)
            above = np.flatnonzero(cf > threshold)
            j = int(above[0]) if above.size else int(np.nanargmax(cf))
        else:
            raise ValueError(f"unknown picking method {method!r}")
        picks[i] = j + lo
        # Quality is the plain energy ratio across the pick: ~1 for noise,
        # tens to hundreds for a clean break. Same scale for both methods.
        er = _energy_ratio(seg, w)
        quality[i] = float(er[j]) if j < er.size and np.isfinite(er[j]) else 0.0

    tm = survey.trace_map
    out = pd.DataFrame(
        {
            "trace": np.arange(data.shape[0]),
            "channel": tm["channel"].to_numpy() if "channel" in tm else np.arange(1, data.shape[0] + 1),
            "receiver_id": tm["receiver_id"].to_numpy(),
            "source_id": tm["source_id"].to_numpy() if "source_id" in tm else None,
            "offset": survey.offsets() if not survey.is_passive else np.nan,
            "sample": picks,
            "time": np.where(picks >= 0, times[np.clip(picks, 0, None)], np.nan),
            "quality": quality,
        }
    )
    out.attrs["method"] = method
    out.attrs["source_file"] = survey.metadata.get("source_file")
    return out


def traveltime_table(*shots: tuple[SeismicSurvey, pd.DataFrame]) -> pd.DataFrame:
    """Merge picks from several shots into one table for interpretation.

    Each argument is ``(survey, picks)`` where ``picks`` is the output of
    :func:`pick_first_breaks`, possibly hand-edited. Adds the along-line
    positions of source and receiver (``source_x``, ``receiver_x``) so the
    table can be split by shot direction for a dipping-layer analysis, and a
    ``shot`` label from the source file.
    """
    frames = []
    for k, (survey, picks) in enumerate(shots):
        geom = survey.geometry.table.set_index(["role", "id"])
        rec = geom.loc["receiver"].reindex(picks["receiver_id"])
        src = geom.loc["source"].reindex(picks["source_id"])
        frame = picks.copy()
        frame["receiver_x"] = rec["x"].to_numpy()
        frame["source_x"] = src["x"].to_numpy()
        frame["shot"] = survey.metadata.get("source_file") or f"shot{k + 1}"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_picks(
    survey: SeismicSurvey,
    picks: pd.DataFrame | None = None,
    *,
    ax=None,
    x: str = "auto",
    tmax: float | None = None,
    gain: float = 1.0,
    clip: float = 2.0,
    color: str = "firebrick",
):
    """Wiggle plot of a shot record, with picks overlaid.

    Each trace is normalised to its own maximum and scaled to ``gain`` times
    the trace spacing, clipped at ``clip`` spacings so a clipped or noisy
    trace cannot cover its neighbours. ``picks`` is the frame from
    :func:`pick_first_breaks`; a hand-edited copy plots the same way.

    Parameters
    ----------
    x
        What to put on the horizontal axis.

        ``"offset"``
            Absolute source-receiver distance, the travel-time domain. Right
            for an end-on shot, but it folds a split spread: a shot in the
            middle of the line puts two geophones at every distance, one on
            each side, drawn on top of each other.
        ``"receiver"``
            Position along the line. Never folds, and the source is marked.
        ``"auto"`` (default)
            ``"receiver"`` when the source lies inside the spread, otherwise
            ``"offset"`` -- so an end-on record plots as it always has and a
            split spread is readable.

    A survey holding several shots (a multi-record SEG-Y) plots all of them
    at once, which is rarely what you want; pass ``survey.gather("S1")``.
    """
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    t = survey.times() * 1e3
    off = survey.offsets()
    pos = survey.geometry.table.set_index(["role", "id"])
    rec_x = pos.loc["receiver"].reindex(survey.trace_map["receiver_id"])["x"].to_numpy()
    # The sources these traces were actually recorded from, not every source
    # in the geometry: a gather taken from a multi-record file keeps the whole
    # line's geometry, and marking its sibling shot would be wrong twice over
    # -- a spurious source line, and a single end-on shot mistaken for a split
    # spread by the axis choice below.
    src_ids = pd.unique(survey.trace_map["source_id"])
    src_x = pos.loc["source"].reindex(src_ids)["x"].to_numpy(float)

    if x == "auto":
        # Several shots, or one shot inside the spread: an offset axis would
        # stack traces on top of each other, so plot along the line instead.
        split = np.any((rec_x.min() < src_x) & (src_x < rec_x.max()))
        x = "receiver" if split or src_x.size > 1 else "offset"
    if x == "offset":
        xs, label = off, "offset (m)"
    elif x == "receiver":
        xs, label = rec_x, "position along line (m)"
    else:
        raise ValueError(f"x must be 'auto', 'offset' or 'receiver', got {x!r}")

    # Trace spacing sets the wiggle amplitude. Taking the median of all
    # neighbouring gaps gives zero as soon as two traces share a coordinate,
    # which is exactly what a split spread does on an offset axis -- and a
    # zero-amplitude wiggle plots as a bare vertical line. Use the smallest
    # non-zero gap instead, with a last-resort fallback for the degenerate
    # case where every trace really is at one place.
    gaps = np.diff(np.sort(xs))
    gaps = gaps[gaps > 0]
    dx = float(gaps.min()) if gaps.size else 1.0

    for i in range(survey.n_traces):
        tr = survey.data[i]
        peak = np.abs(tr).max() or 1.0
        wig = np.clip(tr / peak * gain * dx, -clip * dx, clip * dx)
        ax.plot(xs[i] + wig, t, color="k", lw=0.5)
        ax.fill_betweenx(t, xs[i], xs[i] + wig, where=wig > 0, color="k", alpha=0.35, lw=0)
    if x == "receiver":
        for k, sx in enumerate(src_x):
            ax.axvline(sx, color="tab:blue", ls="--", lw=1,
                       label="source" if k == 0 else None)
    if picks is not None:
        ok = picks["time"].notna()
        px = off[ok.to_numpy()] if x == "offset" else rec_x[ok.to_numpy()]
        ax.plot(px, picks.loc[ok, "time"] * 1e3, "o",
                ms=5, color=color, mec="w", mew=0.6, label="first breaks", zorder=5)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(tmax * 1e3 if tmax else t[-1], t[0])
    ax.set_xlabel(label)
    ax.set_ylabel("time (ms)")
    ax.set_title(str(survey.metadata.get("source_file", "")).split("/")[-1])
    return ax


# A single-hue sequential ramp, light to dark, for colouring shots by where
# they were fired. Shot identity is *ordered* -- the source moves along the
# line -- so a ramp is the honest encoding, and it makes a repeat shot at the
# same source position come out the same colour, which is the point of
# shooting one. A categorical palette cannot do this job: past three or four
# series no set of hues stays separable to a colour-blind reader on a scatter,
# and a refraction line has a dozen shots. Shot direction, which is what the
# horizontal-layer check turns on, is carried by the marker instead.
_SHOT_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"]


def _shot_colours(n: int):
    """``n`` steps of the sequential ramp, none lighter than the 2:1 floor."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("shots", _SHOT_RAMP)
    return cmap(np.linspace(0, 1, n)) if n > 1 else cmap([0.6])


def plot_traveltimes(
    table: pd.DataFrame,
    *,
    ax=None,
    x: str = "offset",
    model=None,
    residual_ax=None,
    connect: bool = False,
    legend: bool = True,
):
    """Every shot's picks on one travel-time graph, optionally against a model.

    This is the plot the whole interpretation rests on. With the picks from
    all the shots drawn together in the offset domain, a horizontally layered
    ground gives branches that lie on top of one another shot to shot;
    anything that does not lie on top is either a bad pick or a hint that the
    layers are not horizontal.

    Shots are coloured light to dark by source position along the line, so a
    repeat shot plots in the same colour as the one it repeats. The marker
    points the way the wave travelled: ``>`` for a source in the near half of
    the line, ``<`` for one in the far half. Forward and reverse arrivals
    that split apart at the same offset are the signature of a dipping
    interface.

    Parameters
    ----------
    table
        Travel-time table from :func:`traveltime_table` -- needs ``offset``
        and ``time``, and uses ``shot`` and ``source_x`` when they are there.
    x
        ``"offset"`` (default) for the interpretation domain, or
        ``"receiver_x"`` to see the picks laid out along the line as they
        were recorded.
    model
        A :class:`~shallowgeo.refraction.layers.LayeredRefractionModel`; its
        predicted first arrival is drawn over the picks and its crossover
        distances marked. Only meaningful with ``x="offset"``.
    residual_ax
        A second axis for observed minus modelled time, in milliseconds.
        Requires ``model``. This is where a systematic pattern shows up: a
        residual that trends with offset means the velocities are off; one
        that separates by shot means the interface is not horizontal.
    connect
        Join each shot's picks with a faint line, in offset order. A single
        mispick then shows as a spike, which is worth having on one shot at a
        time and unreadable with a dozen lines crossing, so it is off here
        and on in the per-shot plots.
    """
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    if x not in table.columns:
        raise ValueError(f"table has no {x!r} column; got {list(table.columns)}")
    if residual_ax is not None and model is None:
        raise ValueError("a residual axis needs a model to take residuals from")

    if "shot" in table.columns:
        order = (table.groupby("shot")["source_x"].first().sort_values()
                 if "source_x" in table.columns
                 else pd.Series(index=pd.Index(table["shot"].unique(), name="shot"),
                                data=np.nan))
        shots = list(order.index)
        positions = order.to_numpy(float)
    else:
        shots, positions = [None], np.array([np.nan])
    colours = _shot_colours(len(shots))
    midpoint = np.nanmedian(table["receiver_x"]) if "receiver_x" in table else np.nan

    for k, shot in enumerate(shots):
        part = table if shot is None else table[table["shot"] == shot]
        part = part.sort_values(x)
        pointing = ">" if not np.isfinite(positions[k]) or positions[k] <= midpoint else "<"
        label = str(shot) if not np.isfinite(positions[k]) else \
            f"{shot}  ({positions[k]:+.1f} m)"
        style = {"color": colours[k], "marker": pointing, "ms": 7, "mec": "w",
                 "mew": 0.8, "ls": "-" if connect else "none", "lw": 1.0,
                 "alpha": 0.85}
        ax.plot(part[x], 1e3 * part["time"], label=label, **style)
        if residual_ax is not None:
            resid = part["time"] - model.predict(part["offset"])
            residual_ax.plot(part[x], 1e3 * resid, **{**style, "ls": "none"})

    if model is not None and x == "offset":
        xx = np.linspace(0, float(table["offset"].max()) * 1.02, 400)
        ax.plot(xx, 1e3 * model.predict(xx), "-", color="#0b0b0b", lw=2.2, zorder=5,
                label="model: " + " / ".join(f"{v:.0f}" for v in model.velocities)
                      + " m/s")
        for xc in np.atleast_1d(model.crossovers):
            if np.isfinite(xc):
                ax.axvline(xc, color="#8a8a86", ls=":", lw=1.2, zorder=1)

    axis_label = "offset (m)" if x == "offset" else "position along line (m)"
    ax.set_ylabel("travel time (ms)")
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    if residual_ax is None:
        ax.set_xlabel(axis_label)
    if legend:
        ax.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.9,
                  title="shot (source position)", title_fontsize=7)

    if residual_ax is not None:
        resid_all = table["time"] - model.predict(table["offset"])
        rms = float(np.sqrt(np.nanmean(resid_all ** 2)))
        residual_ax.axhline(0, color="#0b0b0b", lw=0.9)
        residual_ax.set_xlabel(axis_label)
        residual_ax.set_ylabel("obs - model (ms)")
        residual_ax.grid(alpha=0.25, lw=0.6)
        residual_ax.set_axisbelow(True)
        residual_ax.set_title(f"RMS {1e3 * rms:.2f} ms over {len(table)} picks",
                              fontsize=9)
    return ax
