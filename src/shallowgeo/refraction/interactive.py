"""Hand-corrected first-break picking across all the shots of one line.

An automatic picker is a starting point, not an answer. On real hammer data
some traces are noisy, one channel is always the one somebody stepped on,
and the trace at the source point is saturated. The picks that go into an
interpretation are the ones a person has looked at.

:class:`PickingSession` holds one editable table of picks per shot and keeps
them together: run the automatic picker over every shot at once, correct
individual traces, drop the ones that cannot be picked, and get a single
travel-time table out the other end. Every edit is recorded, so re-running
the automatic picker does not silently discard the corrections.

The session is plain data and works without a notebook -- :meth:`table`,
:meth:`save` and :meth:`load` are all that a script needs.
:meth:`PickingSession.widget` adds an ``ipywidgets`` interface on top for
teaching use: pick by clicking when the matplotlib backend supports it
(``%matplotlib widget``), by slider when it does not, which is what Colab
falls back to.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.survey import SeismicSurvey
from .picking import _energy_ratio, pick_first_breaks, plot_picks, traveltime_table

__all__ = ["PickingSession", "load_shots"]


def load_shots(paths: Iterable[str | Path], *, driver: str | None = None,
               exclude: Iterable[str] = ()) -> dict[str, SeismicSurvey]:
    """Read shot records and split every file into one gather per source.

    A Geode file can hold more than one field record -- ``shallowgeo`` reads
    such a file as one survey with several sources -- and refraction
    interpretation works shot by shot, so they are separated here. Each
    gather is labelled with its field-record number when the headers carry
    one, which is how a field notebook refers to it.

    Parameters
    ----------
    paths
        File paths, in any order; the result is sorted by label.
    exclude
        Skip files whose name contains any of these strings. Use it to leave
        out a record that the shot displays show to be unusable, rather than
        deleting the file.
    """
    from ..drivers import read

    exclude = tuple(exclude)
    shots: dict[str, SeismicSurvey] = {}
    for path in sorted(Path(p) for p in paths):
        if any(e in path.name for e in exclude):
            continue
        survey = read(path, driver=driver) if driver else read(path)
        if survey.is_passive:
            raise ValueError(f"{path.name} has no sources; it is not a shot record")
        for source_id in survey.geometry.sources["id"]:
            gather = survey.gather(source_id)
            record = gather.trace_map.get("record")
            label = str(record.iloc[0]) if record is not None else path.stem
            if label in shots:
                label = f"{label}-{source_id}"
            gather.metadata["shot_label"] = label
            shots[label] = gather
    if not shots:
        raise ValueError("no shot records were read")
    return dict(sorted(shots.items()))


class PickingSession:
    """Editable first-break picks for a set of shots into one spread.

    Parameters
    ----------
    shots
        A mapping ``{label: survey}``, a sequence of ``(label, survey)``, a
        sequence of surveys, or one survey. Surveys with several sources are
        split into gathers, as :func:`load_shots` does.
    method, min_time, max_time
        Passed to :func:`~shallowgeo.refraction.picking.pick_first_breaks`.
        ``max_time`` matters more than it looks: the longest offset over the
        slowest plausible velocity bounds the first break, and without that
        bound a handful of picks on a later arrival will bend the fit.
    min_quality
        Picks below this energy ratio start out excluded. They are not
        deleted -- the display keeps showing them so they can be looked at
        and kept if the picker was right after all.
    autopick
        Run the automatic picker on construction. Turn it off to load saved
        picks into a fresh session.
    """

    def __init__(
        self,
        shots,
        *,
        method: str = "aic",
        min_time: float | None = 0.001,
        max_time: float | None = None,
        min_quality: float = 10.0,
        autopick: bool = True,
    ) -> None:
        self.shots = _as_shot_map(shots)
        self.settings: dict[str, Any] = {
            "method": method,
            "min_time": min_time,
            "max_time": max_time,
            "min_quality": min_quality,
        }
        self.picks: dict[str, pd.DataFrame] = {}
        if autopick:
            self.auto_pick()
        else:
            for label in self.shots:
                self.picks[label] = self._empty_picks(label)

    # -- construction helpers ----------------------------------------------

    def _empty_picks(self, label: str) -> pd.DataFrame:
        survey = self.shots[label]
        tm = survey.trace_map
        return pd.DataFrame(
            {
                "trace": np.arange(survey.n_traces),
                "channel": tm["channel"].to_numpy() if "channel" in tm
                else np.arange(1, survey.n_traces + 1),
                "receiver_id": tm["receiver_id"].to_numpy(),
                "source_id": tm["source_id"].to_numpy(),
                "offset": survey.offsets(),
                "sample": -1,
                "time": np.nan,
                "quality": 0.0,
                "use": False,
                "edited": False,
            }
        )

    # -- picking ------------------------------------------------------------

    @property
    def labels(self) -> list[str]:
        return list(self.shots)

    def _resolve(self, label: str | None) -> list[str]:
        if label is None:
            return self.labels
        if label not in self.shots:
            raise KeyError(f"no shot {label!r}; have {self.labels}")
        return [label]

    def auto_pick(self, label: str | None = None, *, keep_edits: bool = True,
                  **overrides) -> PickingSession:
        """Run the automatic picker on one shot, or on all of them.

        Keyword arguments override the session settings and are remembered,
        so the next call uses them too -- the settings are a property of the
        line, not of one invocation.

        Hand-edited traces are left alone unless ``keep_edits=False``. Re-run
        this freely while tuning ``max_time`` or the method: corrections
        survive.
        """
        self.settings.update({k: v for k, v in overrides.items() if k in self.settings})
        unknown = set(overrides) - set(self.settings)
        if unknown:
            raise TypeError(f"unknown setting(s) {sorted(unknown)}")
        for lab in self._resolve(label):
            survey = self.shots[lab]
            fresh = pick_first_breaks(
                survey,
                method=self.settings["method"],
                min_time=self.settings["min_time"],
                max_time=self.settings["max_time"],
            )
            fresh["use"] = ((fresh["quality"] >= self.settings["min_quality"])
                            & (fresh["offset"] > 0) & fresh["time"].notna())
            fresh["edited"] = False
            old = self.picks.get(lab)
            if keep_edits and old is not None and old["edited"].any():
                keep = old["edited"].to_numpy()
                fresh.loc[keep, ["sample", "time", "quality", "use", "edited"]] = (
                    old.loc[keep, ["sample", "time", "quality", "use", "edited"]].to_numpy()
                )
            self.picks[lab] = fresh
        return self

    def guided_pick(self, predict, *, window: float = 0.004,
                    label: str | None = None, keep_edits: bool = True) -> PickingSession:
        """Re-pick every trace near a predicted first-break time.

        ``predict`` is anything with a ``predict(source_x, receiver_x)``
        method returning seconds -- a
        :class:`~shallowgeo.refraction.tomography.TomographyResult` or a
        :class:`~shallowgeo.refraction.layers.LayeredRefractionModel` -- or a
        plain function of the same two arrays. The AIC picker then searches
        only ``prediction +- window`` on each trace, which rescues the traces
        where the first pass locked onto a later, stronger arrival.

        The model only says where to look. Each pick is still the break the
        picker finds in the trace, with its own quality, so a model that is
        wrong somewhere shows up as picks that disagree with it there.
        Hand-edited traces are left alone unless ``keep_edits=False``.
        """
        fn = predict.predict if hasattr(predict, "predict") else predict
        for lab in self._resolve(label):
            survey = self.shots[lab]
            rec_x = self.receiver_x(lab)
            src_x = float(survey.geometry.table.set_index(["role", "id"]).loc[
                ("source", survey.trace_map["source_id"].iloc[0]), "x"])
            try:
                guide = np.asarray(fn(np.full(rec_x.size, src_x), rec_x), dtype=float)
            except TypeError:  # a layered model predicts from offset alone
                guide = np.asarray(fn(np.abs(rec_x - src_x)), dtype=float)
            fresh = pick_first_breaks(
                survey,
                method=self.settings["method"],
                min_time=self.settings["min_time"],
                max_time=self.settings["max_time"],
                guide=guide,
                guide_window=window,
            )
            fresh["use"] = ((fresh["quality"] >= self.settings["min_quality"])
                            & (fresh["offset"] > 0) & fresh["time"].notna())
            fresh["edited"] = False
            old = self.picks.get(lab)
            if keep_edits and old is not None and old["edited"].any():
                keep = old["edited"].to_numpy()
                fresh.loc[keep, ["sample", "time", "quality", "use", "edited"]] = (
                    old.loc[keep, ["sample", "time", "quality", "use", "edited"]].to_numpy()
                )
            self.picks[lab] = fresh
        return self

    def exclude(self, rows: pd.DataFrame) -> PickingSession:
        """Mark the picks in ``rows`` (with ``shot`` and ``channel``) as not used.

        Takes the output of ``TomographyResult.outliers()`` directly. The
        picks stay in the session and can be re-admitted with
        :meth:`set_use`.
        """
        for shot, channel in zip(rows["shot"].astype(str), rows["channel"]):
            if shot in self.shots:
                self.set_use(shot, channel, False)
        return self

    def _row(self, label: str, channel) -> int:
        picks = self.picks[label]
        hit = np.flatnonzero(picks["channel"].to_numpy() == channel)
        if hit.size == 0:
            raise KeyError(f"shot {label!r} has no channel {channel!r}")
        return int(hit[0])

    def set_pick(self, label: str, channel, time: float, *, use: bool = True) -> None:
        """Move one pick to ``time`` (seconds on the survey's time axis).

        The time is snapped to the nearest sample -- there is no information
        between samples -- and the quality is recomputed at the new position,
        so a pick dragged onto noise reports itself as such.
        """
        survey = self.shots[label]
        i = self._row(label, channel)
        times = survey.times()
        sample = min(max(round((time - survey.delay) / survey.sample_interval), 0),
                     survey.n_samples - 1)
        self.picks[label].loc[i, ["sample", "time", "quality", "use", "edited"]] = [
            sample, float(times[sample]),
            self._quality(survey, i, sample), bool(use), True,
        ]

    def clear_pick(self, label: str, channel) -> None:
        """Remove a pick: this trace cannot be picked and will not be fitted."""
        i = self._row(label, channel)
        self.picks[label].loc[i, ["sample", "time", "quality", "use", "edited"]] = [
            -1, np.nan, 0.0, False, True,
        ]

    def set_use(self, label: str, channel, use: bool) -> None:
        """Include or exclude a pick without moving or deleting it."""
        i = self._row(label, channel)
        self.picks[label].loc[i, ["use", "edited"]] = [bool(use), True]

    def use_shot(self, label: str, use: bool) -> None:
        """Include or exclude every pick of one shot -- how a bad record is dropped."""
        picks = self.picks[label]
        picks.loc[:, "use"] = bool(use) & picks["time"].notna() & (picks["offset"] > 0)
        picks.loc[:, "edited"] = True

    def _quality(self, survey: SeismicSurvey, trace: int, sample: int) -> float:
        """Energy ratio across a pick, on the same scale the auto-picker reports."""
        window = max(round(0.003 / survey.sample_interval), 2)
        er = _energy_ratio(survey.data[trace], window)
        value = er[sample] if 0 <= sample < er.size else np.nan
        return float(value) if np.isfinite(value) else 0.0

    # -- output --------------------------------------------------------------

    def table(self, *, used_only: bool = True) -> pd.DataFrame:
        """All shots' picks in one travel-time table, ready for ``fit_layers``.

        Adds ``source_x`` and ``receiver_x`` and labels each row with its
        shot. With ``used_only`` (the default) only the picks marked for use
        come back; pass ``False`` to see everything, including what was
        dropped and why the count differs.
        """
        frames = []
        for label in self.labels:
            frame = traveltime_table((self.shots[label], self.picks[label]))
            frame["shot"] = label
            frames.append(frame)
        out = pd.concat(frames, ignore_index=True)
        if used_only:
            out = out[out["use"] & out["time"].notna() & (out["offset"] > 0)]
        return out.reset_index(drop=True)

    def summary(self) -> pd.DataFrame:
        """One row per shot: where it was, how many picks it contributes, how clean."""
        rows = []
        for label in self.labels:
            survey, picks = self.shots[label], self.picks[label]
            used = picks[picks["use"] & picks["time"].notna()]
            src = survey.geometry.table.set_index(["role", "id"]).loc[
                ("source", picks["source_id"].iloc[0]), "x"]
            rows.append({
                "shot": label,
                "source_x": float(src),
                "traces": len(picks),
                "picks_used": len(used),
                "edited": int(picks["edited"].sum()),
                "max_offset_m": float(picks["offset"].max()),
                "max_time_ms": 1e3 * float(used["time"].max()) if len(used) else np.nan,
                "median_quality": float(used["quality"].median()) if len(used) else np.nan,
            })
        return pd.DataFrame(rows).sort_values("source_x").reset_index(drop=True)

    def save(self, path: str | Path) -> Path:
        """Write every pick, used or not, to one CSV that :meth:`load` reads back."""
        path = Path(path)
        self.table(used_only=False).to_csv(path, index=False)
        return path

    def load(self, path: str | Path) -> PickingSession:
        """Read picks back from :meth:`save`, matching on shot and channel.

        Shots in the file that are not in this session are ignored, and shots
        here that are not in the file keep whatever picks they have -- so a
        saved set can be dropped onto a session that has one extra record.
        """
        saved = pd.read_csv(path)
        missing = {"shot", "channel", "time", "use"} - set(saved.columns)
        if missing:
            raise ValueError(f"{path} is missing column(s) {sorted(missing)}")
        for label, part in saved.groupby("shot"):
            label = str(label)
            if label not in self.shots:
                continue
            survey = self.shots[label]
            picks = self.picks[label].set_index("channel")
            for _, row in part.iterrows():
                if row["channel"] not in picks.index:
                    continue
                if pd.isna(row["time"]):
                    picks.loc[row["channel"], ["sample", "time", "quality", "use"]] = [
                        -1, np.nan, 0.0, False]
                else:
                    sample = min(max(round(
                        (row["time"] - survey.delay) / survey.sample_interval), 0),
                        survey.n_samples - 1)
                    picks.loc[row["channel"], ["sample", "time", "quality", "use"]] = [
                        sample, float(row["time"]),
                        float(row.get("quality", 0.0) or 0.0), bool(row["use"])]
                picks.loc[row["channel"], "edited"] = bool(row.get("edited", True))
            self.picks[label] = picks.reset_index()[self.picks[label].columns]
        return self

    def __repr__(self) -> str:
        used = sum(int((p["use"] & p["time"].notna()).sum()) for p in self.picks.values())
        total = sum(len(p) for p in self.picks.values())
        edited = sum(int(p["edited"].sum()) for p in self.picks.values())
        return (f"<PickingSession {len(self.shots)} shots, {used}/{total} picks used, "
                f"{edited} hand-edited, method={self.settings['method']!r}>")

    # -- plotting and the widget ---------------------------------------------

    def receiver_x(self, label: str) -> np.ndarray:
        """Along-line position of each trace's geophone, in metres."""
        survey = self.shots[label]
        pos = survey.geometry.table.set_index(["role", "id"])
        return pos.loc["receiver"].reindex(
            survey.trace_map["receiver_id"])["x"].to_numpy(float)

    def plot_shot(self, label: str, *, ax=None, channel=None, tmax=None,
                  gain: float = 1.0):
        """Draw one shot record with its picks: kept in red, dropped in grey."""
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        survey, picks = self.shots[label], self.picks[label]
        plot_picks(survey, None, ax=ax, x="receiver", tmax=tmax, gain=gain)
        rec_x = self.receiver_x(label)
        have = picks["time"].notna().to_numpy()
        keep = have & picks["use"].to_numpy()
        drop = have & ~picks["use"].to_numpy()
        ms = 1e3 * picks["time"].to_numpy()
        ax.plot(rec_x[keep], ms[keep], "o", ms=6, color="firebrick", mec="w",
                mew=0.6, zorder=6, label="picked")
        if drop.any():
            ax.plot(rec_x[drop], ms[drop], "x", ms=7, color="0.45", mew=1.6,
                    zorder=5, label="dropped")
        if channel is not None:
            i = self._row(label, channel)
            ax.axvline(rec_x[i], color="tab:green", lw=1.2, alpha=0.5, zorder=1)
            if have[i]:
                ax.plot(rec_x[i], ms[i], "o", ms=13, mfc="none", mec="tab:green",
                        mew=2, zorder=7)
        used = int(keep.sum())
        src_x = float(survey.geometry.table.set_index(["role", "id"]).loc[
            ("source", picks["source_id"].iloc[0]), "x"])
        ax.set_title(f"shot {label} - source at {src_x:.1f} m - "
                     f"{used}/{len(picks)} picks kept", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        return ax

    def plot_trace(self, label: str, channel, *, ax=None, zoom: float = 0.03,
                   centre: float | None = None):
        """One trace around its pick, at the scale a first break is judged on.

        ``centre`` (seconds) overrides what the view is centred on, which
        matters for a trace with no pick: there is nothing to centre on, and
        the caller usually knows where the operator is looking.
        """
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        survey, picks = self.shots[label], self.picks[label]
        i = self._row(label, channel)
        row = picks.iloc[i]
        t = survey.times() * 1e3
        trace = survey.data[i]
        peak = np.abs(trace).max() or 1.0
        if centre is not None:
            centre = 1e3 * centre
        elif np.isfinite(row["time"]):
            centre = 1e3 * row["time"]
        else:
            centre = t[len(t) // 4]
        ax.plot(t, trace / peak, lw=0.9, color="k")
        ax.axhline(0, color="0.7", lw=0.6)
        if np.isfinite(row["time"]):
            ax.axvline(1e3 * row["time"],
                       color="firebrick" if row["use"] else "0.45",
                       lw=1.8, ls="-" if row["use"] else "--")
        ax.set_xlim(max(t[0], centre - 1e3 * zoom), min(t[-1], centre + 1e3 * zoom))
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("normalised amplitude")
        ax.grid(alpha=0.3)
        state = "kept" if row["use"] else "dropped"
        pick_ms = (f"{1e3 * row['time']:.3f} ms" if np.isfinite(row["time"])
                   else "no pick")
        ax.set_title(f"channel {channel} - offset {row['offset']:.2f} m - "
                     f"{pick_ms} ({state}, quality {row['quality']:.0f})", fontsize=10)
        return ax

    def widget(self, *, tmax: float | None = None, gain: float = 1.0,
               span: float | None = None):
        """An ``ipywidgets`` panel for correcting the picks by hand.

        Returns the widget; ``display`` it, or leave it as the last line of a
        cell. Edits go straight into this session, so the cells after it use
        the corrected picks as soon as they are re-run.

        Moving a pick is split across three controls, because one slider
        cannot do both jobs: a slider spanning the whole record has a sample
        every few pixels and cannot be placed accurately, while a slider fine
        enough to place cannot reach across the record.

        ``coarse``
            The whole record. Use it to move a pick that has landed on the
            wrong arrival entirely.
        ``fine``
            A window around the current pick -- by default plus or minus 20%
            of the pick time, floored at 1 ms and capped at 10 ms, so the
            travel of the slider is the accuracy you need at that offset.
            The window re-centres on the pick after every move, so repeated
            drags walk it anywhere.
        ``-`` and ``+``
            One sample earlier or later. The finest move there is; below one
            sample there is no information.

        ``span`` sets the initial fine half-width in **seconds**, overriding
        the 20% rule; the dropdown beside the slider changes it afterwards.

        Clicking the record or the trace sets a pick when the matplotlib
        backend can report mouse events -- run ``%matplotlib widget`` in the
        notebook first (in Colab also
        ``google.colab.output.enable_custom_widget_manager()``). With the
        default inline backend the plots are static images and the sliders
        do the same job.
        """
        import ipywidgets as W
        import matplotlib
        import matplotlib.pyplot as plt
        from IPython.display import clear_output, display

        backend = matplotlib.get_backend().lower()
        clickable = any(k in backend for k in ("ipympl", "widget", "nbagg"))
        survey0 = self.shots[self.labels[0]]
        dt_ms = 1e3 * survey0.sample_interval
        record_ms = 1e3 * survey0.duration
        tmax = tmax if tmax is not None else survey0.duration

        style = {"description_width": "110px"}
        wide = W.Layout(width="330px")
        very_wide = W.Layout(width="560px")
        # Plus or minus this many milliseconds around the pick; None means the
        # 20%-of-the-pick-time rule, recomputed for every trace.
        span_choices = [("auto (20% of the pick)", None), ("1 ms", 1.0),
                        ("2 ms", 2.0), ("5 ms", 5.0), ("10 ms", 10.0),
                        ("20 ms", 20.0)]
        if span is not None and not any(v == 1e3 * span for _, v in span_choices):
            span_choices.insert(1, (f"{1e3 * span:g} ms", 1e3 * span))

        w_shot = W.Dropdown(options=self.labels, description="shot", style=style,
                            layout=W.Layout(width="200px"))
        w_chan = W.IntSlider(description="channel", continuous_update=False,
                             style=style, layout=wide)
        w_prev = W.Button(description="<", layout=W.Layout(width="40px"))
        w_next = W.Button(description=">", layout=W.Layout(width="40px"))
        w_coarse = W.FloatSlider(description="coarse (ms)", min=0.0, max=record_ms,
                                 step=dt_ms, readout_format=".2f",
                                 continuous_update=False, style=style,
                                 layout=very_wide)
        w_time = W.FloatSlider(description="fine (ms)", min=0.0, max=record_ms,
                               step=dt_ms, readout_format=".3f",
                               continuous_update=False, style=style,
                               layout=very_wide)
        w_span = W.Dropdown(options=span_choices,
                            value=None if span is None else 1e3 * span,
                            description="+/-", style={"description_width": "30px"},
                            layout=W.Layout(width="190px"))
        w_earlier = W.Button(description="-", layout=W.Layout(width="40px"),
                             tooltip="move the pick one sample earlier")
        w_later = W.Button(description="+", layout=W.Layout(width="40px"),
                           tooltip="move the pick one sample later")
        w_use = W.Checkbox(description="keep this pick", indent=False,
                           layout=W.Layout(width="150px"))
        w_clear = W.Button(description="no pick here", layout=W.Layout(width="130px"),
                           tooltip="this trace cannot be picked")
        w_auto1 = W.Button(description="auto-pick this trace",
                           layout=W.Layout(width="160px"))

        w_method = W.Dropdown(options=["aic", "mer", "sta_lta"],
                              value=self.settings["method"], description="method",
                              style=style, layout=W.Layout(width="220px"))
        w_tmin = W.FloatSlider(description="search from (ms)", min=0.0,
                               max=0.2 * record_ms,
                               value=1e3 * (self.settings["min_time"] or 0.0),
                               step=dt_ms, continuous_update=False, style=style,
                               layout=wide)
        w_tmax = W.FloatSlider(description="search to (ms)", min=dt_ms,
                               max=record_ms,
                               value=1e3 * (self.settings["max_time"] or survey0.duration),
                               step=dt_ms, continuous_update=False, style=style,
                               layout=wide)
        w_qual = W.FloatSlider(description="keep quality >", min=0.0, max=100.0,
                               value=self.settings["min_quality"], step=1.0,
                               continuous_update=False, style=style, layout=wide)
        w_repick = W.Button(description="re-pick this shot", button_style="info",
                            layout=W.Layout(width="160px"))
        w_repick_all = W.Button(description="re-pick every shot",
                                layout=W.Layout(width="160px"))
        w_keep_shot = W.Button(description="keep whole shot",
                               layout=W.Layout(width="150px"))
        w_drop_shot = W.Button(description="drop whole shot",
                               layout=W.Layout(width="150px"))
        w_gain = W.FloatSlider(description="wiggle gain", min=0.2, max=6.0,
                               value=gain, step=0.2, continuous_update=False,
                               style=style, layout=wide)
        w_tmaxplot = W.FloatSlider(description="record to (ms)", min=10.0,
                                   max=record_ms, value=1e3 * tmax, step=dt_ms,
                                   continuous_update=False, style=style, layout=wide)
        w_status = W.HTML()

        with plt.ioff():
            fig, (ax_rec, ax_tr) = plt.subplots(
                1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [3, 2]})
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
        out = None if clickable else W.Output()
        busy = {"flag": False}
        window = {"lo": 0.0, "hi": record_ms, "centre": record_ms / 4}

        def channels(label):
            return self.picks[label]["channel"].to_numpy()

        def span_ms(centre):
            """Half-width of the fine slider, in milliseconds."""
            if w_span.value is not None:
                return float(w_span.value)
            if np.isfinite(centre) and centre > 0:
                return float(np.clip(0.2 * centre, max(1.0, 4 * dt_ms), 10.0))
            return 10.0

        def render():
            label, channel = w_shot.value, w_chan.value
            ax_rec.clear()
            ax_tr.clear()
            self.plot_shot(label, ax=ax_rec, channel=channel,
                           tmax=1e-3 * w_tmaxplot.value, gain=w_gain.value)
            # The trace panel shows half again the fine slider's travel, and
            # shades the part of it the slider can actually reach, so the two
            # controls cannot disagree about where you are looking.
            half = 1.5 * span_ms(window["centre"])
            self.plot_trace(label, channel, ax=ax_tr, zoom=1e-3 * half,
                            centre=1e-3 * window["centre"])
            ax_tr.axvspan(window["lo"], window["hi"], color="#cde2fb", alpha=0.45,
                          lw=0, zorder=0)
            fig.tight_layout()
            table = self.table()
            w_status.value = (
                f"<b>{len(table)}</b> picks kept over <b>{len(self.shots)}</b> shots"
                f" &nbsp;|&nbsp; offsets {table['offset'].min():.1f}"
                f"-{table['offset'].max():.1f} m"
                f" &nbsp;|&nbsp; hand-edited "
                f"{sum(int(p['edited'].sum()) for p in self.picks.values())}"
                if len(table) else
                "<b>no picks kept</b> - lower the quality threshold or pick by hand"
            )
            if clickable:
                fig.canvas.draw_idle()
            else:
                with out:
                    clear_output(wait=True)
                    display(fig)

        def sync_controls(*, reset_channel=False):
            """Push the session's state into the controls without re-triggering them."""
            busy["flag"] = True
            try:
                label = w_shot.value
                chans = channels(label)
                w_chan.min, w_chan.max = int(chans.min()), int(chans.max())
                if reset_channel or w_chan.value not in chans:
                    w_chan.value = int(chans.min())
                row = self.picks[label].iloc[self._row(label, w_chan.value)]
                centre = (float(1e3 * row["time"]) if np.isfinite(row["time"])
                          else float(w_coarse.value))
                half = span_ms(centre)
                lo = max(0.0, centre - half)
                hi = min(record_ms, centre + half)
                if hi - lo < 2 * dt_ms:      # degenerate at the record's edge
                    lo, hi = 0.0, max(2 * dt_ms, hi)
                # Widen before moving, narrow after: ipywidgets clamps a value
                # that falls outside the bounds it is given, which would drag
                # the pick to the edge of the old window.
                w_time.min, w_time.max = 0.0, record_ms
                w_time.value = centre
                w_time.min, w_time.max = lo, hi
                w_coarse.value = centre
                window.update(lo=lo, hi=hi, centre=centre)
                w_use.value = bool(row["use"])
            finally:
                busy["flag"] = False

        def refresh(*, reset_channel=False):
            sync_controls(reset_channel=reset_channel)
            render()

        def on_shot(_):
            if not busy["flag"]:
                refresh(reset_channel=True)

        def on_channel(_):
            if not busy["flag"]:
                refresh()

        def on_time(change):
            if busy["flag"]:
                return
            self.set_pick(w_shot.value, w_chan.value, 1e-3 * change["new"])
            refresh()

        def on_coarse(change):
            if busy["flag"]:
                return
            self.set_pick(w_shot.value, w_chan.value, 1e-3 * change["new"])
            refresh()

        def nudge(n_samples):
            """Move the pick by whole samples -- the finest move there is."""
            def handler(_):
                label, channel = w_shot.value, w_chan.value
                row = self.picks[label].iloc[self._row(label, channel)]
                base = (float(row["time"]) if np.isfinite(row["time"])
                        else 1e-3 * w_time.value)
                dt = self.shots[label].sample_interval
                self.set_pick(label, channel, base + n_samples * dt)
                refresh()
            return handler

        def on_use(change):
            if busy["flag"]:
                return
            self.set_use(w_shot.value, w_chan.value, change["new"])
            render()

        def step(delta):
            def handler(_):
                chans = channels(w_shot.value)
                i = int(np.searchsorted(chans, w_chan.value)) + delta
                if 0 <= i < chans.size:
                    w_chan.value = int(chans[i])
            return handler

        def on_clear(_):
            self.clear_pick(w_shot.value, w_chan.value)
            refresh()

        def on_auto_trace(_):
            label, channel = w_shot.value, w_chan.value
            i = self._row(label, channel)
            single = pick_first_breaks(
                _one_trace(self.shots[label], i), method=w_method.value,
                min_time=1e-3 * w_tmin.value, max_time=1e-3 * w_tmax.value)
            self.set_pick(label, channel, float(single["time"].iloc[0]),
                          use=bool(single["quality"].iloc[0] >= w_qual.value))
            refresh()

        def repick(scope):
            def handler(_):
                self.auto_pick(scope and w_shot.value, method=w_method.value,
                               min_time=1e-3 * w_tmin.value,
                               max_time=1e-3 * w_tmax.value,
                               min_quality=w_qual.value)
                refresh()
            return handler

        def shot_use(use):
            def handler(_):
                self.use_shot(w_shot.value, use)
                refresh()
            return handler

        def on_click(event):
            if event.inaxes is ax_rec and event.xdata is not None:
                rec_x = self.receiver_x(w_shot.value)
                i = int(np.argmin(np.abs(rec_x - event.xdata)))
                chan = int(self.picks[w_shot.value]["channel"].iloc[i])
                busy["flag"] = True
                w_chan.value = chan
                busy["flag"] = False
                self.set_pick(w_shot.value, chan, 1e-3 * event.ydata)
                refresh()
            elif event.inaxes is ax_tr and event.xdata is not None:
                self.set_pick(w_shot.value, w_chan.value, 1e-3 * event.xdata)
                refresh()

        w_shot.observe(on_shot, "value")
        w_chan.observe(on_channel, "value")
        w_time.observe(on_time, "value")
        w_coarse.observe(on_coarse, "value")
        w_span.observe(lambda _: refresh(), "value")
        w_earlier.on_click(nudge(-1))
        w_later.on_click(nudge(+1))
        w_use.observe(on_use, "value")
        w_prev.on_click(step(-1))
        w_next.on_click(step(+1))
        w_clear.on_click(on_clear)
        w_auto1.on_click(on_auto_trace)
        w_repick.on_click(repick(True))
        w_repick_all.on_click(repick(False))
        w_keep_shot.on_click(shot_use(True))
        w_drop_shot.on_click(shot_use(False))
        for control in (w_gain, w_tmaxplot):
            control.observe(lambda _: render(), "value")
        if clickable:
            fig.canvas.mpl_connect("button_press_event", on_click)

        hint = ("Click the record to pick that channel, or the right-hand panel to "
                "refine the selected one. <b>coarse</b> reaches the whole record, "
                "<b>fine</b> covers the shaded band around the pick, "
                "<b>-</b>/<b>+</b> move one sample."
                if clickable else
                "Inline plots cannot report clicks, so the sliders do the picking: "
                "<b>coarse</b> reaches the whole record, <b>fine</b> covers the "
                "shaded band around the pick and re-centres after every move, "
                "<b>-</b>/<b>+</b> move one sample. For click-picking run "
                "<code>%matplotlib widget</code> and re-run this cell.")
        auto_box = W.VBox([W.HBox([w_method, w_repick, w_repick_all]),
                           W.HBox([w_tmin, w_tmax, w_qual])])
        display_box = W.HBox([w_gain, w_tmaxplot])
        settings = W.Accordion(children=[auto_box, display_box])
        settings.set_title(0, "automatic picker")
        settings.set_title(1, "display")
        settings.selected_index = None

        panel = W.VBox([
            W.HBox([w_shot, w_keep_shot, w_drop_shot]),
            W.HBox([w_chan, w_prev, w_next, w_use, w_clear, w_auto1]),
            W.HBox([w_coarse]),
            W.HBox([w_time, w_earlier, w_later, w_span]),
            settings,
            W.HTML(f"<i>{hint}</i>"),
            fig.canvas if clickable else out,
            w_status,
        ])
        refresh(reset_channel=True)
        return panel


def _one_trace(survey: SeismicSurvey, index: int) -> SeismicSurvey:
    """A one-trace survey, so the picker can be re-run on a single channel."""
    return SeismicSurvey(
        survey.data[index:index + 1],
        survey.sample_interval,
        survey.geometry,
        survey.trace_map.iloc[index:index + 1],
        delay=survey.delay,
        start_time=survey.start_time,
        metadata=dict(survey.metadata),
    )


def _as_shot_map(shots) -> dict[str, SeismicSurvey]:
    """Accept a survey, a list of surveys, pairs, or a mapping; split multi-source."""
    if isinstance(shots, SeismicSurvey):
        shots = [shots]
    if isinstance(shots, dict):
        items = list(shots.items())
    else:
        items = []
        for entry in shots:
            if isinstance(entry, SeismicSurvey):
                items.append((None, entry))
            else:
                label, survey = entry
                items.append((str(label), survey))

    out: dict[str, SeismicSurvey] = {}
    for label, survey in items:
        if not isinstance(survey, SeismicSurvey):
            raise TypeError(f"expected a SeismicSurvey, got {type(survey).__name__}")
        if survey.is_passive:
            raise ValueError("a passive record has no shots to pick")
        source_ids = list(pd.unique(survey.trace_map["source_id"]))
        for source_id in source_ids:
            gather = survey.gather(source_id) if len(source_ids) > 1 else survey
            if label is not None:
                name = label if len(source_ids) == 1 else f"{label}-{source_id}"
            else:
                record = gather.trace_map.get("record")
                name = (str(record.iloc[0]) if record is not None
                        else str(gather.metadata.get("shot_label")
                                 or Path(str(survey.metadata.get("source_file",
                                                                 "shot"))).stem))
                if len(source_ids) > 1 and name in out:
                    name = f"{name}-{source_id}"
            while name in out:
                name = f"{name}'"
            gather.metadata["shot_label"] = name
            out[name] = gather
    return out
