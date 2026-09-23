import numpy as np
import pandas as pd
import pytest

from shallowgeo.core import Geometry, SeismicSurvey, local_grid
from shallowgeo.refraction import (
    PickingSession,
    crossover_distances,
    depths_from_intercepts,
    fit_layers,
    intercept_times,
    load_shots,
    pick_first_breaks,
    plot_picks,
    plot_traveltimes,
    traveltime_table,
    traveltimes,
)
from shallowgeo.refraction.layers import branch_times, critical_distances
from shallowgeo.refraction.tomography import traveltime_tomography

V3, H3 = [600.0, 1500.0, 3000.0], [5.0, 12.0]


class TestForward:
    def test_two_layer_matches_textbook_formulas(self):
        v1, v2, h = 600.0, 2400.0, 15.0
        theta = np.arcsin(v1 / v2)
        assert intercept_times([v1, v2], [h])[0] == pytest.approx(2 * h * np.cos(theta) / v1)
        assert critical_distances([v1, v2], [h])[0] == pytest.approx(2 * h * np.tan(theta))
        assert crossover_distances([v1, v2], [h])[0] == pytest.approx(
            2 * h * np.sqrt((v2 + v1) / (v2 - v1))
        )

    def test_first_arrival_is_direct_then_head_wave(self):
        x = np.array([1.0, 10.0, 100.0])
        t = traveltimes(x, [600.0, 2400.0], [15.0])
        assert t[0] == pytest.approx(1.0 / 600.0)          # direct
        assert t[2] < 100.0 / 600.0                         # head wave has overtaken

    def test_head_wave_absent_inside_critical_distance(self):
        bt = branch_times([1.0, 50.0], [600.0, 2400.0], [15.0])
        assert np.isnan(bt[1, 0]) and np.isfinite(bt[1, 1])

    def test_velocity_inversion_has_no_head_wave(self):
        assert np.isnan(intercept_times([1000.0, 500.0], [5.0])[0])
        t = traveltimes([50.0], [1000.0, 500.0], [5.0])
        assert t[0] == pytest.approx(0.05)

    def test_depths_invert_intercepts_for_three_layers(self):
        ti = intercept_times(V3, H3)
        np.testing.assert_allclose(depths_from_intercepts(V3, ti), H3)

    def test_depths_reject_velocity_inversion(self):
        with pytest.raises(ValueError, match="increase with depth"):
            depths_from_intercepts([1000.0, 500.0], [0.01])

    def test_thickness_count_is_checked(self):
        with pytest.raises(ValueError, match="thicknesses"):
            traveltimes([1.0], V3, [5.0])


class TestFitLayers:
    x = np.arange(2.0, 120.0, 4.0)

    def test_recovers_three_layers_from_noisy_picks(self):
        rng = np.random.default_rng(1)
        t = traveltimes(self.x, V3, H3) + rng.normal(0, 0.5e-3, self.x.size)
        m = fit_layers(self.x, t, n_layers=3)
        np.testing.assert_allclose(m.velocities, V3, rtol=0.05)
        np.testing.assert_allclose(m.thicknesses, H3, rtol=0.1)
        assert m.rms < 1e-3
        assert len(m.breakpoints) == 2

    def test_recovers_two_layers(self):
        v, h = [500.0, 2000.0], [8.0]
        t = traveltimes(self.x, v, h)
        m = fit_layers(self.x, t, n_layers=2)
        np.testing.assert_allclose(m.velocities, v, rtol=0.01)
        assert m.thicknesses[0] == pytest.approx(8.0, rel=0.02)
        assert m.crossovers[0] == pytest.approx(crossover_distances(v, h)[0], rel=0.05)

    def test_single_layer_is_a_line_through_origin(self):
        m = fit_layers(self.x, self.x / 800.0, n_layers=1)
        assert m.velocities[0] == pytest.approx(800.0)
        assert m.thicknesses.size == 0

    def test_predict_and_summary(self):
        m = fit_layers(self.x, traveltimes(self.x, V3, H3), n_layers=3)
        np.testing.assert_allclose(m.predict(self.x), m.times, atol=1e-4)
        table = m.summary()
        assert list(table["layer"]) == [1, 2, 3]
        assert np.isnan(table["thickness_m"].iloc[-1])
        assert m.branch_of().max() == 2

    def test_velocity_inversion_in_picks_raises(self):
        # A line that gets *slower* with offset cannot be layered.
        t = np.where(self.x < 50, self.x / 2000.0, self.x / 800.0 - 0.0375)
        with pytest.raises(ValueError, match="no partition"):
            fit_layers(self.x, t, n_layers=2)

    def test_too_few_picks_raises(self):
        with pytest.raises(ValueError, match="cannot support"):
            fit_layers([1.0, 2.0, 3.0], [0.001, 0.002, 0.003], n_layers=2)

    def test_trigger_error_shows_as_direct_intercept(self):
        t = traveltimes(self.x, V3, H3) + 0.005
        m = fit_layers(self.x, t, n_layers=3, direct_through_origin=False)
        assert m.metadata["direct_intercept_s"] == pytest.approx(0.005, abs=5e-4)

    def test_nan_and_zero_offset_picks_are_dropped(self):
        x = np.concatenate([[0.0], self.x])
        t = np.concatenate([[np.nan], traveltimes(self.x, V3, H3)])
        m = fit_layers(x, t, n_layers=3)
        assert m.offsets.size == self.x.size


def _synthetic_shot(delay=0.001, noise=0.02, seed=1):
    rng = np.random.default_rng(seed)
    dt, n = 0.00025, 800
    times = np.arange(n) * dt + delay
    xs = np.arange(24) * 2.0 + 3.0
    tt = traveltimes(xs, V3, H3)

    def causal(t, f=80.0, decay=0.006):
        return np.where(t >= 0, np.sin(2 * np.pi * f * t) * np.exp(-t / decay), 0.0)

    data = np.array([causal(times - t0) for t0 in tt]) + rng.normal(0, noise, (24, n))
    geom = Geometry(
        ids=[*range(1, 25), "S1"], x=[*xs, 0.0], y=[0.0] * 25, z=[0.0] * 25,
        roles=[*["receiver"] * 24, "source"], spatial_ref=local_grid(0, 0),
    )
    tmap = pd.DataFrame({"receiver_id": range(1, 25), "source_id": "S1",
                         "channel": range(1, 25)})
    survey = SeismicSurvey(data, dt, geom, tmap, delay=delay,
                           metadata={"source_file": "synthetic.dat"})
    return survey, tt


class TestPicking:
    @pytest.mark.parametrize("method,tol_ms", [("aic", 1.0), ("mer", 1.0), ("sta_lta", 3.0)])
    def test_picks_within_tolerance(self, method, tol_ms):
        survey, truth = _synthetic_shot()
        picks = pick_first_breaks(survey, method=method, window=0.002, threshold=5.0)
        err_ms = 1e3 * np.abs(picks["time"].to_numpy() - truth)
        assert err_ms.max() < tol_ms, err_ms

    def test_delay_is_included_in_pick_times(self):
        # 2 ms delay: every onset (earliest 5 ms) is still inside the record.
        survey, truth = _synthetic_shot(delay=0.002)
        picks = pick_first_breaks(survey)
        # Sample index times dt would be 2 ms early; the time column must not be.
        assert np.abs(picks["time"].to_numpy() - truth).max() < 1e-3
        assert (picks["time"] > picks["sample"] * survey.sample_interval).all()

    def test_quality_is_high_for_clean_breaks(self):
        survey, _ = _synthetic_shot(noise=0.005)
        picks = pick_first_breaks(survey, method="aic", window=0.002)
        assert (picks["quality"] > 5).all()

    def test_min_time_skips_early_spike(self):
        survey, truth = _synthetic_shot()
        survey.data[:, 0:4] += 50.0
        picks = pick_first_breaks(survey, min_time=0.003)
        assert np.abs(picks["time"].to_numpy() - truth).max() < 1e-3

    def test_columns(self):
        survey, _ = _synthetic_shot()
        picks = pick_first_breaks(survey)
        assert {"trace", "channel", "receiver_id", "source_id", "offset",
                "sample", "time", "quality"} <= set(picks.columns)
        assert picks.attrs["method"] == "aic"

    def test_unknown_method(self):
        survey, _ = _synthetic_shot()
        with pytest.raises(ValueError, match="unknown picking method"):
            pick_first_breaks(survey, method="magic")

    def test_traveltime_table_merges_shots(self):
        s1, _ = _synthetic_shot(seed=1)
        s2, _ = _synthetic_shot(seed=2)
        s2.metadata["source_file"] = "second.dat"
        table = traveltime_table((s1, pick_first_breaks(s1)), (s2, pick_first_breaks(s2)))
        assert len(table) == 48
        assert set(table["shot"]) == {"synthetic.dat", "second.dat"}
        assert {"source_x", "receiver_x"} <= set(table.columns)
        assert table["source_x"].eq(0.0).all()

    def test_picks_to_layers_end_to_end(self):
        survey, _ = _synthetic_shot()
        picks = pick_first_breaks(survey)
        m = fit_layers(picks["offset"], picks["time"], n_layers=2)
        assert m.velocities[0] == pytest.approx(600.0, rel=0.05)

    def test_plot_picks_runs(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        survey, _ = _synthetic_shot()
        ax = plot_picks(survey, pick_first_breaks(survey), tmax=0.1)
        assert ax.get_ylim()[0] > ax.get_ylim()[1]  # time increases downward

    def test_plot_picks_end_on_shot_uses_offset_axis(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        survey, _ = _synthetic_shot()  # source at 0, geophones from 3 m out
        ax = plot_picks(survey, pick_first_breaks(survey))
        assert ax.get_xlabel() == "offset (m)"

    def test_plot_picks_split_spread_does_not_collapse(self):
        """A shot inside the spread puts two geophones at every offset.

        The wiggle amplitude is set from the trace spacing; taking it as the
        median gap over an offset axis gives exactly zero here, and every
        trace then draws as a bare vertical line.
        """
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        survey, _ = _synthetic_shot()
        # Mid-spread shot, halfway between the two central geophones -- the
        # symmetric layout a field crew actually uses, which pairs every
        # geophone with one at the same offset on the other side.
        table = survey.geometry.table
        centre = table.loc[table["role"] == "receiver", "x"].iloc[11:13].mean()
        table.loc[table["role"] == "source", "x"] = centre
        assert np.median(np.diff(np.sort(survey.offsets()))) == 0.0

        ax = plot_picks(survey, pick_first_breaks(survey))
        assert ax.get_xlabel() == "position along line (m)"
        # The first n_traces lines are the wiggles, drawn before the source
        # marker and the picks; fill_betweenx adds a collection, not a line.
        wiggles = ax.lines[: survey.n_traces]
        assert len(wiggles) == survey.n_traces
        widths = [np.ptp(np.asarray(line.get_xdata())) for line in wiggles]
        assert min(widths) > 0, "a wiggle collapsed to a vertical line"

    def test_plot_picks_axis_can_be_forced(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        survey, _ = _synthetic_shot()
        assert plot_picks(survey, x="receiver").get_xlabel() == "position along line (m)"
        with pytest.raises(ValueError, match="x must be"):
            plot_picks(survey, x="sideways")


class TestTomographyPlaceholder:
    def test_raises_with_guidance(self):
        table = pd.DataFrame({"shot": ["a", "b"], "time": [0.01, 0.02]})
        with pytest.raises(NotImplementedError, match="fit_layers"):
            traveltime_tomography(table)


class TestFixedCrossovers:
    """Branch boundaries set by hand, which is how the method is taught."""

    x = np.arange(1.0, 80.0, 1.0)

    def test_true_crossovers_recover_the_model(self):
        t = traveltimes(self.x, V3, H3)
        m = fit_layers(self.x, t, 3, crossovers=crossover_distances(V3, H3))
        np.testing.assert_allclose(m.velocities, V3, rtol=1e-6)
        np.testing.assert_allclose(m.thicknesses, H3, rtol=1e-6)
        assert m.metadata["crossovers_fixed"]

    def test_searched_fit_is_not_told_the_crossovers(self):
        t = traveltimes(self.x, V3, H3)
        assert fit_layers(self.x, t, 3).metadata["crossovers_fixed"] is False

    def test_wrong_count_is_rejected(self):
        t = traveltimes(self.x, V3, H3)
        with pytest.raises(ValueError, match="need 2 crossover offsets"):
            fit_layers(self.x, t, 3, crossovers=[20.0])

    def test_crossovers_must_increase(self):
        t = traveltimes(self.x, V3, H3)
        with pytest.raises(ValueError, match="must increase"):
            fit_layers(self.x, t, 3, crossovers=[60.0, 20.0])

    def test_branch_left_too_short_says_so(self):
        t = traveltimes(self.x, V3, H3)
        with pytest.raises(ValueError, match="needs at least"):
            fit_layers(self.x, t, 2, crossovers=[1.5])

    def test_velocity_inversion_names_the_velocities(self):
        """Picks that slow down with offset have no head wave. Say which ones."""
        x = np.arange(1.0, 41.0)
        t = np.where(x < 20, x / 2000.0, 20 / 2000.0 + (x - 20) / 500.0)
        with pytest.raises(ValueError, match=r"\[2000.0, 500.0\].*do not increase"):
            fit_layers(x, t, 2, crossovers=[20.0])

    def test_head_wave_through_the_origin_is_rejected(self):
        """A second branch with no intercept would put the interface at zero depth."""
        x = np.arange(1.0, 41.0)
        t = np.where(x < 15, x / 600.0, x / 2400.0)
        with pytest.raises(ValueError, match="not all positive"):
            fit_layers(x, t, 2, crossovers=[15.0])

    def test_branch_that_arrives_early_is_caught(self):
        """A block of mispicks on a noise burst can fit with a negative slope."""
        x = np.arange(1.0, 41.0)
        t = traveltimes(x, [600.0, 2400.0], [10.0])
        t[-8:] -= 0.02
        with pytest.raises(ValueError, match="negative slope"):
            fit_layers(x, t, 2, crossovers=[30.0])


def _line_shots(n_shots=3, seed=7):
    """Several shots into one 24-geophone spread over a two-layer ground."""
    rng = np.random.default_rng(seed)
    dt, n = 0.00025, 600
    xs = np.arange(24) * 2.0
    surveys = {}
    for k in range(n_shots):
        src_x = -6.0 - 10.0 * k
        offsets = np.abs(xs - src_x)
        tt = traveltimes(offsets, [600.0, 2400.0], [8.0])
        times = np.arange(n) * dt

        def causal(t, f=90.0, decay=0.005):
            return np.where(t >= 0, np.sin(2 * np.pi * f * t) * np.exp(-t / decay), 0.0)

        data = np.array([causal(times - t0) for t0 in tt])
        data += rng.normal(0, 0.02, data.shape)
        geom = Geometry(
            ids=[*range(1, 25), "S1"], x=[*xs, src_x], y=[0.0] * 25, z=[0.0] * 25,
            roles=[*["receiver"] * 24, "source"], spatial_ref=local_grid(0, 0),
        )
        tmap = pd.DataFrame({"receiver_id": range(1, 25), "source_id": "S1",
                             "channel": range(1, 25)})
        surveys[f"shot{k + 1}"] = SeismicSurvey(
            data, dt, geom, tmap, metadata={"source_file": f"shot{k + 1}.dat"})
    return surveys


class TestPickingSession:
    def test_autopicks_every_shot_on_construction(self):
        session = PickingSession(_line_shots())
        assert session.labels == ["shot1", "shot2", "shot3"]
        assert all(len(p) == 24 for p in session.picks.values())
        assert session.table()["time"].notna().all()

    def test_table_feeds_fit_layers(self):
        session = PickingSession(_line_shots())
        table = session.table()
        model = fit_layers(table["offset"], table["time"], n_layers=2)
        assert model.velocities[0] == pytest.approx(600.0, rel=0.05)
        assert model.thicknesses[0] == pytest.approx(8.0, rel=0.3)

    def test_set_pick_snaps_to_a_sample_and_rescores(self):
        session = PickingSession(_line_shots(1))
        session.set_pick("shot1", 5, 0.02001)
        row = session.picks["shot1"].query("channel == 5").iloc[0]
        assert row["time"] == pytest.approx(0.02, abs=1e-9)  # nearest 0.25 ms sample
        assert row["sample"] == 80
        assert row["edited"] and row["use"]
        # Dragged onto quiet ground well before the arrival, quality collapses.
        session.set_pick("shot1", 5, 0.001)
        assert session.picks["shot1"].query("channel == 5")["quality"].iloc[0] < 5

    def test_clear_pick_removes_it_from_the_table(self):
        session = PickingSession(_line_shots(1))
        before = len(session.table())
        session.clear_pick("shot1", 5)
        assert len(session.table()) == before - 1
        assert session.picks["shot1"].query("channel == 5")["time"].isna().all()

    def test_use_shot_drops_a_bad_record_without_deleting_it(self):
        session = PickingSession(_line_shots())
        session.use_shot("shot2", False)
        assert set(session.table()["shot"]) == {"shot1", "shot3"}
        assert session.picks["shot2"]["time"].notna().all()
        session.use_shot("shot2", True)
        assert "shot2" in set(session.table()["shot"])

    def test_hand_edits_survive_a_repick(self):
        """Tuning the picker must not silently throw away corrections."""
        session = PickingSession(_line_shots(1))
        session.set_pick("shot1", 5, 0.0125)
        session.auto_pick(max_time=0.09)
        assert session.picks["shot1"].query("channel == 5")["time"].iloc[0] == 0.0125
        session.auto_pick(keep_edits=False)
        assert session.picks["shot1"].query("channel == 5")["time"].iloc[0] != 0.0125

    def test_settings_are_remembered_and_checked(self):
        session = PickingSession(_line_shots(1), autopick=False)
        session.auto_pick(method="mer")
        assert session.settings["method"] == "mer"
        with pytest.raises(TypeError, match="unknown setting"):
            session.auto_pick(windowsize=3)

    def test_unknown_shot_is_named(self):
        session = PickingSession(_line_shots(1))
        with pytest.raises(KeyError, match="shot9"):
            session.auto_pick("shot9")
        with pytest.raises(KeyError, match="channel 99"):
            session.set_pick("shot1", 99, 0.01)

    def test_save_and_load_round_trip(self, tmp_path):
        session = PickingSession(_line_shots())
        session.set_pick("shot1", 3, 0.011)
        session.clear_pick("shot2", 4)
        session.use_shot("shot3", False)
        path = session.save(tmp_path / "picks.csv")

        reloaded = PickingSession(_line_shots(), autopick=False).load(path)
        pd.testing.assert_frame_equal(
            session.table().reset_index(drop=True),
            reloaded.table().reset_index(drop=True),
            check_dtype=False,
        )

    def test_load_ignores_shots_it_does_not_have(self, tmp_path):
        path = tmp_path / "picks.csv"
        PickingSession(_line_shots()).save(path)
        small = PickingSession(_line_shots(1), autopick=False).load(path)
        assert small.labels == ["shot1"]
        assert small.table()["time"].notna().all()

    def test_load_checks_the_columns(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"shot": ["shot1"], "channel": [1]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing column"):
            PickingSession(_line_shots(1), autopick=False).load(path)

    def test_summary_reports_one_row_per_shot_in_line_order(self):
        session = PickingSession(_line_shots())
        summary = session.summary()
        assert list(summary["shot"]) == ["shot3", "shot2", "shot1"]  # by source_x
        assert (summary["picks_used"] > 0).all()

    def test_passive_record_is_refused(self):
        survey, _ = _synthetic_shot()
        passive = SeismicSurvey(survey.data, survey.sample_interval, survey.geometry,
                                survey.trace_map.drop(columns=["source_id"]))
        with pytest.raises(ValueError, match="no shots to pick"):
            PickingSession(passive)

    def test_multi_source_survey_is_split_into_gathers(self, segy_file_two_records):
        path, _ = segy_file_two_records
        session = PickingSession(load_shots([path]), max_time=0.01)
        assert session.labels == ["4017", "4018"]
        for label in session.labels:
            assert session.shots[label].n_traces == 24
            assert session.picks[label]["source_id"].nunique() == 1
        # The two records were fired from different places, so the offsets differ.
        offsets = {lab: session.picks[lab]["offset"].max() for lab in session.labels}
        assert offsets["4017"] != offsets["4018"]

    def test_load_shots_can_exclude_a_bad_file(self, segy_file, segy_file_two_records):
        good, _ = segy_file
        pair, _ = segy_file_two_records
        assert list(load_shots([good, pair])) == ["4001", "4017", "4018"]
        assert list(load_shots([good, pair], exclude=["4017"])) == ["4001"]

    def test_plot_shot_marks_kept_and_dropped_picks(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        session = PickingSession(_line_shots(1))
        session.set_use("shot1", 1, False)
        ax = session.plot_shot("shot1", channel=2)
        labels = ax.get_legend_handles_labels()[1]
        assert "picked" in labels and "dropped" in labels
        assert "23/24 picks kept" in ax.get_title()

    def test_plot_trace_zooms_around_the_pick(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        session = PickingSession(_line_shots(1))
        session.set_pick("shot1", 6, 0.02)
        ax = session.plot_trace("shot1", 6, zoom=0.005)
        lo, hi = ax.get_xlim()
        assert lo <= 20.0 <= hi and (hi - lo) == pytest.approx(10.0, abs=0.5)


class TestPlotTraveltimes:
    @pytest.fixture
    def table(self):
        return PickingSession(_line_shots()).table()

    def test_one_line_per_shot_plus_the_model(self, table):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        model = fit_layers(table["offset"], table["time"], n_layers=2)
        ax = plot_traveltimes(table, ax=plt.figure().gca(), model=model)
        labels = ax.get_legend_handles_labels()[1]
        assert len(labels) == 4  # three shots and the model curve
        assert any("m/s" in lab for lab in labels)

    def test_shots_are_coloured_in_line_order(self, table):
        """Colour tracks source position, so a repeat shot plots in one colour."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = plot_traveltimes(table, ax=plt.figure().gca())
        order = [lab.split()[0] for lab in ax.get_legend_handles_labels()[1]]
        assert order == ["shot3", "shot2", "shot1"]

    def test_residual_axis_needs_a_model(self, table):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _, (a1, a2) = plt.subplots(2, 1)
        with pytest.raises(ValueError, match="needs a model"):
            plot_traveltimes(table, ax=a1, residual_ax=a2)

    def test_residuals_are_small_for_a_fitted_model(self, table):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        model = fit_layers(table["offset"], table["time"], n_layers=2)
        _, (a1, a2) = plt.subplots(2, 1)
        plot_traveltimes(table, ax=a1, model=model, residual_ax=a2)
        assert "RMS" in a2.get_title()
        drawn = np.concatenate([line.get_ydata() for line in a2.lines[1:]])
        assert np.abs(drawn).max() < 5.0  # milliseconds

    def test_unknown_axis_column(self, table):
        with pytest.raises(ValueError, match="no 'elevation' column"):
            plot_traveltimes(table, x="elevation")
