"""Refraction tomography: forward accuracy, inversion, QC helpers, field data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shallowgeo.core import Geometry, SeismicSurvey, local_grid
from shallowgeo.refraction import (
    PickingSession,
    load_shots,
    refine_picks,
    traveltimes,
)
from shallowgeo.refraction.tomography import (
    TomographyGrid,
    estimate_gradient,
    forward,
    gradient_velocity,
    lambda_sweep,
    make_grid,
    reciprocity,
    synthetic_traveltimes,
    traveltime_tomography,
)

EXAMPLE = (Path(__file__).resolve().parents[1]
           / "examples/data/2026-09-25-srt-line/raw/3000.sgy")


def _two_layer_grid(depth_to_top=4.1, dx=0.25):
    g = make_grid(x=(0.0, 30.0), depth=10.0, dx=dx)
    _, Z = g.mesh()
    return g, np.where(Z < depth_to_top, 800.0, 2500.0)


class TestGrid:
    def test_make_grid_puts_every_geophone_on_a_node(self):
        rx = np.arange(24) * 0.9144
        tab = pd.DataFrame({"source_x": np.repeat([-2.4384, 10.0584], 24),
                            "receiver_x": np.tile(rx, 2), "time": 0.01})
        g = make_grid(tab)
        _, snap = g.surface_node(rx)
        assert snap.max() < 1e-9
        assert g.x[0] < -2.4384 and g.x[-1] > rx[-1]
        assert g.z[-1] == pytest.approx(np.abs(tab.receiver_x - tab.source_x).max() / 3,
                                        abs=g.dx)

    def test_rejects_non_square_cells(self):
        with pytest.raises(ValueError, match="square"):
            TomographyGrid(np.arange(5.0), np.arange(3) * 0.5)

    def test_topography_removes_air_nodes(self):
        g = make_grid(x=(0, 10), depth=4, dx=0.5,
                      elevation=([0, 10], [100.0, 98.0]))
        assert not g.active[0, -1] and g.active[0, 0]
        assert g.surface[-1] == pytest.approx(2.0)


class TestForward:
    def test_homogeneous_times_are_exact(self):
        g = make_grid(x=(0, 30), depth=10, dx=0.25)
        rx = np.array([1.0, 7.5, 18.0, 30.0])
        out = forward(g, np.full(g.shape, 1000.0), np.zeros(4), rx)
        np.testing.assert_allclose(out["time"], rx / 1000.0, rtol=1e-9)

    def test_jacobian_reproduces_times(self):
        g, v = _two_layer_grid()
        out = forward(g, v, np.zeros(10), np.linspace(2, 30, 10))
        np.testing.assert_allclose(out["J"] @ (1 / v).ravel(), out["time"], rtol=1e-12)

    def test_head_wave_matches_the_layered_formula(self):
        g, v = _two_layer_grid(depth_to_top=4.0)  # on a node row
        x = np.linspace(1, 30, 30)
        t = forward(g, v, np.zeros_like(x), x, jacobian=False)["time"]
        # Slowness is interpolated between nodes, so the step from the last
        # slow row to the first fast one acts as an interface half a cell up.
        t_true = traveltimes(x, [800.0, 2500.0], [4.0 - g.dx / 2])
        np.testing.assert_allclose(t, t_true, rtol=0.01)

    def test_rays_start_and_end_at_the_sensors(self):
        g, v = _two_layer_grid()
        out = forward(g, v, [0.0], [25.0], rays=True)
        ray = out["rays"][0]
        assert ray[0, 0] == 0.0 and ray[-1, 0] == 25.0
        assert ray[:, 1].max() > 3.9  # refracted along the interface


@pytest.fixture(scope="module")
def synthetic():
    ft = 0.3048
    rx = np.arange(24) * 3 * ft
    sx = np.r_[-8, np.arange(3, 70, 6), 78] * ft
    g = make_grid(x=(-3.5, 25.0), depth=8.0, dx=1.5 * ft)
    X, Z = g.mesh()
    v = np.where(Z < 3.0 + 0.1 * X, 500 + 60 * Z, 2200.0)
    tab = synthetic_traveltimes(g, v, sx, rx, noise=3e-4, seed=3)
    return g, v, tab


class TestInversion:
    def test_recovers_a_dipping_refractor(self, synthetic):
        g, _, tab = synthetic
        res = traveltime_tomography(tab, grid=g, lam=10, error=3e-4, rel_error=0.0)
        assert res.chi2 < 1.5
        assert res.rms < 5e-4
        # The 1300 m/s contour should dip with the true interface: shallower
        # on the left than on the right.
        cov = res.covered()
        depth_1300 = []
        for xq in (3.0, 18.0):
            i = int(np.argmin(np.abs(g.x - xq)))
            col = np.where(cov[:, i], res.velocity[:, i], np.nan)
            depth_1300.append(g.z[np.nanargmax(col > 1300)])
        assert depth_1300[0] < depth_1300[1]
        assert depth_1300[0] == pytest.approx(3.3, abs=1.0)
        assert depth_1300[1] == pytest.approx(4.8, abs=1.0)

    def test_misfit_falls_and_history_is_recorded(self, synthetic):
        g, _, tab = synthetic
        res = traveltime_tomography(tab, grid=g)
        assert res.history["chi2"].iloc[-1] < res.history["chi2"].iloc[0] / 5
        assert {"predicted", "residual", "sigma"} <= set(res.table.columns)
        assert res.coverage.shape == g.shape and res.coverage.max() > 0
        assert np.isfinite(res.depth_of_investigation()).any()

    def test_smoothing_trades_fit_for_roughness(self, synthetic):
        g, _, tab = synthetic
        sweep = lambda_sweep(tab, lams=(3, 300), grid=g)
        assert sweep["roughness"].iloc[0] > sweep["roughness"].iloc[1]
        assert sweep["chi2"].iloc[0] < sweep["chi2"].iloc[1]

    def test_too_few_shots_is_refused_with_advice(self):
        tab = pd.DataFrame({"shot": ["a"] * 3 + ["b"] * 3, "source_x": [0.0] * 3 + [10.0] * 3,
                            "receiver_x": [2.0, 4.0, 6.0] * 2, "time": 0.01})
        with pytest.raises(ValueError, match="fit_layers"):
            traveltime_tomography(tab)

    def test_unused_and_zero_offset_rows_are_dropped(self, synthetic):
        g, _, tab = synthetic
        tab = tab.copy()
        tab["use"] = True
        tab.loc[:9, "use"] = False
        extra = tab.iloc[[0]].assign(receiver_x=tab.source_x.iloc[0], use=True)
        res = traveltime_tomography(pd.concat([tab, extra]), grid=g, max_iter=1)
        assert len(res.table) == len(tab) - 10

    def test_starting_gradient_is_in_range(self, synthetic):
        g, _, tab = synthetic
        v_top, v_bottom = estimate_gradient(tab)
        assert 400 < v_top < 800 and v_bottom > v_top
        v0 = gradient_velocity(g, v_top, v_bottom)
        assert v0[0, 0] == pytest.approx(v_top) and v0[-1, 0] == pytest.approx(v_bottom)

    def test_outliers_flag_a_planted_mispick(self, synthetic):
        g, _, tab = synthetic
        tab = tab.copy()
        tab.loc[40, "time"] += 0.012
        res = traveltime_tomography(tab, grid=g)
        assert 40 in res.outliers().index


class TestReciprocity:
    def test_finds_pairs_and_measures_the_difference(self):
        tab = pd.DataFrame({
            "shot": ["A", "A", "B", "B"],
            "source_x": [0.0, 0.0, 6.0, 6.0],
            "receiver_x": [6.0, 3.0, 0.0, 3.0],
            "time": [0.0100, 0.006, 0.0104, 0.005],
        })
        pairs = reciprocity(tab)
        assert len(pairs) == 1
        assert pairs["dt"].iloc[0] == pytest.approx(-0.0004)
        assert pairs["offset"].iloc[0] == pytest.approx(6.0)


def _line_survey(src_x, v=(600.0, 2400.0), h=(3.0,), seed=0, late=None):
    """One shot into a 24-geophone spread; ``late`` puts a stronger fake
    arrival at that time on every trace, to lure the picker."""
    rng = np.random.default_rng(seed)
    dt, n = 0.000125, 800
    xs = np.arange(24) * 1.0
    tt = traveltimes(np.abs(xs - src_x), list(v), list(h))
    t = np.arange(n) * dt

    def wavelet(t0, amp):
        s = t - t0
        return amp * np.where(s >= 0, np.sin(2 * np.pi * 120 * s) * np.exp(-s / 0.004), 0.0)

    data = np.array([wavelet(t0, 0.3) for t0 in tt])
    if late is not None:
        data[12:] += np.array([wavelet(late, 3.0) for _ in range(12)])
    data += rng.normal(0, 0.005, data.shape)
    geom = Geometry(ids=[*range(1, 25), "S1"], x=[*xs, src_x], y=[0.0] * 25, z=[0.0] * 25,
                    roles=[*["receiver"] * 24, "source"], spatial_ref=local_grid(0, 0))
    tmap = pd.DataFrame({"receiver_id": range(1, 25), "source_id": "S1",
                         "channel": range(1, 25)})
    return SeismicSurvey(data, dt, geom, tmap)


class TestGuidedPicking:
    def test_guide_rescues_traces_locked_on_a_later_arrival(self):
        survey = _line_survey(-1.0, late=0.045)
        session = PickingSession({"s": survey}, min_time=0.001, max_time=0.06,
                                 min_quality=0)
        true = traveltimes(np.abs(np.arange(24) - -1.0), [600.0, 2400.0], [3.0])
        before = session.picks["s"]["time"].to_numpy()
        assert np.abs(before[12:] - true[12:]).max() > 0.02  # fooled

        session.guided_pick(lambda sx, rx: traveltimes(np.abs(rx - sx), [600.0, 2400.0],
                                                       [3.0]) + 0.001)
        after = session.picks["s"]["time"].to_numpy()
        np.testing.assert_allclose(after[1:], true[1:], atol=4e-4)

    def test_hand_edits_survive(self):
        session = PickingSession({"s": _line_survey(-1.0)}, min_time=0.001)
        session.set_pick("s", 5, 0.0123)
        session.guided_pick(lambda sx, rx: np.full(rx.size, 0.01))
        row = session.picks["s"].set_index("channel").loc[5]
        assert row["time"] == pytest.approx(0.0123, abs=2e-4) and row["edited"]

    def test_exclude_takes_outlier_rows(self):
        session = PickingSession({"s": _line_survey(-1.0)}, min_time=0.001)
        session.exclude(pd.DataFrame({"shot": ["s", "s"], "channel": [3, 7]}))
        used = session.picks["s"].set_index("channel")["use"]
        assert not used[3] and not used[7] and used[4]


@pytest.fixture(scope="module")
def example_line():
    if not EXAMPLE.exists():
        pytest.skip("example dataset not present")
    session = PickingSession(load_shots([EXAMPLE]), min_time=0.001, max_time=0.075)
    return session, refine_picks(session)


class TestExampleLine:
    """The committed 2026-09-25 line, end to end. Numbers are in dataset.md."""

    def test_sixteen_shots_from_one_file(self, example_line):
        session, _ = example_line
        assert len(session.labels) == 16

    def test_fits_to_the_pick_error(self, example_line):
        session, res = example_line
        assert len(res.table) > 270
        assert res.chi2 < 1.2
        assert res.rms < 0.001
        pairs = reciprocity(session.table())
        assert len(pairs) > 40
        assert pairs["dt"].abs().median() < 0.001

    def test_slow_cover_over_fast_ground(self, example_line):
        _, res = example_line
        prof = res.profile(10.0)
        near = prof.loc[prof["depth"] < 0.6, "velocity"].mean()
        deep = prof.loc[(prof["depth"] > 4.5) & (prof["depth"] < 6.0), "velocity"].mean()
        assert 250 < near < 700
        assert deep > 3000


class TestPyGIMLiBackend:
    def test_agrees_with_native(self):
        pytest.importorskip("pygimli")
        ft = 0.3048
        g = make_grid(x=(-3.5, 25.0), depth=8.0, dx=1.5 * ft)
        X, Z = g.mesh()
        v = np.where(Z < 3.0 + 0.1 * X, 500 + 60 * Z, 2200.0)
        tab = synthetic_traveltimes(g, v, np.r_[-8, np.arange(3, 70, 6), 78] * ft,
                                    np.arange(24) * 3 * ft, noise=3e-4)
        nat = traveltime_tomography(tab, grid=g)
        pgr = traveltime_tomography(tab, grid=g, backend="pygimli")
        both = nat.covered() & pgr.covered()
        diff = np.abs(nat.velocity - pgr.velocity)[both] / pgr.velocity[both]
        assert np.median(diff) < 0.15
        assert pgr.chi2 < 2
