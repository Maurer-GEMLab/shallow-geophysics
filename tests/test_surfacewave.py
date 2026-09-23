import numpy as np
import pandas as pd
import pytest

from shallowgeo.core import Geometry, SeismicSurvey, local_grid
from shallowgeo.surfacewave import (
    DispersionCurve,
    LayeredModel,
    dispersion_image,
    phase_shift,
    spread_wavelength_limits,
)

def _c_true(f):
    """A normally dispersive curve: fast at low frequency, slow at high."""
    return 150.0 + 250.0 * np.exp(-f / 20.0)


def _dispersive_gather(dt=0.001, n=2000, nfft=4096, spacing=1.0, n_traces=24, near=5.0):
    x = np.arange(n_traces) * spacing + near
    f = np.fft.rfftfreq(nfft, dt)
    source = np.exp(-((f - 25.0) / 15.0) ** 2)
    safe_c = np.where(f > 0, _c_true(f), 1.0)
    data = np.zeros((x.size, n))
    for i, xi in enumerate(x):
        phase = np.exp(-2j * np.pi * f * xi / safe_c)
        data[i] = np.fft.irfft(source * phase, nfft)[:n]
    return data, x, dt


class TestDispersionCurve:
    def test_sorts_by_frequency_and_derives_wavelength(self):
        c = DispersionCurve([30.0, 10.0, 20.0], [200.0, 400.0, 300.0])
        np.testing.assert_allclose(c.frequency, [10, 20, 30])
        np.testing.assert_allclose(c.wavelength, [40, 15, 200 / 30])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            DispersionCurve([1.0, 2.0], [100.0])

    def test_clip_wavelength_keeps_resolvable_points(self):
        c = DispersionCurve([5.0, 20.0, 80.0], [400.0, 300.0, 200.0])
        clipped = c.clip_wavelength(3.0, 40.0)
        assert clipped.n_points == 1 and clipped.frequency[0] == 20.0
        assert clipped.metadata["clipped_to"] == (3.0, 40.0)

    def test_dataframe_round_trip(self):
        c = DispersionCurve([10.0, 20.0], [300.0, 250.0], [10.0, 12.0])
        back = DispersionCurve.from_dataframe(c.to_dataframe())
        np.testing.assert_allclose(back.velocity_std, [10.0, 12.0])

    def test_resample_stays_inside_range(self):
        c = DispersionCurve([10.0, 20.0, 30.0], [300.0, 250.0, 200.0])
        r = c.resample([5.0, 15.0, 25.0, 35.0])
        np.testing.assert_allclose(r.frequency, [15.0, 25.0])
        np.testing.assert_allclose(r.velocity, [275.0, 225.0])

    def test_spread_limits(self):
        lam_min, lam_max = spread_wavelength_limits(np.arange(24) * 2.0 + 5.0)
        assert lam_min == 4.0 and lam_max == 46.0


class TestPhaseShift:
    def test_recovers_known_dispersion(self):
        data, x, dt = _dispersive_gather()
        img = phase_shift(data, x, dt, fmin=5, fmax=60, vmin=50, vmax=600, dv=2.0)
        assert img.image.shape == (img.frequency.size, img.velocity.size)
        assert img.image.max() == pytest.approx(1.0)  # normalised rows
        # Wavelengths longer than the 23 m spread are poorly resolved (they
        # miss by ~10 m/s here); inside the resolvable band the pick is exact
        # to the velocity step.
        curve = img.pick(fmin=8, fmax=50, method="track")
        resolvable = curve.clip_wavelength(*spread_wavelength_limits(x))
        assert resolvable.n_points > 100
        err = resolvable.velocity - _c_true(resolvable.frequency)
        assert np.abs(err).max() < 2 * 2.0 + 1.0
        assert np.abs(curve.velocity - _c_true(curve.frequency)).max() < 15.0

    def test_max_method_also_works_on_clean_data(self):
        data, x, dt = _dispersive_gather()
        img = phase_shift(data, x, dt, fmin=5, fmax=60, vmin=50, vmax=600, dv=2.0)
        curve = img.pick(fmin=8, fmax=50, method="max")
        assert np.abs(curve.velocity - _c_true(curve.frequency)).max() < 10.0

    def test_pick_reports_peak_width_as_std(self):
        data, x, dt = _dispersive_gather()
        img = phase_shift(data, x, dt, fmin=5, fmax=60, vmin=50, vmax=600, dv=2.0)
        curve = img.pick(fmin=8, fmax=50)
        assert curve.velocity_std is not None and (curve.velocity_std > 0).all()
        assert "quality" in curve.metadata

    def test_empty_pick_window_raises(self):
        data, x, dt = _dispersive_gather()
        img = phase_shift(data, x, dt, fmin=5, fmax=60)
        with pytest.raises(ValueError, match="no image samples"):
            img.pick(fmin=100, fmax=200)

    def test_offsets_must_match_traces(self):
        data, x, dt = _dispersive_gather()
        with pytest.raises(ValueError, match="one entry per trace"):
            phase_shift(data, x[:-1], dt)

    def test_equal_offsets_rejected(self):
        with pytest.raises(ValueError, match="no geometry"):
            phase_shift(np.zeros((4, 100)), np.zeros(4), 0.001)

    def test_dispersion_image_from_survey(self):
        data, x, dt = _dispersive_gather(n_traces=8)
        geom = Geometry(
            ids=[*range(1, 9), "S1"], x=[*x, 0.0], y=[0.0] * 9, z=[0.0] * 9,
            roles=[*["receiver"] * 8, "source"], spatial_ref=local_grid(0, 0),
        )
        tmap = pd.DataFrame({"receiver_id": range(1, 9), "source_id": "S1"})
        survey = SeismicSurvey(data, dt, geom, tmap)
        img = dispersion_image(survey, fmin=5, fmax=60)
        assert img.metadata["wavelength_limits"] == (2.0, 7.0)
        assert img.metadata["frequency_floor"] == pytest.approx(1.0)  # 2 s record

    def test_passive_survey_rejected(self):
        geom = Geometry([1], [0], [0], [0], ["receiver"], local_grid(0, 0))
        survey = SeismicSurvey(np.zeros((1, 100)), 0.004, geom,
                               pd.DataFrame({"receiver_id": [1]}))
        with pytest.raises(ValueError, match="active-source"):
            dispersion_image(survey)


class TestLayeredModel:
    def test_defaults_fill_vp_and_density(self):
        m = LayeredModel([3.0, 6.0], [180.0, 320.0, 600.0])
        assert m.n_layers == 3
        np.testing.assert_allclose(m.vp, m.vs * np.sqrt(3))
        assert m.density.size == 3
        np.testing.assert_allclose(m.depth_top, [0, 3, 9])
        assert np.isinf(m.depth_bottom[-1])

    def test_halfspace_thickness_entry_is_tolerated(self):
        m = LayeredModel([3.0, 6.0, 0.0], [180.0, 320.0, 600.0])
        assert m.thickness.size == 2

    def test_wrong_thickness_count_raises(self):
        with pytest.raises(ValueError, match="expected 2"):
            LayeredModel([3.0], [180.0, 320.0, 600.0])

    def test_to_disba_units(self):
        m = LayeredModel([3.0], [200.0, 400.0])
        arr = m.to_disba()
        assert arr.shape == (2, 4)
        assert arr[0, 0] == pytest.approx(0.003)   # km
        assert arr[1, 2] == pytest.approx(0.4)     # km/s
        assert arr[0, 3] == pytest.approx(1.9)     # g/cm3

    def test_profile_is_a_step_function(self):
        z, v = LayeredModel([3.0], [200.0, 400.0]).profile(zmax=10.0)
        assert z.tolist() == [0.0, 3.0, 3.0, 10.0]
        assert v.tolist() == [200.0, 200.0, 400.0, 400.0]


disba_available = True
try:
    import disba  # noqa: F401
except ImportError:
    disba_available = False

needs_disba = pytest.mark.skipif(not disba_available, reason="masw extra (disba) not installed")


@needs_disba
class TestForwardAndInversion:
    truth = LayeredModel([3.0, 6.0], [180.0, 320.0, 600.0])
    freqs = np.linspace(5.0, 50.0, 30)

    def test_forward_is_normally_dispersive(self):
        from shallowgeo.surfacewave import forward_dispersion

        curve = forward_dispersion(self.truth, self.freqs)
        assert curve.n_points == 30
        assert np.all(np.diff(curve.velocity) <= 1e-6)  # slower at higher frequency
        # Bracketed by the Vs of the top layer and the half-space (Rayleigh ~0.9 Vs).
        assert 0.85 * 180 < curve.velocity.min() < 180
        assert curve.velocity.max() < 600

    def test_initial_model_spans_the_data_depths(self):
        from shallowgeo.surfacewave import forward_dispersion, initial_model

        curve = forward_dispersion(self.truth, self.freqs)
        init = initial_model(curve, 5)
        assert init.n_layers == 5
        assert init.thickness.size == 4
        # deepest interface at about a third of the longest wavelength
        assert init.depth_top[-1] == pytest.approx(curve.wavelength.max() / 3, rel=0.05)

    def test_inversion_with_thickness_recovers_truth(self):
        from shallowgeo.surfacewave import forward_dispersion, invert_dispersion

        obs = forward_dispersion(self.truth, self.freqs)
        start = LayeredModel([2.0, 8.0], [220.0, 280.0, 500.0])
        res = invert_dispersion(obs, start, invert_thickness=True, max_evaluations=300)
        np.testing.assert_allclose(res.model.vs, self.truth.vs, rtol=0.03)
        np.testing.assert_allclose(res.model.thickness, self.truth.thickness, rtol=0.1)
        assert res.rms < 2.0
        assert res.initial is start

    def test_fixed_thickness_many_layers_fits_the_data(self):
        from shallowgeo.surfacewave import forward_dispersion, invert_dispersion

        obs = forward_dispersion(self.truth, self.freqs)
        res = invert_dispersion(obs, 6, smoothing=0.3)
        assert res.rms < 10.0
        assert res.model.n_layers == 6
        # Half-space-ish velocity at depth, soil-ish at the surface.
        assert res.model.vs[0] < 250 and res.model.vs[-1] > 400

    def test_result_residuals_and_plot(self):
        from shallowgeo.surfacewave import forward_dispersion, invert_dispersion

        obs = forward_dispersion(self.truth, self.freqs)
        res = invert_dispersion(obs, LayeredModel([3.0, 6.0], [200.0, 300.0, 550.0]))
        assert res.residuals().size == obs.n_points
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        axes = res.plot()
        assert len(axes) == 2
