"""Passive surface-wave processing.

The diagnostics get the most attention here, because they are the part that
has to be *right when the data are wrong*. A dispersion routine that returns a
curve from noise is not a minor inconvenience: the curve it returns looks
entirely ordinary, and the error only shows up as a Vs profile that quietly
disagrees with the ground.

So each diagnostic is tested twice -- once against a synthetic wavefield where
it must pass, and once against a failure mode where it must fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shallowgeo import passive as P
from shallowgeo.core.crs import SpatialRef
from shallowgeo.core.geometry import Geometry
from shallowgeo.core.survey import SeismicSurvey
from shallowgeo.surfacewave import DispersionCurve

FS = 100.0
WGS84 = SpatialRef("EPSG:4326", vertical_datum="ellipsoidal")
METRE_PER_DEGREE = 111320.0


def make_survey(offsets_m, data, *, start="2026-09-18 17:00:00"):
    """A passive survey with stations on a north-south line."""
    ids = [f"n{i}" for i in range(len(offsets_m))]
    geometry = Geometry(
        ids=ids,
        x=[-91.7764] * len(ids),
        y=[37.925 + d / METRE_PER_DEGREE for d in offsets_m],
        z=[330.0] * len(ids),
        roles=["receiver"] * len(ids),
        spatial_ref=WGS84,
    )
    return SeismicSurvey(
        data=np.asarray(data, dtype=float),
        sample_interval=1.0 / FS,
        geometry=geometry,
        trace_map=pd.DataFrame({"receiver_id": ids, "channel": "Z"}),
        start_time=pd.Timestamp(start, tz="UTC"),
    )


def propagating_noise(offsets_m, *, seconds=400.0, velocity=250.0,
                      n_sources=300, seed=0, local_noise=0.0):
    r"""Non-dispersive plane waves crossing the array from every azimuth.

    The azimuths are drawn uniformly, which is precisely the condition SPAC
    assumes: averaging :math:`e^{i k r \\cos\\theta}` over a full circle is
    what produces the :math:`J_0(kr)` that the method inverts. A handful of
    discrete azimuths does not converge to a Bessel function, and a single
    azimuth does not either -- that case is ``broadside_noise``.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    out = np.zeros((len(offsets_m), n))
    positions = np.asarray(offsets_m, dtype=float)
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    band = (freqs >= 2.0) & (freqs <= 20.0)
    for _ in range(n_sources):
        azimuth = rng.uniform(0.0, 2.0 * np.pi)
        # The array runs north-south, so only that component projects onto it.
        slowness = np.cos(azimuth) / velocity
        spectrum = np.fft.rfft(rng.normal(size=n))
        spectrum[~band] = 0.0
        for i, x in enumerate(positions):
            shifted = spectrum * np.exp(-2j * np.pi * freqs * slowness * x)
            out[i] += np.fft.irfft(shifted, n=n)
    if local_noise:
        out += local_noise * rng.normal(size=out.shape) * out.std()
    return out


def broadside_noise(offsets_m, *, seconds=400.0, seed=1):
    """One waveform, identical at every station: the common-mode failure.

    This is what a wavefield arriving perpendicular to the array looks like,
    and also what shared instrument noise looks like. The coherency is high
    and flat, and there is no velocity information in it at all.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    shared = rng.normal(size=n)
    spectrum = np.fft.rfft(shared)
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    spectrum[(freqs < 2.0) | (freqs > 20.0)] = 0.0
    shared = np.fft.irfft(spectrum, n=n)
    return np.tile(shared, (len(offsets_m), 1)) + 0.05 * rng.normal(
        size=(len(offsets_m), n)
    )


# -- geometry and limits ----------------------------------------------------


def test_array_layout_recognises_a_line():
    survey = make_survey([0, 20, 40, 60, 80], np.zeros((5, 100)))
    layout = P.array_layout(survey)
    assert layout.is_linear
    assert layout.aperture == pytest.approx(80.0, abs=1.0)
    assert len(layout.separations) == 10
    assert layout.separations["distance"].min() == pytest.approx(20.0, abs=0.5)


def test_resolution_limits_bracket_the_usable_band():
    survey = make_survey([0, 20, 40, 60, 80], np.zeros((5, 100)))
    limits = P.resolution_limits(P.array_layout(survey))
    assert limits.lambda_min == pytest.approx(40.0, abs=1.0)   # 2 * spacing
    assert limits.lambda_max == pytest.approx(160.0, abs=2.0)  # 2 * aperture
    lo, hi = limits.frequency_band(200.0)
    assert lo < hi
    assert hi == pytest.approx(200.0 / limits.lambda_min, rel=1e-6)


# -- coverage, windows, trimming --------------------------------------------


def test_coverage_and_common_window_ignore_nan():
    data = np.ones((3, 1000))
    data[1, :400] = np.nan          # node 1 starts late
    data[2, 900:] = np.nan          # node 2 stops early
    survey = make_survey([0, 20, 40], data)

    coverage = P.node_coverage(survey).set_index("node")
    assert coverage.loc["n0", "coverage"] == pytest.approx(1.0)
    assert coverage.loc["n1", "coverage"] == pytest.approx(0.6)

    start, end = P.common_window(survey)
    assert (start - survey.start_time).total_seconds() == pytest.approx(4.0)
    assert (end - survey.start_time).total_seconds() == pytest.approx(9.0)

    trimmed = P.trim(survey, start, end)
    assert not np.isnan(trimmed.data).any()


def test_common_window_takes_the_longest_unbroken_run():
    """A node dropping out mid-record must not yield a window full of holes."""
    data = np.ones((2, 1000))
    data[1, 100:150] = np.nan
    survey = make_survey([0, 20], data)
    start, end = P.common_window(survey)
    assert (end - start).total_seconds() == pytest.approx(8.5)
    assert not np.isnan(P.trim(survey, start, end).data).any()


def test_common_window_raises_when_nodes_never_overlap():
    data = np.full((2, 1000), np.nan)
    data[0, :500] = 1.0
    data[1, 500:] = 1.0
    with pytest.raises(ValueError, match="never recorded simultaneously"):
        P.common_window(make_survey([0, 20], data))


def test_trim_keeps_geometry_consistent():
    survey = make_survey([0, 20, 40], np.ones((3, 1000)))
    trimmed = P.trim(survey, nodes=["n0", "n2"])
    assert trimmed.n_traces == 2
    assert len(trimmed.geometry) == 2
    assert set(trimmed.geometry.table["id"]) == {"n0", "n2"}


# -- SPAC on a wavefield that satisfies its assumptions ---------------------


# A compact array, so that every pair stays inside the first lobe of J0 over
# the band tested. Past the first zero the inversion is not single-valued and
# SPAC legitimately declines to answer -- that is tested separately.
SPAC_OFFSETS = [0, 4, 8, 14, 20]


def test_spac_recovers_a_known_velocity():
    survey = make_survey(SPAC_OFFSETS,
                         propagating_noise(SPAC_OFFSETS, velocity=250.0))
    result = P.spac_coherency(survey, window=20.0, fmin=2.0, fmax=6.0)
    test = P.separation_test(result)
    assert test.usable.mean() > 0.3, test.verdict

    curve = P.spac_dispersion(result, test=test)
    assert np.median(curve.velocity) == pytest.approx(250.0, rel=0.25)


def test_coherency_falls_with_separation_for_a_real_wavefield():
    survey = make_survey(SPAC_OFFSETS,
                         propagating_noise(SPAC_OFFSETS, velocity=250.0))
    result = P.spac_coherency(survey, window=20.0, fmin=2.0, fmax=6.0)
    row = result.at(3.0).sort_values("distance")
    assert row["coherency"].iloc[0] > row["coherency"].iloc[-1]
    # ...and it should track the Bessel function it is supposed to be.
    from scipy.special import j0
    expected = j0(2 * np.pi * 3.0 * row["distance"].to_numpy() / 250.0)
    assert np.corrcoef(row["coherency"], expected)[0, 1] > 0.9


# -- the diagnostics must reject the failure modes --------------------------


def test_separation_test_rejects_common_mode_noise():
    """Identical signal at every node: high coherency, no wave."""
    offsets = SPAC_OFFSETS
    survey = make_survey(offsets, broadside_noise(offsets))
    result = P.spac_coherency(survey, window=20.0, fmin=2.0, fmax=6.0)

    row = result.at(3.0)
    assert row["coherency"].min() > 0.8, "fixture should be highly coherent"
    assert row["coherency"].std() < 0.1, "...and flat across separation"

    test = P.separation_test(result)
    assert test.usable.mean() < 0.15
    assert "fail" in test.verdict
    with pytest.raises(ValueError, match="no frequency passed"):
        P.spac_dispersion(result, test=test)


def test_wavelength_test_accepts_a_real_dispersion_curve():
    frequency = np.linspace(4.0, 25.0, 40)
    velocity = 120.0 + 900.0 / frequency        # normal dispersion
    check = P.wavelength_test(DispersionCurve(frequency, velocity))
    assert check.plausible
    assert check.index > 1.0


def test_wavelength_test_accepts_a_uniform_half_space():
    """Constant velocity is not dispersion, but it is a real measurement."""
    frequency = np.linspace(4.0, 25.0, 40)
    check = P.wavelength_test(DispersionCurve(frequency, np.full(40, 250.0)))
    assert check.plausible
    assert check.index == pytest.approx(1.0, abs=0.05)


def test_wavelength_test_rejects_a_pinned_wavelength():
    """``c`` proportional to ``f`` is what a flat coherency inverts to."""
    frequency = np.linspace(2.0, 30.0, 40)
    velocity = 20.0 * frequency                 # lambda fixed at 20 m
    check = P.wavelength_test(DispersionCurve(frequency, velocity))
    assert not check.plausible
    assert check.index == pytest.approx(0.0, abs=0.05)
    assert "pinned" in check.verdict


def test_wavelength_test_refuses_a_narrow_band():
    frequency = np.linspace(4.0, 4.5, 20)
    with pytest.raises(ValueError, match="only a factor"):
        P.wavelength_test(DispersionCurve(frequency, np.full(20, 250.0)))


# -- event detection and passive MASW ---------------------------------------


def test_detect_events_requires_every_node():
    rng = np.random.default_rng(5)
    n = int(120 * FS)
    data = rng.normal(size=(4, n))
    fs = int(FS)
    # A burst on one node only: must NOT be detected.
    data[1, 10 * fs:12 * fs] *= 40.0
    # A burst on all four: must be detected.
    data[:, 60 * fs:62 * fs] *= 40.0

    survey = make_survey([0, 20, 40, 60], data)
    events = P.detect_events(survey, window=2.0, threshold=3.0, band=None)
    times = {int((e.start_time - survey.start_time).total_seconds()) for e in events}
    assert 60 in times
    assert not ({10} & times), "a single loud node is not an event"


def test_event_dispersion_image_finds_a_planted_velocity():
    offsets = [0, 8, 16, 24, 32, 40]
    rng = np.random.default_rng(6)
    n = int(120 * FS)
    velocity = 200.0
    data = 0.02 * rng.normal(size=(len(offsets), n))
    fs = int(FS)
    for onset in (20, 50, 80):
        source = rng.normal(size=4 * fs) * np.hanning(4 * fs)
        spectrum = np.fft.rfft(source)
        freqs = np.fft.rfftfreq(4 * fs, 1.0 / FS)
        spectrum[(freqs < 4.0) | (freqs > 25.0)] = 0.0
        for i, x in enumerate(offsets):
            delayed = np.fft.irfft(
                spectrum * np.exp(-2j * np.pi * freqs * x / velocity), n=4 * fs
            )
            data[i, onset * fs:onset * fs + 4 * fs] += delayed

    survey = make_survey(offsets, data)
    events = P.detect_events(survey, window=4.0, threshold=2.0)
    assert events

    image = P.event_dispersion_image(survey, events, fmin=5, fmax=20,
                                     vmin=80, vmax=500, dv=2)
    peaks = [image.velocity[np.argmax(image.image[k])]
             for k in range(image.frequency.size)]
    assert np.median(peaks) == pytest.approx(velocity, rel=0.2)


def test_event_image_refuses_a_2d_array():
    ids = [f"n{i}" for i in range(4)]
    geometry = Geometry(
        ids=ids, x=[-91.7764, -91.7761, -91.7764, -91.7761],
        y=[37.925, 37.925, 37.9253, 37.9253], z=[330.0] * 4,
        roles=["receiver"] * 4, spatial_ref=WGS84,
    )
    survey = SeismicSurvey(
        data=np.random.default_rng(7).normal(size=(4, int(60 * FS))),
        sample_interval=1.0 / FS, geometry=geometry,
        trace_map=pd.DataFrame({"receiver_id": ids, "channel": "Z"}),
        start_time=pd.Timestamp("2026-09-18 17:00:00", tz="UTC"),
    )
    events = P.detect_events(survey, window=4.0, threshold=0.0, max_events=3)
    with pytest.raises(ValueError, match="line array"):
        P.event_dispersion_image(survey, events)


def test_anti_trigger_protects_the_coherency():
    """One node's local transient must not be allowed to erase the signal.

    Without the amplitude gate a handful of very loud windows dominate both
    the cross-spectrum and the auto-spectra, and the measured coherency
    collapses even though the ambient field is unchanged. This is not a
    theoretical worry: on the September 2026 ATOM-1C deployment it took the
    measured coherency from 0.53 to 0.03.
    """
    clean = propagating_noise(SPAC_OFFSETS, velocity=250.0, seconds=300.0)
    contaminated = clean.copy()
    fs = int(FS)
    rng = np.random.default_rng(8)
    for onset in (40, 90, 140, 190):
        contaminated[1, onset * fs:(onset + 5) * fs] += (
            200.0 * clean.std() * rng.normal(size=5 * fs)
        )
    survey = make_survey(SPAC_OFFSETS, contaminated)

    gated = P.spac_coherency(survey, window=20.0, fmin=2.0, fmax=4.0)
    ungated = P.spac_coherency(survey, window=20.0, fmin=2.0, fmax=4.0,
                               anti_trigger=None)
    assert gated.n_windows_rejected > 0

    # Pair 0 involves the contaminated node, so it is the one at risk.
    assert gated.pairs.loc[0, "node_a"] == "n0"
    assert gated.pairs.loc[0, "node_b"] == "n1"
    assert gated.coherency[0].mean() > 0.7
    assert ungated.coherency[0].mean() < 0.3 * gated.coherency[0].mean()


# -- nodes used as an active-source receiver array ---------------------------


def _shot_record(offsets_m, shot_seconds, *, velocity=220.0, seconds=200.0, seed=9):
    """Quiet background with surface-wave trains at known times."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    data = 0.01 * rng.normal(size=(len(offsets_m), n))
    length = int(2 * FS)
    freqs = np.fft.rfftfreq(length, 1.0 / FS)
    for onset in shot_seconds:
        spectrum = np.fft.rfft(rng.normal(size=length) * np.hanning(length))
        spectrum[(freqs < 4.0) | (freqs > 20.0)] = 0.0
        for i, x in enumerate(offsets_m):
            delayed = np.fft.irfft(
                spectrum * np.exp(-2j * np.pi * freqs * x / velocity), n=length
            )
            start = int(onset * FS)
            data[i, start:start + length] += delayed
    return data


def test_shot_windows_finds_known_shots():
    offsets = [0, 10, 20, 30, 40]
    shots = [30.0, 70.0, 120.0]
    survey = make_survey(offsets, _shot_record(offsets, shots))

    times = [survey.start_time + pd.Timedelta(seconds=t) for t in shots]
    windows = P.shot_windows(survey, times, window=6.0, pre=0.5, search=3.0)
    assert len(windows) == 3
    for w, t in zip(windows, shots):
        found = (w.start_time - survey.start_time).total_seconds() + 0.5
        assert found == pytest.approx(t, abs=0.5)
        assert w.score > 5


def test_shot_windows_tolerates_a_wrong_clock():
    """Seismograph headers are a second-resolution stamp from a free clock."""
    offsets = [0, 10, 20, 30, 40]
    shots = [30.0, 70.0]
    survey = make_survey(offsets, _shot_record(offsets, shots))
    # Nominal times two seconds late, as an unsynced seismograph would give.
    times = [survey.start_time + pd.Timedelta(seconds=t + 2.0) for t in shots]

    windows = P.shot_windows(survey, times, window=6.0, pre=0.5, search=3.0)
    for w, t in zip(windows, shots):
        found = (w.start_time - survey.start_time).total_seconds() + 0.5
        assert found == pytest.approx(t, abs=0.5), "refinement should recover the onset"


def test_shot_windows_rejects_a_time_zone_error():
    offsets = [0, 10, 20]
    survey = make_survey(offsets, _shot_record(offsets, [30.0]))
    # Local time passed where UTC was wanted: five hours out, so nothing lands.
    times = [survey.start_time - pd.Timedelta(hours=5)]
    with pytest.raises(ValueError, match="time zone"):
        P.shot_windows(survey, times, window=6.0)


def test_source_position_does_not_change_the_dispersion_image():
    """Offsetting every receiver equally leaves the phase-shift image alone.

    This is why nodes can be used for an active shot whose position was never
    recorded: only the receiver geometry and the propagation direction enter.
    """
    from shallowgeo.surfacewave.dispersion import phase_shift

    offsets = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    shots = [30.0]
    data = _shot_record(offsets, shots)
    block = data[:, int(29.5 * FS):int(35.5 * FS)]

    near = phase_shift(block, offsets, 1.0 / FS, fmin=4, fmax=20,
                       vmin=100, vmax=400, dv=2)
    far = phase_shift(block, offsets + 137.0, 1.0 / FS, fmin=4, fmax=20,
                      vmin=100, vmax=400, dv=2)
    np.testing.assert_allclose(near.image, far.image, atol=1e-9)

    peaks = [near.velocity[np.argmax(near.image[k])]
             for k in range(near.frequency.size)]
    assert np.median(peaks) == pytest.approx(220.0, rel=0.15)


# -- the joint Bessel fit ----------------------------------------------------


def test_separation_rings_groups_equal_distances():
    distances = np.array([10.0, 10.2, 20.0, 20.1, 19.9, 40.0])
    rings = P.separation_rings(distances, tolerance=0.05)
    assert rings[0] == rings[1]
    assert rings[2] == rings[3] == rings[4]
    assert len({rings[0], rings[2], rings[5]}) == 3


def _ring_coherency(frequency, distances, velocity, alpha):
    """Exactly what SPAC should measure for a partly coherent wavefield."""
    from scipy.special import j0
    return alpha * j0(2 * np.pi * frequency * distances / velocity)


def test_fit_is_unbiased_when_the_field_is_only_partly_coherent():
    """A free amplitude is what keeps incoherent noise out of the velocity.

    Point-wise inversion charges the whole coherency deficit to the velocity;
    the joint fit accounts for it separately and gets the velocity right.
    """
    from shallowgeo.passive.spac import _fit_bessel, _invert_bessel

    distances = np.array([4.0, 10.0, 22.0, 45.0])
    truth_v, alpha, f = 240.0, 0.45, 6.0
    rho = _ring_coherency(f, distances, truth_v, alpha)

    fitted, amplitude, residual = _fit_bessel(
        f, distances, rho, np.arange(80.0, 900.0, 1.0)
    )
    assert fitted == pytest.approx(truth_v, rel=0.05)
    assert amplitude == pytest.approx(alpha, rel=0.1)
    assert residual < 0.01

    # The same data inverted one separation at a time: each is charged the
    # missing coherence separately, so they scatter instead of agreeing.
    pointwise = [_invert_bessel(f, r_, d) for d, r_ in zip(distances, rho)]
    pointwise = np.array([v for v in pointwise if np.isfinite(v)])
    assert pointwise.size >= 3
    assert pointwise.max() / pointwise.min() > 3.0, (
        "point-wise inversion of a partly coherent field should disagree "
        "across separations, while the joint fit gives one answer"
    )


def test_fit_rejects_a_flat_coherency():
    """Common-mode correlation must not be fitted as a very fast wave.

    A constant rho is matched by any velocity large enough that J0 is flat
    across the array, so the amplitude alone explains the data. The aperture
    guard is what stops that being reported.
    """
    offsets = [0, 6, 14, 30]
    survey = make_survey(offsets, broadside_noise(offsets))
    result = P.spac_coherency(survey, window=20.0, fmin=2.0, fmax=12.0)

    row = result.at(5.0)
    assert row["coherency"].std() < 0.1, "fixture should be flat across separation"

    with pytest.raises(ValueError, match="no frequency passed"):
        P.spac_dispersion(result, min_rings=3)


def test_fit_recovers_a_dispersion_curve_the_pointwise_route_cannot():
    offsets = [0, 4, 9, 18, 34]
    survey = make_survey(
        offsets, propagating_noise(offsets, velocity=250.0, local_noise=1.2)
    )
    result = P.spac_coherency(survey, window=20.0, fmin=2.0, fmax=12.0)
    curve = P.spac_dispersion(result, only_usable=False, min_rings=3,
                              vmin=80.0, vmax=800.0)
    assert curve.n_points > 10
    assert np.median(curve.velocity) == pytest.approx(250.0, rel=0.2)
    assert 0.0 < curve.metadata["coherent_fraction"] < 1.0


def test_fit_needs_enough_distinct_separations():
    # Three evenly spaced stations give only two distinct separations, which
    # is not enough to solve for velocity and coherent fraction together.
    offsets = [0, 10, 20]
    survey = make_survey(offsets, propagating_noise(offsets, velocity=250.0))
    result = P.spac_coherency(survey, window=20.0, fmin=2.0, fmax=12.0)
    with pytest.raises(ValueError, match="at least 3 distinct station separations"):
        P.spac_dispersion(result, only_usable=False, min_rings=3)


def test_suggest_array_spans_the_requested_depths():
    table = P.suggest_array(9, depth_min=2.0, depth_max=30.0)
    assert len(table) == 9
    radii = table["radius"].to_numpy()[1:]
    # Geometric, not even: the ratio between successive radii is constant.
    ratios = radii[1:] / radii[:-1]
    assert np.allclose(ratios, ratios[0], rtol=0.02)
    assert radii.max() / radii.min() > 10

    lam_lo, lam_hi = table.attrs["lambda_range"]
    assert lam_lo == pytest.approx(6.0)
    assert lam_hi == pytest.approx(90.0)


def test_suggest_array_spreads_azimuths():
    table = P.suggest_array(13)
    azimuths = np.sort(table["azimuth"].to_numpy()[1:])
    gaps = np.diff(np.r_[azimuths, azimuths[0] + 360.0])
    assert gaps.max() < 90.0, "no large azimuthal hole"


def test_suggest_array_refuses_too_few_nodes():
    with pytest.raises(ValueError, match="at least 4 nodes"):
        P.suggest_array(3)
