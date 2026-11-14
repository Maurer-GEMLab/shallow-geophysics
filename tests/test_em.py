"""Frequency-domain EM: forward model, deployment metadata, GEM-2 reader, processing."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import shallowgeo as sg
from shallowgeo import em
from shallowgeo.drivers.gem2 import parse_header_line, read_gem2, read_gem2_gps
from shallowgeo.em import plotting

S = 1.66
EXAMPLE = (Path(__file__).parents[1] / "examples" / "data" / "gem2-038-em-lines"
           / "raw")

HEADER = ("#File:001_7007.GPX,Exporter:1.45,Time:18:56-18:57,6 Samples, 2 GPS,"
          "sn:G2-000-7007,fw:TEST,Voltage:10.4V 12.2V,Temperature:31.1C 36.0C\n"
          "Line ,Sample,Latitude,     Longitude,   Mark,Status,GPSStat,GPSalt(m),"
          "Time(ms) ,Time(hhmmss.sss) ,PowerLn(nT),I_5310Hz(ppm)   ,Q_5310Hz(ppm)   ,"
          "I_63030Hz(ppm) ,Q_63030Hz(ppm) ,QSum(ppm)     ,Econd_5310Hz (mS/m)  ,"
          "Econd_63030Hz (mS/m) ,MSusc_63030Hz (1/1000) ,FitError(%)\n")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture
def gem2_file(tmp_path):
    """Two short lines walked in opposite directions, 6 Hz samples."""
    rows = []
    t0 = 68196582.5
    for i in range(6):
        line = 1 if i < 3 else 2
        lat = 37.9478 + (i if line == 1 else 8 - i) * 2e-6
        q63 = 5000 + 10 * i
        rows.append(
            f" {line}, {i % 3 + 1}, {lat:.8f}, -91.78625, 0, 0, 2, 340.5, "
            f"{t0 + 166.7 * i:.1f}, {185636.58 + 0.1667 * i:.2f}, 1.8, "
            f"-340.0, 130.0, -1900.0, {q63:.1f}, 5130.0, 7.5, 28.5, 19.0, 120.0\n")
    p = tmp_path / "001_7007_gem+inv.csv"
    p.write_text(HEADER + "".join(rows))
    return p


# -- forward model ----------------------------------------------------------------

@pytest.mark.parametrize("orientation", ["HCP", "VCP"])
@pytest.mark.parametrize("sigma", [1e-3, 0.03, 0.3, 3.0])
@pytest.mark.parametrize("freq", [450.0, 63030.0])
def test_halfspace_matches_wait_closed_form(orientation, sigma, freq):
    numeric = em.halfspace_response(sigma, freq, S, 0.0, orientation)
    exact = em.wait_halfspace(sigma, freq, S, orientation)
    assert abs(numeric - exact) / abs(exact) < 1e-3


@pytest.mark.parametrize("orientation", ["HCP", "VCP"])
@pytest.mark.parametrize("height", [0.0, 0.25, 1.0, 3.0])
def test_lin_limit_with_height(orientation, height):
    # Very low induction number: the full solution reduces to McNeill's
    # formula with the cumulative-response height factor.
    full = em.halfspace_response(1e-3, 100.0, S, height, orientation).imag
    lin = em.lin_quadrature(1e-3, 100.0, S, height, orientation)
    assert full == pytest.approx(lin, rel=0.01)


def test_layered_reduces_to_halfspace():
    a = em.layered_response([0.02, 0.02, 0.02], [0.5, 1.0], [1e3, 6e4], S, 1.0)
    b = em.halfspace_response(0.02, [1e3, 6e4], S, 1.0)
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_layered_conductive_basement_raises_quadrature():
    top = em.layered_response([0.01, 0.1], [2.0], 63030, S, 1.0).imag
    assert em.halfspace_response(0.01, 63030, S, 1.0).imag < top
    assert top < em.halfspace_response(0.1, 63030, S, 1.0).imag


@pytest.mark.parametrize("orientation", ["HCP", "VCP"])
@pytest.mark.parametrize("height", [0.0, 0.2, 1.0])
def test_susceptibility_static_factor(orientation, height):
    full = em.halfspace_response(1e-7, 300.0, S, height, orientation,
                                 susceptibility=1e-3).real
    assert full == pytest.approx(
        1e-3 * em.lin_susceptibility_factor(height, S, orientation), rel=0.01)


def test_hcp_inphase_changes_sign_with_height():
    assert em.lin_susceptibility_factor(0.25, S) > 0     # sled
    assert em.lin_susceptibility_factor(1.0, S) < 0      # hand-carried
    assert em.lin_susceptibility_factor(0.354 * S, S) == pytest.approx(0, abs=1e3)


def test_depth_of_investigation():
    assert em.depth_of_investigation(S, "HCP") / S == pytest.approx(1.59, abs=0.01)
    assert em.depth_of_investigation(S, "VCP") / S == pytest.approx(0.76, abs=0.01)
    assert em.cumulative_response(0.0, S, "HCP", 1.0) == pytest.approx(1.0)


def test_skin_depth_and_induction_number():
    assert em.skin_depth(0.01, 1000.0) == pytest.approx(503.3 * np.sqrt(100 / 1000), rel=1e-3)
    assert em.induction_number(0.03, 63030, S) < 0.15


def test_apparent_conductivity_round_trip_per_sample_heights():
    sig = np.array([0.002, 0.02, 0.2])
    h = np.array([0.3, 1.0, 2.5])
    q = np.array([em.halfspace_response(s, 18030, S, hh).imag for s, hh in zip(sig, h)])
    back = em.apparent_conductivity(q, 18030, S, h)
    np.testing.assert_allclose(back, sig, rtol=2e-3)


def test_apparent_conductivity_nan_for_nonpositive_quadrature():
    out = em.apparent_conductivity(np.array([-5.0, 0.0, 300.0]), 8850, S, 1.0)
    assert np.isnan(out[:2]).all() and np.isfinite(out[2])
    lin = em.apparent_conductivity(np.array([-5.0, 300.0]), 8850, S, 1.0, method="lin")
    assert np.isnan(lin[0]) and lin[1] > 0


# -- deployment --------------------------------------------------------------------

def test_deployment_platform_defaults_are_marked_assumed():
    d = em.Deployment.for_platform("sled")
    assert d.sensor_height == 0.25
    assert d.height_source == "nominal"
    assert {"sensor_height", "gps_offset", "tow_distance"} <= set(d.assumed)
    m = d.with_changes(sensor_height=0.18)
    assert "sensor_height" not in m.assumed and m.height_source == "measured"
    assert em.Deployment.from_dict(d.to_dict()) == d


def test_drone_needs_a_height():
    with pytest.raises(ValueError, match="drone"):
        em.Deployment.for_platform("drone")
    d = em.Deployment.for_platform("drone", heights_per_sample=True, sling_length=3.0)
    assert d.sensor_height is None and "sling_length" not in d.assumed


def test_deployment_rejects_unknown_values():
    with pytest.raises(ValueError, match="platform"):
        em.Deployment.for_platform("backpack")
    with pytest.raises(ValueError, match="orientation"):
        em.Deployment.for_platform("handheld", orientation="coaxial")


# -- reader -----------------------------------------------------------------------

def test_reader_requires_platform(gem2_file):
    with pytest.raises(ValueError, match="platform"):
        read_gem2(gem2_file)


def test_reader_parses_channels_vendor_and_time(gem2_file):
    s = read_gem2(gem2_file, platform="handheld", date="2026-11-12")
    assert list(s.frequencies) == [5310.0, 63030.0]
    assert list(s.channels.index) == ["I_5310", "Q_5310", "I_63030", "Q_63030"]
    assert "econd_63030_mS_m" in s.vendor and "econd_63030_mS_m" not in s.readings
    assert s.readings["time"].iloc[0] == pd.Timestamp("2026-11-12 18:56:36.5825", tz="UTC")
    assert s.metadata["serial"] == "G2-000-7007" and s.metadata["date_known"]
    assert s.spatial_ref.crs.to_epsg() == 32615
    assert s.heights().tolist() == [1.0] * 6
    assert s.quadrature(63030)[1] == pytest.approx(5010.0)
    assert s.provenance[0].parameters["platform"] == "handheld"


def test_reader_per_sample_height_for_drone(gem2_file):
    h = np.linspace(2.0, 3.0, 6)
    s = read_gem2(gem2_file, platform="drone", height=h, sling_length=2.5)
    np.testing.assert_allclose(s.heights(), h)
    assert s.deployment.height_source == "per_sample"


def test_driver_registered_and_sniffs(gem2_file, tmp_path):
    assert sg.identify(gem2_file)[0].name == "gem2"
    s = sg.read(gem2_file, platform="sled", sensor_height=0.2)
    assert s.deployment.platform == "sled" and s.deployment.sensor_height == 0.2
    other = tmp_path / "x.csv"
    other.write_text("a,b\n1,2\n")
    assert "gem2" not in [d.name for d in sg.identify(other)]


def test_header_line():
    facts = parse_header_line(HEADER.splitlines()[0])
    assert facts["n_samples"] == 6 and facts["n_gps"] == 2 and facts["fw"] == "TEST"


# -- processing ---------------------------------------------------------------------

def _synthetic(n=60, offset_reverse=0.0):
    """Two straight east-west lines 2 m apart, the second walked westwards."""
    x = np.concatenate([np.linspace(0, 10, n), np.linspace(10, 0, n)])
    y = np.concatenate([np.zeros(n), np.full(n, 2.0)])
    line = np.repeat([1, 2], n)
    rng = np.random.default_rng(1)
    q = em.halfspace_response(0.02, 18030, S, 1.0).imag + rng.normal(0, 5, 2 * n)
    q[line == 2] += offset_reverse
    r = pd.DataFrame({
        "line": line, "sample": np.arange(2 * n), "x": x, "y": y, "z": 340.0,
        "time": pd.Timestamp("2026-11-12", tz="UTC") + pd.to_timedelta(np.arange(2 * n) / 6, "s"),
        "I_18030": rng.normal(0, 5, 2 * n), "Q_18030": q})
    return em.FDEMSurvey(r, instrument=em.GEM2,
                         deployment=em.Deployment.for_platform("handheld"),
                         spatial_ref=sg.local_grid(0, 0))


def test_add_distance_straight_line():
    s = em.add_distance(_synthetic())
    r = s.readings
    assert r.loc[r.line == 1, "distance"].max() == pytest.approx(10.0)
    assert r.loc[r.line == 1, "heading"].iloc[0] == pytest.approx(90.0)
    assert r.loc[r.line == 2, "heading"].iloc[0] == pytest.approx(270.0)


def test_despike_removes_injected_spike():
    s = _synthetic()
    s.readings.loc[20, "Q_18030"] += 500
    d = em.despike(s)
    assert d.readings["spike"].sum() == 1 and np.isnan(d.readings.loc[20, "Q_18030"])
    assert d.provenance[-1].parameters["removed"] == 1


def test_heading_check_flags_direction_offset():
    clean = em.heading_check(em.add_distance(_synthetic()))
    bad = em.heading_check(em.add_distance(_synthetic(offset_reverse=60.0)))
    assert clean.loc["Q_18030", "ratio"] < 2 < bad.loc["Q_18030", "ratio"]


def test_subtract_offsets_once_with_source():
    s = em.subtract_offsets(_synthetic(), {"I_18030": 10.0}, source="test")
    with pytest.raises(ValueError, match="already"):
        em.subtract_offsets(s, {"I_18030": 10.0}, source="test")


def test_gps_layback_moves_positions_back_along_travel():
    s = _synthetic().with_deployment(gps_offset=(3.0, 0.0, 0.0))
    out = em.apply_gps_offset(s)
    r0, r1 = s.readings, out.readings
    np.testing.assert_allclose(r1.loc[r1.line == 1, "x"], r0.loc[r0.line == 1, "x"] - 3.0)
    np.testing.assert_allclose(r1.loc[r1.line == 2, "x"], r0.loc[r0.line == 2, "x"] + 3.0)


def test_heights_from_altitude():
    s = _synthetic().with_deployment(platform="drone", sling_length=2.0)
    s.readings["z"] = 346.0
    out = em.heights_from_altitude(s, 340.0)
    np.testing.assert_allclose(out.heights(), 4.0)
    assert out.deployment.height_source == "gps_minus_ground"


def test_apparent_conductivity_column_recovers_ground():
    s = em.add_apparent_conductivity(_synthetic(), 18030)
    assert np.nanmedian(s.readings["cond_18030"]) == pytest.approx(20.0, rel=0.02)


def test_grid_masks_far_from_track():
    _, gy, z = em.grid(_synthetic(), "Q_18030", cell=0.5, max_distance=0.6)
    assert np.isnan(z[gy == 1.0]).all()
    assert np.isfinite(z[gy == 0.0]).sum() > 10


def test_plots_run():
    s = em.add_apparent_conductivity(em.add_distance(_synthetic()), 18030)
    plotting.plot_qc(s)
    plotting.plot_track(s)
    plotting.plot_track(s, "cond_18030")
    plotting.plot_channels(s, "Q")
    plotting.plot_profiles(s, ["Q_18030", "cond_18030"])
    plotting.plot_map(s, "cond_18030")
    plotting.plot_frequency_response(s)
    plotting.plot_depth_response()


# -- the example dataset ---------------------------------------------------------------

@pytest.mark.skipif(not EXAMPLE.is_dir(), reason="example dataset not present")
def test_example_dataset_matches_vendor_conductivity():
    s = sg.read(EXAMPLE / "038_7007_gem+inv.csv", platform="handheld")
    assert len(s) == 492 and s.lines == [1, 2, 3, 4, 5]
    assert s.metadata["samples_in_header"] == 526
    s = em.add_apparent_conductivity(em.despike(s), [5310, 18030])
    # Geophex's own apparent conductivity is the half-space value for HCP
    # coils 1 m up: reproducing it is how the deployment was inferred.
    for f in (5310, 18030):
        ratio = s.readings[f"cond_{f}"] / s.vendor[f"econd_{f}_mS_m"]
        assert np.nanmedian(ratio) == pytest.approx(1.0, abs=0.02)
    plotting.plot_vendor_inversion(em.add_distance(s), 1)
    gps = read_gem2_gps(EXAMPLE / "038_7007_gps.csv")
    assert len(gps) == 86 and (gps["quality"] == 2).all()
