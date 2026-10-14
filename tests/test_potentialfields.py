"""2-D gravity and magnetic kernels and the linear inversion."""

import numpy as np
import pytest

from shallowgeo.potentialfields import (
    GRAVITATIONAL_CONSTANT,
    LinearInversion,
    Mesh2D,
    depth_weights,
    field_direction,
    gravity_kernel,
    magnetic_kernel,
    make_mesh,
)


def _point_sums(cell, xs, h, f, n=300, field=52_000.0):
    """Gravity and magnetic response of one cell as a sum of n x n line sources."""
    (x0, x1), (z0, z1) = cell
    xc = x0 + (np.arange(n) + 0.5) * (x1 - x0) / n
    zc = z0 + (np.arange(n) + 0.5) * (z1 - z0) / n
    X, Z = np.meshgrid(xc, zc)
    dA = (x1 - x0) * (z1 - z0) / n**2
    fx, _, fz = f
    g, t = [], []
    for x in xs:
        dx, dz = X - x, Z + h
        r2 = dx**2 + dz**2
        g.append(np.sum(2 * GRAVITATIONAL_CONSTANT * 1e8 * dz / r2) * dA)
        # second derivatives of ln(1/r) for a line source
        u_zz, u_xz = (dz**2 - dx**2) / r2**2, 2 * dx * dz / r2**2
        t.append(np.sum(field / (2 * np.pi) * ((fz**2 - fx**2) * u_zz + 2 * fx * fz * u_xz)) * dA)
    return np.array(g), np.array(t)


def test_gravity_of_wide_layer_is_bouguer_slab():
    slab = Mesh2D([-1e6, 1e6], [1.0, 3.0])
    g = gravity_kernel(slab, [0.0], height=0.2)[0, 0]
    # 2 pi G rho t for 1 g/cm^3 and 2 m, in mGal
    assert g == pytest.approx(2 * np.pi * GRAVITATIONAL_CONSTANT * 1000 * 2 * 1e5, rel=1e-5)


def test_kernels_match_summed_line_sources():
    cell = ((1.0, 3.0), (2.0, 5.0))
    mesh = Mesh2D(cell[0], cell[1])
    xs, h = np.linspace(-6, 8, 7), 0.7
    angles = {"inclination": 64.0, "declination": -3.0, "profile_azimuth": 20.0}
    g = gravity_kernel(mesh, xs, h)[:, 0]
    t = magnetic_kernel(mesh, xs, h, **angles)[:, 0]
    f = field_direction(angles["inclination"], angles["declination"], angles["profile_azimuth"])
    g_ref, t_ref = _point_sums(cell, xs, h, f)
    np.testing.assert_allclose(g, g_ref, rtol=1e-4)
    np.testing.assert_allclose(t, t_ref, rtol=1e-4, atol=1e-3 * np.abs(t_ref).max())


def test_magnetic_anomaly_shape():
    mesh = Mesh2D([-1.0, 1.0], [3.0, 5.0])
    xs = np.linspace(-20, 20, 81)
    # Vertical field: symmetric, positive over a susceptible body.
    t = magnetic_kernel(mesh, xs, 1.0, inclination=90.0)[:, 0]
    np.testing.assert_allclose(t, t[::-1], rtol=1e-9, atol=1e-12)
    assert t[40] > 0 and t[40] == t.max()
    # Northern hemisphere, profile south to north: high to the south, low to the north.
    t = magnetic_kernel(mesh, xs, 1.0, inclination=64.0, profile_azimuth=0.0)[:, 0]
    assert xs[np.argmax(t)] < 0 < xs[np.argmin(t)]
    assert t.min() < 0
    # Profile west to east across a north-south body: the northward part of the
    # field lies along strike, so only the vertical part counts.
    t_ew = magnetic_kernel(mesh, xs, 1.0, inclination=64.0, profile_azimuth=90.0)[:, 0]
    np.testing.assert_allclose(t_ew, t_ew[::-1], atol=1e-9)


@pytest.mark.parametrize("azimuth", [0.0, 90.0, 30.0])
def test_magnetic_kernel_matches_long_3d_prism(azimuth):
    """Independent of the 2-D algebra: sum 3-D point dipoles over a prism
    running 2 km each way along strike, with the strike set by the azimuth."""
    cell, h, field = ((-2.0, 2.0), (4.0, 7.0)), 1.0, 52_000.0
    xs = np.linspace(-15, 15, 7)
    f = np.array(field_direction(64.0, 0.0, azimuth))
    t = magnetic_kernel(Mesh2D(*cell), xs, h, inclination=64.0, profile_azimuth=azimuth)[:, 0]

    d = 0.25
    xc = np.arange(cell[0][0] + d / 2, cell[0][1], d)
    zc = np.arange(cell[1][0] + d / 2, cell[1][1], d)
    half = np.r_[0, np.geomspace(0.05, 2000, 150)]
    y_edges = np.r_[-half[:0:-1], half]
    yc, dy = 0.5 * (y_edges[1:] + y_edges[:-1]), np.diff(y_edges)
    X, Z, Y = np.meshgrid(xc, zc, yc, indexing="ij")
    dV = d * d * dy
    ref = []
    for x in xs:
        r = np.stack([X - x, Y, Z + h])
        R = np.sqrt(np.sum(r**2, axis=0))
        fr = np.tensordot(f, r, 1) / R
        ref.append(np.sum(field / (4 * np.pi) * (3 * fr**2 - 1) / R**3 * dV))
    np.testing.assert_allclose(t, ref, atol=2e-3 * np.abs(ref).max())


def test_field_direction_is_unit():
    f = field_direction(64.0, -2.0, 30.0)
    assert np.linalg.norm(f) == pytest.approx(1.0)
    assert f[2] == pytest.approx(np.sin(np.radians(64.0)))


def test_station_below_ground_rejected():
    with pytest.raises(ValueError, match="above the ground"):
        gravity_kernel(Mesh2D([0, 1], [1, 2]), [0.0], height=0.0)


def test_mesh_rectangle_and_padding():
    mesh = make_mesh(-10, 10, 6, 0.5)
    assert mesh.x_edges[0] < -10 and mesh.x_edges[-1] > 10 and mesh.z_edges[-1] > 6
    frac = mesh.rectangle(-2.2, 1.9, 4.1, 7.0)
    assert np.sum(frac * mesh.area) == pytest.approx(4.1 * 2.9)
    assert frac.max() == 1.0
    with pytest.raises(ValueError, match="whole numbers"):
        make_mesh(-10, 10.3, 6, 0.5)


@pytest.fixture(scope="module")
def tunnel():
    mesh = make_mesh(-30, 30, 12, 0.5)
    model = -2.65 * mesh.rectangle(-2, 2, 4, 7)
    x = np.arange(-25, 25.1, 2.0)
    G = gravity_kernel(mesh, x, 0.2)
    rng = np.random.default_rng(3)
    std = 0.005
    data = G @ model + std * rng.standard_normal(x.size)
    return mesh, G, data, std, model


def test_target_beta_fits_to_noise(tunnel):
    mesh, G, data, std, _ = tunnel
    r = LinearInversion(mesh, G, data, std, weighting="gravity", z0=0.2).solve()
    assert r.chi2 == pytest.approx(1.0, rel=1e-2)
    sweep = LinearInversion(mesh, G, data, std).sweep(np.logspace(0, 4, 5))
    assert np.all(np.diff(sweep["phi_d"]) > 0) and np.all(np.diff(sweep["phi_m"]) < 0)


def test_depth_weighting_moves_the_body_down(tunnel):
    mesh, G, data, std, _ = tunnel
    _, z = mesh.centers()
    flat = LinearInversion(mesh, G, data, std, weighting="none").solve().model
    weighted = LinearInversion(mesh, G, data, std, weighting="gravity", z0=0.2).solve().model
    assert z[np.argmin(flat)] < 1.0
    assert 4.0 <= z[np.argmin(weighted)] <= 7.0
    w = depth_weights(mesh, 0.5, 0.2)
    assert w.max() == 1.0 and np.all(np.diff(w.reshape(mesh.shape)[:, 0]) < 0)


def test_bounds_and_compact(tunnel):
    mesh, G, data, std, model = tunnel
    inv = LinearInversion(mesh, G, data, std, weighting="gravity", z0=0.2)
    smooth = inv.solve()
    bounded = inv.solve(bounds=(-2.65, 0.0))
    assert bounded.model.min() >= -2.65 - 1e-9 and bounded.model.max() <= 1e-9
    assert bounded.chi2 == pytest.approx(1.0, abs=0.05)
    compact = inv.solve_compact(bounds=(-2.65, 0.0), n_iter=8)
    assert compact.chi2 == pytest.approx(1.0, abs=0.1)
    assert compact.model.min() < 3 * bounded.model.min() < 0
    # The smooth setup survives the compact run.
    np.testing.assert_allclose(inv.solve(smooth.beta).model, smooth.model)
    # Mass deficit is what gravity pins down best.
    inside = mesh.core(-15, 15, 12)
    true_mass = np.sum(model * mesh.area)
    assert np.sum((compact.model * mesh.area)[inside]) == pytest.approx(true_mass, rel=0.25)


def test_inversion_reports_data_it_cannot_use(tunnel):
    mesh, G, data, std, _ = tunnel
    # Wrong-sign bounds: no model fits, so say so rather than search forever.
    inv = LinearInversion(mesh, G, data, std, weighting="gravity", z0=0.2)
    with pytest.raises(ValueError, match="sign of the bounds"):
        inv.solve(bounds=(0.0, 2.65))
    # Data already within their noise of zero: there is no beta to find.
    quiet = LinearInversion(mesh, G, 0.1 * std * np.ones(data.size), std)
    with pytest.raises(ValueError, match="nothing to invert"):
        quiet.solve()
