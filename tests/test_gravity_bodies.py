"""Closed-form gravity of simple bodies, the 3-D prism kernel and 3-D inversion."""

import numpy as np
import pytest

from shallowgeo.potentialfields import (
    GRAVITATIONAL_CONSTANT as G,
)
from shallowgeo.potentialfields import (
    LinearInversion,
    Mesh2D,
    Mesh3D,
    depth_weights,
    gravity_kernel,
    gravity_kernel_3d,
    horizontal_cylinder,
    line_mass,
    make_mesh_3d,
    point_mass,
    prism_2d,
    sphere,
    vertical_cylinder,
    vertical_cylinder_on_axis,
)

MGAL = 1e5


def test_line_mass_peak_and_half_width():
    x = np.array([0.0, 10.2, -10.2])
    g = line_mass(x, depth=10.0, mass_per_length=5e4, height=0.2)
    assert g[0] == pytest.approx(2 * G * 5e4 / 10.2 * MGAL)
    np.testing.assert_allclose(g[1:], g[0] / 2)       # half-width equals depth


def test_horizontal_cylinder_is_a_line_mass():
    x = np.linspace(-30, 30, 13)
    np.testing.assert_allclose(
        horizontal_cylinder(x, depth=8.0, radius=3.0, density=-2.0),
        line_mass(x, depth=8.0, mass_per_length=-2000 * np.pi * 9))
    with pytest.raises(ValueError):
        horizontal_cylinder(x, depth=2.0, radius=3.0, density=1.0)


def test_prism_2d_matches_mesh_kernel_and_far_field():
    x = np.linspace(-20, 20, 9)
    g = prism_2d(x, x_left=-1, x_right=2, top=3, bottom=5, density=0.7, height=0.3)
    ref = 0.7 * gravity_kernel(Mesh2D([-1, 2], [3, 5]), x, 0.3)[:, 0]
    np.testing.assert_allclose(g, ref)
    # far away a prism is a line mass through its centre
    far = np.array([500.0])
    np.testing.assert_allclose(
        prism_2d(far, x_left=-1, x_right=1, top=4, bottom=6, density=1.0),
        line_mass(far, depth=5.0, mass_per_length=1000 * 4.0), rtol=1e-4)


def test_point_mass_and_sphere_are_identical():
    X, Y = np.meshgrid(np.linspace(-20, 20, 5), np.linspace(-10, 30, 5))
    R, rho = 3.0, 1.5
    m = rho * 1000 * 4 / 3 * np.pi * R**3
    np.testing.assert_allclose(
        sphere(X, Y, x0=1, y0=2, depth=7, radius=R, density=rho),
        point_mass(X, Y, x0=1, y0=2, depth=7, mass=m))
    # peak G M / z^2 and 3-D half-width rule
    z = 7.2
    assert point_mass(1.0, 2.0, x0=1, y0=2, depth=7, mass=m) == pytest.approx(G * m / z**2 * MGAL)
    hw = z * np.sqrt(2 ** (2 / 3) - 1)
    assert point_mass(1 + hw, 2.0, x0=1, y0=2, depth=7, mass=m) == pytest.approx(
        G * m / z**2 * MGAL / 2)


def test_vertical_cylinder_on_axis_and_far_field():
    kw = {"radius": 4.0, "top": 2.0, "bottom": 15.0, "density": 0.8, "height": 0.2}
    exact = vertical_cylinder_on_axis(**kw)
    assert vertical_cylinder(0.0, 0.0, **kw) == pytest.approx(exact, rel=1e-10)
    # a short, thin cylinder far away is a point mass at its centre
    far = vertical_cylinder(300.0, 400.0, radius=1.0, top=9.5, bottom=10.5, density=1.0)
    pm = point_mass(300.0, 400.0, depth=10.0, mass=1000 * np.pi)
    assert far == pytest.approx(pm, rel=1e-4)


def test_vertical_cylinder_matches_summed_points_off_axis():
    # Brute force: a fine grid of point masses filling the cylinder.
    R, top, bottom, rho = 3.0, 2.0, 8.0, 1.0
    n = 60
    c = (np.arange(n) + 0.5) / n
    xs, ys, zs = np.meshgrid(2 * R * c - R, 2 * R * c - R, top + (bottom - top) * c,
                             indexing="ij")
    inside = xs**2 + ys**2 <= R**2
    dv = (2 * R / n) ** 2 * (bottom - top) / n
    sx, sy = np.array([0.0, 5.0, 12.0]), np.array([0.0, 2.0, -7.0])
    ref = [np.sum(point_mass(px, py, x0=xs[inside], y0=ys[inside], depth=zs[inside],
                             mass=rho * 1000 * dv)) for px, py in zip(sx, sy)]
    g = vertical_cylinder(sx, sy, radius=R, top=top, bottom=bottom, density=rho)
    np.testing.assert_allclose(g, ref, rtol=2e-3)


def test_gravity_kernel_3d_matches_point_sum_and_slab():
    cell = Mesh3D([-1.0, 2.0], [-2.0, 1.0], [3.0, 5.0])
    x = np.array([-6.0, 0.5, 4.0, 10.0])
    y = np.array([0.0, -0.5, 3.0, -8.0])
    g = gravity_kernel_3d(cell, x, y, 0.3)[:, 0]
    n = 40
    c = (np.arange(n) + 0.5) / n
    px, py, pz = np.meshgrid(-1 + 3 * c, -2 + 3 * c, 3 + 2 * c, indexing="ij")
    dv = 3 * 3 * 2 / n**3
    ref = [np.sum(point_mass(a, b, x0=px.ravel(), y0=py.ravel(), depth=pz.ravel(),
                             mass=1000 * dv, height=0.3)) for a, b in zip(x, y)]
    np.testing.assert_allclose(g, ref, rtol=1e-3)
    slab = Mesh3D([-1e5, 1e5], [-1e5, 1e5], [1.0, 3.0])
    assert gravity_kernel_3d(slab, [0.0], [0.0], 0.2)[0, 0] == pytest.approx(
        2 * np.pi * G * 1000 * 2 * MGAL, rel=1e-4)


def test_mesh3d_layout_and_fractions():
    mesh = make_mesh_3d(-10, 10, -6, 6, 8, cell=2.0, n_pad=2)
    nz, ny, nx = mesh.shape
    assert (nx, ny, nz) == (14, 10, 6)
    x, y, z = mesh.centers()
    model = np.zeros(mesh.n_cells)
    model[np.argmin(np.abs(x - 3) + np.abs(y + 1) + np.abs(z - 5))] = 1
    k, j, i = np.unravel_index(np.argmax(model), mesh.shape)
    assert (mesh.x_centers[i], mesh.y_centers[j], mesh.z_centers[k]) == (3.0, -1.0, 5.0)
    box = mesh.box(-2, 2, -2, 2, 0, 4)
    assert np.sum(box * mesh.volume) == pytest.approx(4 * 4 * 4)
    ball = mesh.fraction(lambda x, y, z: x**2 + y**2 + (z - 4) ** 2 < 3**2, n_sub=8)
    assert np.sum(ball * mesh.volume) == pytest.approx(4 / 3 * np.pi * 27, rel=0.03)
    with pytest.raises(ValueError):
        make_mesh_3d(0, 5, 0, 4, 4, cell=1.5)


def test_mesh2d_fraction_circle():
    mesh = Mesh2D(np.linspace(-5, 5, 21), np.linspace(0, 10, 21))
    f = mesh.fraction(lambda x, z: x**2 + (z - 5) ** 2 < 4**2, n_sub=10)
    assert np.sum(f * mesh.area) == pytest.approx(np.pi * 16, rel=0.01)


def test_3d_inversion_finds_a_buried_sphere():
    xy = np.arange(-24, 24.01, 4.0)
    X, Y = np.meshgrid(xy, xy)
    mesh = make_mesh_3d(-24, 24, -24, 24, 20, cell=4.0, n_pad=3)
    Gm = gravity_kernel_3d(mesh, X, Y, 0.2)
    d = sphere(X, Y, x0=4.0, y0=-4.0, depth=9.0, radius=4.0, density=1.0).ravel()
    std = 0.002
    d = d + np.random.default_rng(0).normal(0, std, d.size)
    inv = LinearInversion(mesh, Gm, d, std, length=4.0, z0=0.2)
    # 3-D gravity gets the 3-D depth weight
    np.testing.assert_allclose(inv.weights, depth_weights(mesh, 1.0, 0.2))
    r = inv.solve()
    assert r.chi2 == pytest.approx(1.0, rel=1e-2)
    k, j, i = np.unravel_index(np.argmax(r.model), mesh.shape)
    assert abs(mesh.x_centers[i] - 4) <= 4 and abs(mesh.y_centers[j] + 4) <= 4
    assert 4 <= mesh.z_centers[k] <= 16
    assert r.model.max() > 0
