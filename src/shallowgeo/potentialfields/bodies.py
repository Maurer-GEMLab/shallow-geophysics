"""Gravity of simple bodies in closed form: the shapes of a first course.

Each function returns the vertical gravity anomaly ``g_z`` in mGal, positive
down, so a mass excess gives a positive anomaly. Density contrasts are in
g/cm^3 and masses in kg, as elsewhere in :mod:`shallowgeo.potentialfields`.
``depth`` is below the ground and stations sit ``height`` metres above it, so
the vertical distance from a station to a source is ``depth + height``.

2-D bodies (infinite along strike, profile ``x`` across it)

``line_mass``
    The 2-D "point mass": a horizontal line of mass per metre ``lambda``,

        g_z = 2 G lambda z / (x^2 + z^2).

    A horizontal cylinder of radius ``R`` has exactly this field outside it,
    with ``lambda = drho pi R^2`` (:func:`horizontal_cylinder`).
``prism_2d``
    A rectangle in cross-section; the line mass integrated over its area.

3-D bodies (stations on a map ``x, y``)

``point_mass``
    Newton's law, vertical component: ``g_z = G M z / r^3``.
``sphere``
    Identical to a point mass of ``M = 4/3 pi R^3 drho`` at its centre --
    the shell theorem. No station outside a sphere can tell the two apart.
``vertical_cylinder``
    A bundle of vertical line segments, each integrated in closed form over
    depth, summed over the disc by Gauss-Legendre quadrature.
"""

from __future__ import annotations

import numpy as np

from .forward import GRAVITATIONAL_CONSTANT, gravity_kernel
from .mesh import Mesh2D

__all__ = [
    "horizontal_cylinder",
    "line_mass",
    "point_mass",
    "prism_2d",
    "sphere",
    "vertical_cylinder",
    "vertical_cylinder_on_axis",
]

_MGAL = 1e5          # m/s^2 -> mGal
_KG_M3 = 1e3         # g/cm^3 -> kg/m^3


def _check(depth, height):
    """Depth of the source below the stations; arrays broadcast."""
    if np.any(np.asarray(height) < 0):
        raise ValueError("stations must be on or above the ground: height >= 0")
    z = np.asarray(depth, dtype=float) + np.asarray(height, dtype=float)
    if np.any(z <= 0):
        raise ValueError("the source must be below the stations: depth + height > 0")
    return z


# --- 2-D -------------------------------------------------------------------

def line_mass(x, *, x0: float = 0.0, depth: float, mass_per_length: float,
              height: float = 0.2) -> np.ndarray:
    """``g_z`` (mGal) of an infinite horizontal line mass along strike.

    ``mass_per_length`` in kg/m (negative for a mass deficit).
    """
    z = _check(depth, height)
    dx = np.asarray(x, dtype=float) - x0
    return 2 * GRAVITATIONAL_CONSTANT * mass_per_length * z / (dx**2 + z**2) * _MGAL


def horizontal_cylinder(x, *, x0: float = 0.0, depth: float, radius: float,
                        density: float, height: float = 0.2) -> np.ndarray:
    """``g_z`` (mGal) of a horizontal cylinder: a line mass of
    ``density * pi * radius^2`` through its axis at ``depth``."""
    if depth - radius < 0:
        raise ValueError("the cylinder sticks out of the ground (depth < radius)")
    lam = density * _KG_M3 * np.pi * radius**2
    return line_mass(x, x0=x0, depth=depth, mass_per_length=lam, height=height)


def prism_2d(x, *, x_left: float, x_right: float, top: float, bottom: float,
             density: float, height: float = 0.2) -> np.ndarray:
    """``g_z`` (mGal) of a 2-D prism of rectangular cross-section.

    Closed form: the corner sum of ``x ln r + z atan(x / z)`` (see
    :func:`~shallowgeo.potentialfields.gravity_kernel`, which this calls on a
    one-cell mesh).
    """
    if not (x_right > x_left and bottom > top >= 0):
        raise ValueError("need x_right > x_left and bottom > top >= 0")
    if height <= 0:
        raise ValueError("prism_2d needs height > 0 (put the sensor a little above ground)")
    cell = Mesh2D([x_left, x_right], [top, bottom])
    return density * gravity_kernel(cell, x, height)[:, 0]


# --- 3-D -------------------------------------------------------------------

def point_mass(x, y, *, x0: float = 0.0, y0: float = 0.0, depth: float, mass: float,
               height: float = 0.2) -> np.ndarray:
    """``g_z`` (mGal) of a point mass ``mass`` kg at ``(x0, y0, depth)``.

    ``x`` and ``y`` broadcast against each other: give a profile with
    ``y=0`` or a map with two 2-D arrays.
    """
    z = _check(depth, height)
    dx = np.asarray(x, dtype=float) - x0
    dy = np.asarray(y, dtype=float) - y0
    return GRAVITATIONAL_CONSTANT * mass * z / (dx**2 + dy**2 + z**2) ** 1.5 * _MGAL


def sphere(x, y, *, x0: float = 0.0, y0: float = 0.0, depth: float, radius: float,
           density: float, height: float = 0.2) -> np.ndarray:
    """``g_z`` (mGal) of a buried sphere: a point mass of ``4/3 pi R^3 drho``."""
    if depth - radius < 0:
        raise ValueError("the sphere sticks out of the ground (depth < radius)")
    mass = density * _KG_M3 * 4 / 3 * np.pi * radius**3
    return point_mass(x, y, x0=x0, y0=y0, depth=depth, mass=mass, height=height)


def vertical_cylinder(x, y, *, x0: float = 0.0, y0: float = 0.0, radius: float,
                      top: float, bottom: float, density: float, height: float = 0.2,
                      n: int = 24) -> np.ndarray:
    """``g_z`` (mGal) of a vertical cylinder from ``top`` to ``bottom`` depth.

    A vertical line segment of mass ``rho dA`` per metre, at horizontal
    distance ``s`` from the station, pulls down with

        G rho dA [1 / sqrt(s^2 + z1^2) - 1 / sqrt(s^2 + z2^2)]

    (``z1``, ``z2`` its ends below the station). The cylinder is the sum of
    such segments over its disc, done by ``n``-point Gauss-Legendre in
    radius and a ``2n``-point rule in angle -- exact to well under a
    microgal for stations outside the cylinder at the default ``n``.
    """
    if not (bottom > top >= 0 and radius > 0):
        raise ValueError("need bottom > top >= 0 and radius > 0")
    z1, z2 = top + height, bottom + height
    if z1 <= 0:
        raise ValueError("the cylinder must be below the stations")
    u, w = np.polynomial.legendre.leggauss(n)
    r = radius * (u + 1) / 2
    wr = w * radius / 2 * r                       # dA = r dr dtheta
    theta = np.arange(2 * n) * np.pi / n
    wt = np.pi / n
    xs = (r[None, :] * np.cos(theta[:, None])).ravel() + x0
    ys = (r[None, :] * np.sin(theta[:, None])).ravel() + y0
    dA = np.broadcast_to(wr[None, :] * wt, (2 * n, n)).ravel()

    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    s2 = (x[..., None] - xs) ** 2 + (y[..., None] - ys) ** 2
    k = 1 / np.sqrt(s2 + z1**2) - 1 / np.sqrt(s2 + z2**2)
    return GRAVITATIONAL_CONSTANT * density * _KG_M3 * (k @ dA) * _MGAL


def vertical_cylinder_on_axis(*, radius: float, top: float, bottom: float, density: float,
                              height: float = 0.2) -> float:
    """``g_z`` (mGal) directly above the axis of a vertical cylinder, exact:

        2 pi G rho [L + sqrt(z1^2 + R^2) - sqrt(z2^2 + R^2)]

    (Telford et al., 1990). A check on :func:`vertical_cylinder`.
    """
    z1, z2 = top + height, bottom + height
    return float(2 * np.pi * GRAVITATIONAL_CONSTANT * density * _KG_M3
                 * ((z2 - z1) + np.hypot(z1, radius) - np.hypot(z2, radius)) * _MGAL)
