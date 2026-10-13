"""Gravity and magnetic fields of 2-D rectangular prisms, in closed form.

Both fields are linear in the physical property, so for a mesh of cells each
datum is a weighted sum of the cell values: ``d = G @ m``. The functions here
build ``G``. Row ``i`` is the sensitivity of station ``i`` to every cell,
which is worth plotting in its own right.

Gravity
-------
A 2-D body of density contrast ``rho`` pulls down with (Telford et al.,
1990; Blakely, 1995)

    g_z = 2 G rho  integral (z' - z0) / r^2  dx' dz'

and for a rectangle the integral is the corner sum of
``x ln r + z atan(x / z)`` with ``x, z`` measured from the station. Units of
the kernel: mGal per g/cm^3.

Magnetics
---------
Induced magnetisation only: each cell is magnetised along the Earth's field
``F`` with strength ``chi F / mu0`` (no remanence, no self-demagnetisation --
fine for chi << 1). The anomalous field of a 2-D body is
``B = (chi F / 2 pi) grad grad U``, where ``U`` is the logarithmic potential
of the cell; its second derivatives over a rectangle are

    U_zz = -U_xx = -corner sum of atan(x / z)
    U_xz =         -corner sum of ln r

What a total-field magnetometer reads is the component of ``B`` along the
Earth's field. Only the part of the field in the plane of the profile
matters: magnetisation along strike makes no external field for an infinite
prism. Units of the kernel: nT per SI susceptibility.

``x`` is along the profile and ``z`` is depth (down); stations are at
``height`` metres above flat ground.
"""

from __future__ import annotations

import numpy as np

from .mesh import Mesh2D

__all__ = [
    "GRAVITATIONAL_CONSTANT",
    "field_direction",
    "gravity_kernel",
    "magnetic_kernel",
]

GRAVITATIONAL_CONSTANT = 6.674e-11  # m^3 kg^-1 s^-2
_MGAL_PER_GCC = 1e5 * 1e3  # m/s^2 -> mGal, and g/cm^3 -> kg/m^3


def _corner_sum(f, mesh: Mesh2D, x: np.ndarray, height: np.ndarray) -> np.ndarray:
    """``(n_stations, n_cells)``: ``f(x_c - x_i, z_c + h_i)`` summed over cell corners."""
    xs = mesh.x_edges[None, None, :] - x[:, None, None]
    zs = mesh.z_edges[None, :, None] + height[:, None, None]
    v = f(xs, zs)
    out = v[:, 1:, 1:] - v[:, :-1, 1:] - v[:, 1:, :-1] + v[:, :-1, :-1]
    return out.reshape(x.size, -1)


def _stations(x, height) -> tuple[np.ndarray, np.ndarray]:
    x = np.atleast_1d(np.asarray(x, dtype=float))
    height = np.broadcast_to(np.asarray(height, dtype=float), x.shape).astype(float)
    if np.any(height <= 0):
        raise ValueError("stations must be above the ground: height > 0 "
                         "(a gravimeter sensor sits about 0.2 m up)")
    return x, height


def gravity_kernel(mesh: Mesh2D, x, height=0.2) -> np.ndarray:
    """Vertical gravity, mGal, at stations ``x`` per g/cm^3 in each cell.

    Positive down, so a mass excess gives a positive anomaly and a void a
    negative one.
    """
    x, height = _stations(x, height)

    def f(dx, z):
        r2 = dx**2 + z**2
        return 0.5 * dx * np.log(r2) + z * np.arctan2(dx, z)

    return 2 * GRAVITATIONAL_CONSTANT * _MGAL_PER_GCC * _corner_sum(f, mesh, x, height)


def field_direction(inclination: float, declination: float = 0.0,
                    profile_azimuth: float = 0.0) -> tuple[float, float, float]:
    """Unit vector of the Earth's field as ``(along profile, along strike, down)``.

    Angles in degrees. ``profile_azimuth`` is the direction of increasing
    ``x``, clockwise from north: 0 for a profile run south to north, 90 for
    west to east.
    """
    inc, rel = np.radians(inclination), np.radians(declination - profile_azimuth)
    return (float(np.cos(inc) * np.cos(rel)), float(np.cos(inc) * np.sin(rel)),
            float(np.sin(inc)))


def magnetic_kernel(
    mesh: Mesh2D,
    x,
    height=1.0,
    *,
    field: float = 52_000.0,
    inclination: float = 64.0,
    declination: float = 0.0,
    profile_azimuth: float = 0.0,
) -> np.ndarray:
    """Total-field anomaly, nT, at stations ``x`` per SI susceptibility in each cell.

    ``field`` is the Earth's field strength in nT; the defaults are roughly
    central Missouri (look up your site with the NOAA/IGRF calculator).
    """
    x, height = _stations(x, height)
    fx, _, fz = field_direction(inclination, declination, profile_azimuth)
    u_zz = -_corner_sum(lambda dx, z: np.arctan2(dx, z), mesh, x, height)
    u_xz = -_corner_sum(lambda dx, z: 0.5 * np.log(dx**2 + z**2), mesh, x, height)
    # B_a = (chi F / 2 pi) U_ab f_b with U_xx = -U_zz; project onto f.
    return field / (2 * np.pi) * ((fz**2 - fx**2) * u_zz + 2 * fx * fz * u_xz)
