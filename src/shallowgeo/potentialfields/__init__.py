"""Gravity and magnetics: forward models and inversion.

Most of this is 2-D: bodies run to infinity along strike, at right angles to
a straight profile -- a fair description of a tunnel, a buried channel or a
dyke crossed near its middle. Gravity is also done in 3-D, on a map of
stations over a block of prisms. The fields are computed in closed form, so
nothing beyond NumPy and SciPy is needed.

``bodies``
    Closed-form gravity of simple shapes: the 2-D line mass, horizontal
    cylinder and rectangular prism; the 3-D point mass, sphere and vertical
    cylinder.
``mesh``
    :class:`Mesh2D`, a section of rectangular cells, :class:`Mesh3D`, a
    block of prisms, and :func:`make_mesh` / :func:`make_mesh_3d` with
    padding cells around a core.
``forward``
    :func:`gravity_kernel` (mGal per g/cm^3), :func:`gravity_kernel_3d` and
    :func:`magnetic_kernel` (nT per SI susceptibility, induced
    magnetisation): the matrices ``G`` in ``d = G m``.
``inversion``
    :class:`LinearInversion`: smallest-and-smoothest model with depth
    weighting, ``beta`` chosen by the discrepancy principle, optional bounds;
    on either mesh.

SimPEG's ``potential_fields`` does all of this in 3-D at scale; ADR-001
plans it for the canonical-grid work. This module is the teaching version.
"""

from .bodies import (
    horizontal_cylinder,
    line_mass,
    point_mass,
    prism_2d,
    sphere,
    vertical_cylinder,
    vertical_cylinder_on_axis,
)
from .forward import (
    GRAVITATIONAL_CONSTANT,
    field_direction,
    gravity_kernel,
    gravity_kernel_3d,
    magnetic_kernel,
)
from .inversion import (
    DEPTH_EXPONENT,
    DEPTH_EXPONENT_3D,
    InversionResult,
    LinearInversion,
    depth_weights,
)
from .mesh import Mesh2D, Mesh3D, make_mesh, make_mesh_3d

__all__ = [
    "DEPTH_EXPONENT",
    "DEPTH_EXPONENT_3D",
    "GRAVITATIONAL_CONSTANT",
    "InversionResult",
    "LinearInversion",
    "Mesh2D",
    "Mesh3D",
    "depth_weights",
    "field_direction",
    "gravity_kernel",
    "gravity_kernel_3d",
    "horizontal_cylinder",
    "line_mass",
    "magnetic_kernel",
    "make_mesh",
    "make_mesh_3d",
    "point_mass",
    "prism_2d",
    "sphere",
    "vertical_cylinder",
    "vertical_cylinder_on_axis",
]
