"""Gravity and magnetics on a 2-D cross-section: forward models and inversion.

Everything here is 2-D: bodies run to infinity along strike, at right angles
to a straight profile -- a fair description of a tunnel, a buried channel or
a dyke crossed near its middle. The fields are computed in closed form, so
nothing beyond NumPy and SciPy is needed.

``mesh``
    :class:`Mesh2D`, a section of rectangular cells, and :func:`make_mesh`
    with padding cells around a core.
``forward``
    :func:`gravity_kernel` (mGal per g/cm^3) and :func:`magnetic_kernel`
    (nT per SI susceptibility, induced magnetisation): the matrices ``G`` in
    ``d = G m``.
``inversion``
    :class:`LinearInversion`: smallest-and-smoothest model with depth
    weighting, ``beta`` chosen by the discrepancy principle, optional bounds.

SimPEG's ``potential_fields`` does the same in 3-D; ADR-001 plans it for the
canonical-grid work. This module is the teaching version.
"""

from .forward import (
    GRAVITATIONAL_CONSTANT,
    field_direction,
    gravity_kernel,
    magnetic_kernel,
)
from .inversion import DEPTH_EXPONENT, InversionResult, LinearInversion, depth_weights
from .mesh import Mesh2D, make_mesh

__all__ = [
    "DEPTH_EXPONENT",
    "GRAVITATIONAL_CONSTANT",
    "InversionResult",
    "LinearInversion",
    "Mesh2D",
    "depth_weights",
    "field_direction",
    "gravity_kernel",
    "magnetic_kernel",
    "make_mesh",
]
