"""Electrical resistivity tomography (ERT).

``data``
    :class:`ERTData` -- electrodes and four-electrode measurements -- and
    the RES2DINV ``.dat`` reader and writer, including the repairs a
    SuperSting export needs: integer ``n`` and dropping measurements made
    with electrodes past the end of the cable (:meth:`ERTData.trim`), and
    finding readings missing on the cable (:meth:`ERTData.coverage`).
    Pseudosections, geometric factors, and measurement schemes for
    synthetic surveys.
``forward``
    2.5-D finite-element forward modelling on a rectangular mesh, with
    sensitivities by reciprocity. NumPy and SciPy only.
``inversion``
    Smoothness-constrained Gauss-Newton inversion for a 2-D resistivity
    image, the choice of ``lam``, and the Oldenburg & Li depth of
    investigation.

Assumptions: 2-D ground, electrodes on a straight line on a flat surface.
"""

from .data import (
    ARRAYS,
    ERTData,
    dipole_dipole,
    geometric_factor,
    median_depth,
    read_res2dinv,
    schlumberger,
    wenner,
)
from .forward import ERTMesh, apparent_resistivity, forward, make_mesh
from .inversion import ERTResult, doi_index, lambda_sweep, resistivity_tomography

__all__ = [
    "ARRAYS", "ERTData", "read_res2dinv", "geometric_factor", "median_depth",
    "dipole_dipole", "wenner", "schlumberger",
    "ERTMesh", "make_mesh", "forward", "apparent_resistivity",
    "ERTResult", "resistivity_tomography", "lambda_sweep", "doi_index",
]
