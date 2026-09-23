"""Active-source surface-wave analysis (MASW).

Three steps, three modules:

``dispersion``
    Wavefield transform of a shot gather into a frequency--phase-velocity
    image (Park et al., 1998 phase-shift method) and extraction of a
    dispersion curve from it.
``models``
    The 1D layered shear-wave model and its forward dispersion response,
    computed with ``disba`` (``pip install "shallow-geophysics[masw]"``).
``inversion``
    Fitting a layered model to a picked dispersion curve.

The result of an MASW inversion is a 1D Vs profile located at the spread
midpoint. Projecting many such profiles onto the shared subsurface model is
Layer 3 work (see ``docs/architecture.md``, ADR-001), not done here.
"""

from .dispersion import (
    DispersionCurve,
    DispersionImage,
    dispersion_image,
    phase_shift,
    spread_wavelength_limits,
)
from .inversion import InversionResult, initial_model, invert_dispersion
from .models import LayeredModel, forward_dispersion

__all__ = [
    "DispersionCurve", "DispersionImage", "dispersion_image", "phase_shift",
    "spread_wavelength_limits",
    "LayeredModel", "forward_dispersion",
    "InversionResult", "initial_model", "invert_dispersion",
]
