"""Passive surface-wave analysis: ambient noise and traffic-sourced records.

Where :mod:`shallowgeo.surfacewave` handles a hammer shot into a spread, this
handles a set of free-running nodes with no source at all. The end product is
the same -- a dispersion curve to feed to
:func:`~shallowgeo.surfacewave.invert_dispersion` -- but getting there takes a
different route and, more to the point, a different set of things that can go
wrong.

The workflow, and the module that owns each step:

``deployment``
    Merge, trim to a window where every node recorded, and work out what
    wavelengths the array can resolve at all.
``spac``
    Spatial autocorrelation for ambient noise. Includes
    :func:`separation_test`, which decides whether the SPAC assumptions hold
    for the data in hand instead of assuming they do.
``events``
    Passive MASW, for sites where the ambient field is too weak but passing
    traffic is not.

Which of the two dispersion routes applies is a property of the recording,
not a preference. Run the diagnostics first: a quiet site with a strong road
wants ``events``, a noisy urban site with a compact array wants ``spac``, and
a site that offers neither is worth knowing about before a Vs profile gets
built on top of it.
"""

from .deployment import (
    ArrayLayout,
    ResolutionLimits,
    array_layout,
    common_window,
    node_coverage,
    resolution_limits,
    suggest_array,
    trim,
)
from .events import (
    EventWindow,
    detect_events,
    event_dispersion_image,
    shot_windows,
    two_station_dispersion,
)
from .spac import (
    SeparationTest,
    SpacResult,
    WavelengthTest,
    konno_ohmachi,
    separation_rings,
    separation_test,
    spac_coherency,
    spac_dispersion,
    wavelength_test,
)

__all__ = [
    "ArrayLayout",
    "EventWindow",
    "ResolutionLimits",
    "SeparationTest",
    "SpacResult",
    "WavelengthTest",
    "array_layout",
    "common_window",
    "detect_events",
    "event_dispersion_image",
    "konno_ohmachi",
    "node_coverage",
    "resolution_limits",
    "separation_rings",
    "separation_test",
    "shot_windows",
    "spac_coherency",
    "spac_dispersion",
    "suggest_array",
    "trim",
    "two_station_dispersion",
    "wavelength_test",
]
