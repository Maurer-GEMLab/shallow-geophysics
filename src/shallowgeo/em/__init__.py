"""Frequency-domain electromagnetic induction (FDEM): ground-conductivity meters.

Written for the Geophex GEM-2 -- a 1.66 m ski with six or so programmable
frequencies -- carried by hand, towed on a sled, or flown under a drone.
The reading is the ground's secondary field in parts per million of the
transmitter's, in-phase and quadrature, at each frequency.

``deployment``
    :class:`Deployment`: the platform (``handheld``/``sled``/``drone``),
    sensor height, coil orientation, GPS antenna offset, tow distance or
    sling length -- and which of those were assumed. :class:`Instrument`
    and :data:`GEM2` hold the coil geometry.
``survey``
    :class:`FDEMSurvey`: samples along the track, channels ``I_<f>`` and
    ``Q_<f>``, the deployment, and the instrument software's own products
    kept apart in ``vendor``.
``forward``
    Response of a layered ground at any height and orientation (checked
    against Wait's closed form), the low-induction-number formulas, skin
    depth, induction number, McNeill's cumulative response and depth of
    investigation, and apparent conductivity from the quadrature.
``processing``
    Distance along line, noise and signal per channel, despiking, zero
    levels, the heading check, line levelling, GPS layback, drone heights
    from altitude, apparent conductivity, gridding.
``plotting``
    QC, track, channels, profiles, maps, frequency response, depth
    response, and the vendor's layered inversion.

Read a GEM-2 export with :func:`shallowgeo.drivers.gem2.read_gem2` or
``shallowgeo.read(path, platform="handheld")``.
"""

from .deployment import (
    BOOM_DIRECTIONS,
    GEM2,
    ORIENTATIONS,
    PLATFORMS,
    Deployment,
    Instrument,
)
from .forward import (
    apparent_conductivity,
    cumulative_response,
    depth_of_investigation,
    halfspace_response,
    induction_number,
    layered_response,
    lin_conductivity,
    lin_height_factor,
    lin_quadrature,
    lin_susceptibility,
    lin_susceptibility_factor,
    skin_depth,
    wait_halfspace,
)
from .processing import (
    add_apparent_conductivity,
    add_distance,
    apply_gps_offset,
    despike,
    free_air_offsets,
    grid,
    heading_check,
    heights_from_altitude,
    level_lines,
    noise_table,
    subtract_offsets,
)
from .survey import FDEMSurvey, channel_name, channel_table

__all__ = [
    "PLATFORMS", "ORIENTATIONS", "BOOM_DIRECTIONS", "Deployment", "Instrument", "GEM2",
    "FDEMSurvey", "channel_name", "channel_table",
    "skin_depth", "induction_number", "lin_height_factor", "cumulative_response",
    "depth_of_investigation", "lin_quadrature", "lin_conductivity",
    "lin_susceptibility", "lin_susceptibility_factor",
    "layered_response", "halfspace_response", "wait_halfspace", "apparent_conductivity",
    "add_distance", "noise_table", "despike", "free_air_offsets", "subtract_offsets",
    "heading_check", "level_lines", "apply_gps_offset", "heights_from_altitude",
    "add_apparent_conductivity", "grid",
]
