"""GNSS positioning: corrected exports in, georeferenced stations out.

This package consumes *solutions*, not observations. RINEX processing is left
to RTKLIB (Emlid Studio, RTKPOST) and PPP services (CSRS-PPP, OPUS); see
:mod:`shallowgeo.positions.readers` for why. What lives here is the part that
actually goes wrong in practice: antenna-height reduction, quality gating,
absolute time, duplicate marks, base shifts, and occupation matching.
"""

from .base_station import BaseStation, apply_base_shift, assumed_base_from
from .matching import attach_positions, match_occupations
from .readers import HEIGHT_REFERENCES, PROFILES, read_positions
from .table import PositionTable

__all__ = [
    "PositionTable",
    "read_positions", "PROFILES", "HEIGHT_REFERENCES",
    "BaseStation", "apply_base_shift", "assumed_base_from",
    "match_occupations", "attach_positions",
]
