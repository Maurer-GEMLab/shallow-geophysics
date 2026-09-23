"""Seismic refraction: first breaks, layered interpretation, tomography.

``picking``
    Automatic first-break picking on a ``SeismicSurvey`` and assembly of a
    travel-time table across shots.
``interactive``
    A session that holds the picks for every shot of one line and lets them
    be corrected by hand, in a notebook or from a script.
``layers``
    Forward travel times for horizontal layers, and the intercept-time
    (slope-intercept) fit for two- and three-layer models -- the classroom
    method, done carefully.
``tomography``
    Placeholder for the pyGIMLi ``TravelTimeManager`` wrapper (Milestone 5).
    Dense multi-shot data is required; a four-shot teaching line is not it.
"""

from .interactive import PickingSession, load_shots
from .layers import (
    LayeredRefractionModel,
    crossover_distances,
    depths_from_intercepts,
    fit_layers,
    intercept_times,
    traveltimes,
)
from .picking import (
    pick_first_breaks,
    plot_picks,
    plot_traveltimes,
    traveltime_table,
)

__all__ = [
    "pick_first_breaks", "plot_picks", "plot_traveltimes", "traveltime_table",
    "PickingSession", "load_shots",
    "traveltimes", "intercept_times", "crossover_distances",
    "depths_from_intercepts", "fit_layers", "LayeredRefractionModel",
]
