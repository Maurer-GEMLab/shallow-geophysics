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
    Refraction tomography: a 2-D velocity image from the picks of many
    shots into one spread. Shortest-path rays and a regularised Gauss-Newton
    inversion in NumPy/SciPy, with pyGIMLi as an optional second backend.
    Needs a shot every one or two geophones; a handful of shots is a job for
    ``layers``.
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
from .tomography import (
    TomographyGrid,
    TomographyResult,
    lambda_sweep,
    make_grid,
    reciprocity,
    refine_picks,
    synthetic_traveltimes,
    traveltime_tomography,
)

__all__ = [
    "pick_first_breaks", "plot_picks", "plot_traveltimes", "traveltime_table",
    "PickingSession", "load_shots",
    "traveltimes", "intercept_times", "crossover_distances",
    "depths_from_intercepts", "fit_layers", "LayeredRefractionModel",
    "traveltime_tomography", "make_grid", "reciprocity", "refine_picks", "lambda_sweep",
    "synthetic_traveltimes", "TomographyGrid", "TomographyResult",
]
