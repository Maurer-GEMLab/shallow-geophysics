"""Ground magnetics: station tables, quality checks, and plots.

Works on the :class:`~shallowgeo.core.survey.PointSurvey` that the
``mag-summary`` driver (per-station tables) produces. A G-857 dump read with
the ``g857`` driver plots the same way once its readings carry ``line`` and
``distance`` columns.

``plot_profiles``
    One panel per line, field against distance, with the stations whose
    repeat readings disagreed marked. The first plot to look at.
``plot_stacked_profiles``
    Every line drawn at its own position on one plan view, the anomaly as a
    wiggle -- shows whether an anomaly lines up from line to line.
``plot_map``
    Gridded contour map of the field or the anomaly.
``diurnal_correction`` / ``plot_diurnal``
    Base-station drift removal, and the drift curve to check it by.
``plot_corrections`` / ``plot_repeatability``
    The two checks on a hand-in table: what was added to each line, and how
    well the readings at each station agreed.
``synthetic_table``
    A made-up survey with base-station readings, for demonstrations.
"""

from .diurnal import base_station_id, diurnal_correction, plot_diurnal
from .plotting import (
    flag_noisy,
    plot_corrections,
    plot_map,
    plot_profiles,
    plot_repeatability,
    plot_stacked_profiles,
    remove_regional,
    station_table,
)
from .synthetic import synthetic_table

__all__ = [
    "base_station_id",
    "diurnal_correction",
    "flag_noisy",
    "plot_corrections",
    "plot_diurnal",
    "plot_map",
    "plot_profiles",
    "plot_repeatability",
    "plot_stacked_profiles",
    "remove_regional",
    "station_table",
    "synthetic_table",
]
