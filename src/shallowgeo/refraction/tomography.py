"""Refraction travel-time tomography -- placeholder (roadmap Milestone 5).

Design, so the eventual implementation has a target:

* Input is the travel-time table from :func:`~shallowgeo.refraction.picking.traveltime_table`
  for **many** shots into the same spread. Tomography needs ray coverage;
  four shots into 24 geophones gives a layered interpretation, not an image.
* The solver is pyGIMLi's ``TravelTimeManager`` (SimPEG has no refraction
  tomography; see ADR-001). The mesh is generated from the canonical grid
  and results are projected back onto it, per ADR-001 Option B.
* The wrapper builds a pyGIMLi ``DataContainer`` with sensors at the union
  of source and receiver positions, ``s``/``g`` index columns, ``t`` in
  seconds, and an error column from pick quality; runs the inversion with
  a starting gradient model from :func:`~shallowgeo.refraction.layers.fit_layers`;
  and returns a velocity field plus ray coverage on the mesh.

Until then this module only tells you what it needs.
"""

from __future__ import annotations

import pandas as pd

MIN_SHOTS = 5


def traveltime_tomography(table: pd.DataFrame, **kwargs):
    """Not yet implemented. Raises with the data requirement it will have."""
    n_shots = table["shot"].nunique() if "shot" in table else 1
    raise NotImplementedError(
        "Travel-time tomography is roadmap Milestone 5 (pyGIMLi "
        "TravelTimeManager wrapper). It will need dense coverage: at least "
        f"{MIN_SHOTS} shots into the spread, ideally shots at every few "
        f"geophones. This table has {n_shots}. For sparse data use "
        "shallowgeo.refraction.fit_layers."
    )
