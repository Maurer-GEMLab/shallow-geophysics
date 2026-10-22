"""Standard array shapes and fitting them to GPS positions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shallowgeo import passive as P
from shallowgeo.core.crs import SpatialRef
from shallowgeo.core.geometry import Geometry
from shallowgeo.core.survey import SeismicSurvey

WGS84 = SpatialRef("EPSG:4326", vertical_datum="ellipsoidal")
METRE_PER_DEGREE = 111320.0
LAT0, LON0 = 37.925, -91.7766


def survey_at(east, north, ids=None):
    """A passive survey whose nodes sit at local east/north metres."""
    ids = ids or [f"n{i}" for i in range(len(east))]
    coslat = np.cos(np.radians(LAT0))
    geometry = Geometry(
        ids=ids,
        x=[LON0 + e / (METRE_PER_DEGREE * coslat) for e in east],
        y=[LAT0 + n / METRE_PER_DEGREE for n in north],
        z=[330.0] * len(ids),
        roles=["receiver"] * len(ids),
        spatial_ref=WGS84,
    )
    return SeismicSurvey(
        data=np.zeros((len(ids), 100)),
        sample_interval=0.01,
        geometry=geometry,
        trace_map=pd.DataFrame({"receiver_id": ids, "channel": "Z"}),
        start_time=pd.Timestamp("2026-10-01 15:30", tz="UTC"),
    )


def test_circle_gives_the_three_spac_separations():
    template = P.array_template("circle", 7.0)
    survey = survey_at(template.east, template.north)
    distances = np.sort(P.array_layout(survey).separations.distance.to_numpy())
    np.testing.assert_allclose(distances[:4], 7.0, atol=0.01)
    np.testing.assert_allclose(distances[4:8], 7.0 * np.sqrt(2), atol=0.01)
    np.testing.assert_allclose(distances[8:], 14.0, atol=0.01)


def test_triangle_and_l_shapes():
    triangle = P.array_template("triangle", 10.0)
    assert len(triangle) == 4
    np.testing.assert_allclose(triangle.radius[1:], 10.0)
    corners = triangle[["east", "north"]].to_numpy()[1:]
    sides = [np.hypot(*(corners[i] - corners[(i + 1) % 3])) for i in range(3)]
    np.testing.assert_allclose(sides, 10.0 * np.sqrt(3))

    ell = P.array_template("L", 20.0, n_nodes=7)
    np.testing.assert_allclose(sorted(ell.radius), [0, 20 / 3, 20 / 3, 40 / 3, 40 / 3, 20, 20])
    np.testing.assert_allclose(ell.east[1:4], 0.0, atol=1e-9)      # first arm north
    np.testing.assert_allclose(ell.north[4:], 0.0, atol=1e-9)      # second arm east


def test_rotation_turns_clockwise():
    t = P.array_template("circle", 5.0, rotation=90.0)
    assert t.east[1] == pytest.approx(5.0) and t.north[1] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    "shape, n_nodes, match",
    [("circle", 3, "at least 3 ring"), ("triangle", 5, "4 nodes"),
     ("L", 4, "odd number"), ("star", None, "unknown shape")],
)
def test_bad_templates_are_refused(shape, n_nodes, match):
    with pytest.raises(ValueError, match=match):
        P.array_template(shape, 5.0, n_nodes=n_nodes)


def test_fit_recovers_assignment_rotation_and_shift_from_noisy_gps():
    truth = P.array_template("circle", 7.0, rotation=33.0)
    rng = np.random.default_rng(1)
    order = [3, 0, 4, 1, 2]                    # node i sits in slot order[i]
    east = truth.east.to_numpy()[order] + 12.0 + rng.normal(0, 1.0, 5)
    north = truth.north.to_numpy()[order] - 4.0 + rng.normal(0, 1.0, 5)
    survey = survey_at(east, north)

    fit = P.fit_template(survey, P.array_template("circle", 7.0))
    placed = fit.table.set_index("node")
    assert placed.loc["n1", "slot"] == 0                     # n1 got the centre slot
    # The ring is four-fold symmetric, so slot numbers are only defined up to
    # a turn; what matters is that every node lands next to where it truly was.
    # Fitted positions are about the GPS centroid; so is the noise-free truth.
    for i in range(5):
        true_e = truth.east[order[i]] + 12.0 - east.mean()
        true_n = truth.north[order[i]] - 4.0 - north.mean()
        miss = np.hypot(placed.loc[f"n{i}", "east"] - true_e,
                        placed.loc[f"n{i}", "north"] - true_n)
        assert miss < 1.5
    # A 4-fold ring: 33 deg and 33 + 90k deg are the same layout.
    assert min(abs((fit.rotation - 33.0 + 45.0) % 90.0 - 45.0), 90) < 5.0
    assert fit.rms < 1.5

    # The positions handed back have the template's exact geometry.
    snapped = survey_at([0] * 5, [0] * 5)
    snapped.geometry = Geometry(
        ids=list(fit.positions),
        x=[p[0] for p in fit.positions.values()],
        y=[p[1] for p in fit.positions.values()],
        z=[p[2] for p in fit.positions.values()],
        roles=["receiver"] * 5,
        spatial_ref=WGS84,
    )
    distances = np.sort(P.array_layout(snapped).separations.distance.to_numpy())
    np.testing.assert_allclose(distances[:4], 7.0, atol=0.01)
    np.testing.assert_allclose(distances[8:], 14.0, atol=0.01)


def test_fit_pins_centre_and_rotation():
    truth = P.array_template("circle", 7.0)
    survey = survey_at(truth.east, truth.north, ids=["a", "b", "c", "d", "e"])
    fit = P.fit_template(survey, truth, centre="c", rotation=0.0)
    assert fit.table.loc[0, "node"] == "c"
    assert fit.rotation == 0.0
    # "c" really sits on the ring, so forcing it to the centre costs metres.
    assert fit.table.residual.max() > 3.0


def test_fit_can_mirror_an_l():
    mirrored = P.array_template("L", 10.0)
    survey = survey_at(-mirrored.east, mirrored.north)     # second arm west
    fit = P.fit_template(survey, P.array_template("L", 10.0))
    assert fit.mirrored
    assert fit.rms < 1e-3


def test_fit_refuses_a_node_count_mismatch():
    survey = survey_at([0, 1, 2, 3], [0, 1, 2, 3])
    with pytest.raises(ValueError, match="4 nodes in the survey but 5"):
        P.fit_template(survey, P.array_template("circle", 7.0))
