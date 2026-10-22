"""Standard passive-array shapes, and fitting one to a deployment's GPS.

The nodes' own GPS scatters by 1-3 m per fix, which on a 5-7 m separation is
a 20-50 % error in the one number SPAC divides by. The usual cure is a tape,
but nobody tapes five nodes to the centimetre and writes down sixty
coordinates; what actually gets recorded in a field book is "circle, 7 m,
101529 in the middle".

:func:`array_template` turns that sentence into coordinates, and
:func:`fit_template` lines the template up with the GPS. GPS cannot place a
node to a metre, but it can easily tell which node went in the middle and
which way the ring was turned, so the fit chooses the node-to-station
assignment, the rotation and the position of the array, and then hands back
the template's exact geometry as ``positions=`` for
:func:`~shallowgeo.drivers.atom_atm.read_atm_deployment`. The per-node
residual it reports is worth reading: a residual well above the GPS scatter
means either that node was not where the field book says, or the template
is wrong.

The shapes, in metres relative to the centre node (or the corner of the
L):

``"circle"``
    One node at the centre and the rest evenly around a ring of radius
    ``scale``. With five nodes, pairs at :math:`R`, :math:`R\\sqrt{2}` and
    :math:`2R` -- three distinct separations, the minimum the Bessel fit
    solves velocity and coherent fraction from. The standard SPAC layout.
``"triangle"``
    One node at the centre and three at the corners of an equilateral
    triangle whose corners are ``scale`` from the centre. Pairs at
    :math:`R` and :math:`R\\sqrt{3}`. Four nodes only: the classic minimum
    SPAC array.
``"L"``
    A corner node and two perpendicular arms of length ``scale``, the rest
    of the nodes spaced evenly along them, the first arm pointing north
    before rotation. An odd number of nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any

import numpy as np
import pandas as pd

from ..core.survey import SeismicSurvey

METRE_PER_DEGREE = 111320.0
SHAPES = ("circle", "triangle", "L")


def array_template(
    shape: str,
    scale: float,
    *,
    n_nodes: int | None = None,
    rotation: float = 0.0,
) -> pd.DataFrame:
    """Station coordinates of a standard array shape, in local metres.

    ``scale`` is the ring radius (``"circle"``), the centre-to-corner
    distance (``"triangle"``) or the arm length (``"L"``). ``rotation`` turns
    the whole array clockwise, in degrees; at zero the first ring station
    (or the first arm) points north.

    Returns one row per station: ``slot`` (0 is the centre or the corner),
    ``east``, ``north``, ``radius`` and ``azimuth`` from the reference point.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if shape == "circle":
        n = 5 if n_nodes is None else n_nodes
        if n < 4:
            raise ValueError("a circle needs a centre and at least 3 ring nodes")
        azimuths = np.arange(n - 1) * 360.0 / (n - 1)
        radii = np.full(n - 1, float(scale))
    elif shape == "triangle":
        n = 4 if n_nodes is None else n_nodes
        if n != 4:
            raise ValueError("a triangle is a centre and three corners: 4 nodes")
        azimuths = np.array([0.0, 120.0, 240.0])
        radii = np.full(3, float(scale))
    elif shape == "L":
        n = 5 if n_nodes is None else n_nodes
        if n < 3 or n % 2 == 0:
            raise ValueError("an L needs a corner plus the same number on each arm: "
                             "an odd number of nodes, at least 3")
        per_arm = (n - 1) // 2
        steps = np.arange(1, per_arm + 1) * scale / per_arm
        azimuths = np.r_[np.zeros(per_arm), np.full(per_arm, 90.0)]
        radii = np.r_[steps, steps]
    else:
        raise ValueError(f"unknown shape {shape!r}; choose from {SHAPES}")

    azimuths = (azimuths + rotation) % 360.0
    east = np.r_[0.0, radii * np.sin(np.radians(azimuths))]
    north = np.r_[0.0, radii * np.cos(np.radians(azimuths))]
    frame = pd.DataFrame(
        {
            "slot": np.arange(n),
            "east": east,
            "north": north,
            "radius": np.r_[0.0, radii],
            "azimuth": np.r_[0.0, azimuths],
        }
    )
    frame.attrs.update(shape=shape, scale=float(scale), rotation=float(rotation))
    return frame


@dataclass
class TemplateFit:
    """A template placed on a deployment.

    ``table`` has one row per node: the ``slot`` it was assigned, its
    template position (``east``, ``north``, local metres about the GPS
    centroid), its median GPS position (``gps_east``, ``gps_north``) and the
    distance between the two (``residual``). ``positions`` is ready to pass
    to :func:`~shallowgeo.drivers.atom_atm.read_atm_deployment`.
    """

    table: pd.DataFrame
    rotation: float
    mirrored: bool
    rms: float
    positions: dict[str, tuple[float, float, float]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        worst = self.table.loc[self.table.residual.idxmax()]
        return (
            f"TemplateFit({self.metadata.get('shape')} {self.metadata.get('scale'):g} m, "
            f"rotation {self.rotation:.0f} deg{', mirrored' if self.mirrored else ''}, "
            f"GPS rms {self.rms:.1f} m, worst {worst.node} {worst.residual:.1f} m)"
        )


def _fit_rigid(source: np.ndarray, target: np.ndarray, allow_mirror: bool):
    """Rotation (and optionally a reflection) plus translation that best maps
    ``source`` onto ``target``, by the 2-D Kabsch solution."""
    s0, t0 = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - s0).T @ (target - t0))
    rotation = u @ vt
    mirrored = bool(np.linalg.det(rotation) < 0)
    if mirrored and not allow_mirror:
        u[:, -1] *= -1
        rotation = u @ vt
        mirrored = False
    placed = (source - s0) @ rotation + t0
    return placed, rotation, mirrored


def fit_template(
    survey: SeismicSurvey,
    template: pd.DataFrame,
    *,
    centre: str | None = None,
    rotation: float | None = None,
) -> TemplateFit:
    """Place ``template`` on the survey's GPS positions.

    Every assignment of nodes to template stations is tried, each with the
    rotation and shift that fit the GPS best, and the assignment with the
    smallest RMS misfit wins. The template's shape and size are kept exactly;
    only where it sits, how it is turned, and which node is where, come from
    the GPS. An ``"L"`` may also be mirrored, since which arm went which way
    is exactly what a field book forgets.

    ``centre`` pins a node to slot 0 (the centre, or the corner of an L).
    ``rotation`` pins the orientation, in degrees clockwise as in
    :func:`array_template`, leaving only the shift to fit; use it when the
    first ring node was set out on a compass bearing.
    """
    geometry = survey.geometry.table.set_index("id")
    nodes = [n for n in survey.trace_map["receiver_id"] if n in geometry.index]
    if len(nodes) != len(template):
        raise ValueError(
            f"{len(nodes)} nodes in the survey but {len(template)} stations in the "
            "template; drop the extra nodes with trim(nodes=...) or pass n_nodes"
        )
    if centre is not None and centre not in nodes:
        raise ValueError(f"centre node {centre!r} is not in the survey ({nodes})")
    if not survey.geometry.spatial_ref.is_projected:
        lon, lat = geometry.loc[nodes, "x"].to_numpy(float), geometry.loc[nodes, "y"].to_numpy(float)
        lat0, lon0 = float(lat.mean()), float(lon.mean())
        coslat = np.cos(np.radians(lat0))
        gps = np.c_[(lon - lon0) * METRE_PER_DEGREE * coslat, (lat - lat0) * METRE_PER_DEGREE]
    else:
        raise ValueError("fit_template expects geographic (longitude, latitude) positions")
    elevation = geometry.loc[nodes, "z"].to_numpy(float)

    shape = template.attrs.get("shape", "")
    stations = template[["east", "north"]].to_numpy(float)
    base = template.attrs.get("rotation", 0.0)
    if rotation is not None:
        # Turn the template to the pinned bearing; only the shift is fitted.
        turn = np.radians(rotation - base)
        clockwise = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
        stations = stations @ clockwise

    best = None
    for order in permutations(range(len(nodes))):
        # order[k] is the index of the node placed in slot k
        if centre is not None and nodes[order[0]] != centre:
            continue
        target = gps[list(order)]
        if rotation is None:
            placed, matrix, mirrored = _fit_rigid(stations, target, shape == "L")
        else:
            placed = stations - stations.mean(axis=0) + target.mean(axis=0)
            matrix, mirrored = np.eye(2), False
        rms = float(np.sqrt(np.mean(np.sum((placed - target) ** 2, axis=1))))
        if best is None or rms < best[0] - 1e-9:
            best = (rms, order, placed, matrix, mirrored)

    rms, order, placed, matrix, mirrored = best
    if rotation is None:
        # Where the template's north axis ended up, clockwise from north.
        east_n, north_n = np.array([0.0, 1.0]) @ matrix
        total_rotation = (base + np.degrees(np.arctan2(east_n, north_n))) % 360.0
    else:
        total_rotation = float(rotation) % 360.0

    rows, positions = [], {}
    for slot, i in enumerate(order):
        node = nodes[i]
        east, north = placed[slot]
        rows.append(
            {
                "node": node,
                "slot": slot,
                "east": float(east),
                "north": float(north),
                "gps_east": float(gps[i, 0]),
                "gps_north": float(gps[i, 1]),
                "residual": float(np.hypot(east - gps[i, 0], north - gps[i, 1])),
            }
        )
        positions[node] = (
            lon0 + east / (METRE_PER_DEGREE * coslat),
            lat0 + north / METRE_PER_DEGREE,
            float(elevation[i]),
        )
    table = pd.DataFrame(rows).sort_values("slot").reset_index(drop=True)
    return TemplateFit(
        table=table,
        rotation=total_rotation,
        mirrored=mirrored,
        rms=rms,
        positions=positions,
        metadata={"shape": shape, "scale": template.attrs.get("scale")},
    )
