"""Seismic refraction tomography (SRT): a velocity image from first breaks.

The intercept-time method in :mod:`~shallowgeo.refraction.layers` assumes
flat layers of constant velocity and asks the data for a handful of numbers.
Tomography drops the layers: the ground is a grid of velocities, every
first-break pick is a ray through it, and the velocities are adjusted until
the rays' travel times match the picks. It needs many shots into the same
spread -- a shot every one or two geophones -- because each grid node has to
be crossed by rays from several directions before its velocity means
anything.

Forward problem
---------------
Travel times are computed by the **shortest-path method** (Moser, 1991):
the grid nodes are joined into a network, each link costs its length times
the average slowness along it, and Dijkstra's algorithm finds the quickest
route from each source to every node. By Fermat's principle that route is
the ray and its cost is the first-arrival time. Links reach up to
``radius`` nodes away in every direction whose step is not a multiple of a
shorter one, so rays can leave a node at many angles; with the default
radius of 4 the travel-time error from the finite angle set is well under
1 %. It needs nothing beyond SciPy, runs anywhere a student has Python, and
the rays it returns are what the Jacobian is built from.

Inverse problem
---------------
The model is the logarithm of slowness at every grid node, which keeps
velocities positive without clipping. Each Gauss-Newton step solves

    minimise  || (t_obs - t(m)) / sigma ||^2  +  lam || R m ||^2

where ``sigma`` is the pick error and ``R`` takes differences between
neighbouring nodes, with vertical differences weighted by ``z_weight`` --
below 1 lets velocity change faster with depth than along the line, which is
how the ground usually is. ``lam`` trades fit against smoothness and is the
one number a student has to choose; :func:`lambda_sweep` draws the trade-off.
Iterations stop when the misfit reaches the noise level (chi-squared of
about 1) or stops improving.

A pyGIMLi backend (``backend="pygimli"``) runs the same data through
``TravelTimeManager`` on an unstructured mesh, as ADR-001 plans for the
joint-inversion work, and returns the result on the same regular grid so the
two can be compared directly. pyGIMLi is optional; install it from conda
(``pixi run -e full ...``) or ``pip install pygimli`` on Linux and Colab.

What tomography cannot do
-------------------------
Below the deepest ray there is no information at all, only the smoothness
constraint filling in: every plot here shades nodes by ray coverage so the
unconstrained part of the image is obvious. A velocity decrease with depth
is invisible to first arrivals in any method. And the image is smooth by
construction, so a sharp interface comes back as a gradient -- the
intercept-time depth is the better estimate of *where* a boundary is,
tomography of *how it varies* along the line.

Assumptions: 2-D, isotropic, source and receivers on a straight line. The
surface is flat unless ``elevation`` is given to :func:`make_grid`, in which
case nodes above the ground surface are removed from the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import lsqr

__all__ = [
    "TomographyGrid",
    "TomographyResult",
    "make_grid",
    "gradient_velocity",
    "estimate_gradient",
    "forward",
    "synthetic_traveltimes",
    "reciprocity",
    "traveltime_tomography",
    "refine_picks",
    "lambda_sweep",
    "MIN_SHOTS",
]

MIN_SHOTS = 5


# ---------------------------------------------------------------------------
# Grid and ray network
# ---------------------------------------------------------------------------


@dataclass
class TomographyGrid:
    """A regular 2-D grid of nodes: ``x`` along the line, ``z`` depth (down).

    Velocities live on the nodes, as arrays of shape ``(nz, nx)``. Build one
    with :func:`make_grid`, which sizes it to the data.

    ``surface`` is the depth of the ground surface below the top of the grid
    at each ``x`` (zero for a flat line); nodes above it are *air* and are
    left out of the ray network.
    """

    x: np.ndarray
    z: np.ndarray
    surface: np.ndarray | None = None
    radius: int = 4
    _network: dict | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.z = np.asarray(self.z, dtype=float)
        if self.x.ndim != 1 or self.z.ndim != 1 or self.x.size < 2 or self.z.size < 2:
            raise ValueError("x and z must be 1-D with at least two nodes each")
        for name, v in (("x", self.x), ("z", self.z)):
            steps = np.diff(v)
            if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-6):
                raise ValueError(f"{name} nodes must be evenly spaced and increasing")
        if not np.isclose(self.dx, self.dz, rtol=1e-6):
            raise ValueError("the grid must be square (dx == dz) for the ray network")
        if self.surface is None:
            self.surface = np.full(self.x.size, self.z[0])
        self.surface = np.asarray(self.surface, dtype=float)
        if self.surface.shape != self.x.shape:
            raise ValueError("surface needs one depth per x node")

    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0])

    @property
    def dz(self) -> float:
        return float(self.z[1] - self.z[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.z.size, self.x.size)

    @property
    def n_nodes(self) -> int:
        return self.x.size * self.z.size

    @property
    def active(self) -> np.ndarray:
        """``(nz, nx)`` bool: nodes at or below the ground surface."""
        return self.z[:, None] >= self.surface[None, :] - 1e-9 * self.dz

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """``X, Z`` node coordinates, each ``(nz, nx)``."""
        return np.meshgrid(self.x, self.z)

    def surface_node(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nearest surface node to each along-line position, and the snap distance."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        i = np.clip(np.round((x - self.x[0]) / self.dx).astype(int), 0, self.x.size - 1)
        j = np.clip(np.round((self.surface[i] - self.z[0]) / self.dz).astype(int),
                    0, self.z.size - 1)
        return j * self.x.size + i, np.abs(self.x[i] - x)

    # -- the shortest-path network ------------------------------------------

    def network(self) -> dict:
        """Links of the ray network and how each samples the node slowness.

        Built once per grid. ``W`` is a sparse (links x nodes) matrix whose
        row ``k`` holds the weights that turn node slownesses into the travel
        time along link ``k``: the link is sampled at evenly spaced points,
        each point bilinearly interpolated from its four surrounding nodes.
        A travel time is then a sum of rows of ``W`` times slowness, which is
        what makes the Jacobian exact for this discretisation.
        """
        if self._network is not None:
            return self._network
        nz, nx = self.shape
        r = int(self.radius)
        if r < 1:
            raise ValueError("radius must be at least 1")
        steps = [(di, dj) for di in range(-r, r + 1) for dj in range(r + 1)
                 if (dj > 0 or di > 0) and gcd(abs(di), dj) == 1]
        active = self.active.ravel()
        a_all, b_all, w_rows, w_cols, w_vals, lengths = [], [], [], [], [], []
        n_links = 0
        jj, ii = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
        for di, dj in steps:
            ok = (ii + di >= 0) & (ii + di < nx) & (jj + dj < nz)
            i0, j0 = ii[ok], jj[ok]
            a = j0 * nx + i0
            b = (j0 + dj) * nx + (i0 + di)
            keep = active[a] & active[b]
            a, b, i0, j0 = a[keep], b[keep], i0[keep], j0[keep]
            m = a.size
            if m == 0:
                continue
            length = self.dx * np.hypot(di, dj)
            n_samp = 2 * max(abs(di), dj)
            f = (np.arange(n_samp) + 0.5) / n_samp
            for fk in f:
                # Point along the link, in fractional node units.
                pi = i0 + fk * di
                pj = j0 + fk * dj
                ib = np.minimum(np.floor(pi).astype(int), nx - 2)
                jb = np.minimum(np.floor(pj).astype(int), nz - 2)
                u, v = pi - ib, pj - jb
                for dii, djj, wt in ((0, 0, (1 - u) * (1 - v)), (1, 0, u * (1 - v)),
                                     (0, 1, (1 - u) * v), (1, 1, u * v)):
                    w_rows.append(np.arange(n_links, n_links + m))
                    w_cols.append((jb + djj) * nx + (ib + dii))
                    w_vals.append(wt * length / n_samp)
            a_all.append(a)
            b_all.append(b)
            lengths.append(np.full(m, length))
            n_links += m
        a = np.concatenate(a_all)
        b = np.concatenate(b_all)
        W = sparse.csr_matrix(
            (np.concatenate(w_vals), (np.concatenate(w_rows), np.concatenate(w_cols))),
            shape=(n_links, self.n_nodes),
        )
        W.sum_duplicates()
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        key = lo.astype(np.int64) * self.n_nodes + hi
        order = np.argsort(key)
        self._network = {
            "a": a, "b": b, "W": W, "length": np.concatenate(lengths),
            "key_sorted": key[order], "key_order": order,
        }
        return self._network


def make_grid(
    table: pd.DataFrame | None = None,
    *,
    x: tuple[float, float] | None = None,
    depth: float | None = None,
    dx: float | None = None,
    nodes_per_spacing: int = 2,
    radius: int = 4,
    elevation: pd.DataFrame | tuple | None = None,
) -> TomographyGrid:
    """A grid sized to a travel-time table.

    Parameters
    ----------
    table
        Travel-time table with ``source_x`` and ``receiver_x`` (metres).
        Sets the horizontal extent (every source and receiver, plus one
        node) and the defaults below.
    x
        ``(xmin, xmax)`` to override the extent.
    depth
        How deep to make the grid. Default: a third of the longest offset,
        the usual rule of thumb for how deep first arrivals see. Rays will
        not go much deeper than the data allow however deep the grid is;
        a deeper grid only costs time.
    dx
        Node spacing. Default: the geophone spacing divided by
        ``nodes_per_spacing``, so that every geophone sits on a node.
    radius
        Reach of the ray network in nodes; see the module notes.
    elevation
        Optional ground surface for a line with topography: a DataFrame with
        ``x`` and ``elevation`` columns, or a tuple ``(x, elevation)``.
        Depths in the grid are then measured down from the highest point.
    """
    if table is not None:
        xs = np.concatenate([table["source_x"].to_numpy(float),
                             table["receiver_x"].to_numpy(float)])
        rx = np.unique(np.round(table["receiver_x"].to_numpy(float), 4))
        spacing = float(np.median(np.diff(rx))) if rx.size > 1 else 1.0
        max_offset = float(np.abs(table["receiver_x"] - table["source_x"]).max())
    elif x is None or depth is None or dx is None:
        raise ValueError("without a table, give x, depth and dx")
    if dx is None:
        dx = spacing / nodes_per_spacing
    if x is None:
        lo, hi = xs.min(), xs.max()
        # Align the grid on the receivers so that geophones fall on nodes.
        anchor = rx[0] if table is not None else lo
        x0 = anchor - np.ceil((anchor - lo) / dx + 1e-9) * dx - dx
        x1 = anchor + np.ceil((hi - anchor) / dx - 1e-9) * dx + dx
    else:
        x0, x1 = x
    if depth is None:
        depth = max_offset / 3
    nx = int(round((x1 - x0) / dx)) + 1
    xg = x0 + dx * np.arange(nx)

    surface = None
    if elevation is not None:
        if isinstance(elevation, pd.DataFrame):
            ex, ez = elevation["x"].to_numpy(float), elevation["elevation"].to_numpy(float)
        else:
            ex, ez = (np.asarray(v, dtype=float) for v in elevation)
        order = np.argsort(ex)
        elev = np.interp(xg, ex[order], ez[order])
        top = elev.max()
        surface = np.round((top - elev) / dx) * dx
        depth = depth + surface.max()
    nz = int(np.ceil(depth / dx)) + 1
    zg = dx * np.arange(nz)
    return TomographyGrid(xg, zg, surface=surface, radius=radius)


def gradient_velocity(grid: TomographyGrid, v_top: float, v_bottom: float) -> np.ndarray:
    """Velocity increasing linearly with depth below the ground surface.

    The usual starting model for refraction tomography: it has no structure
    of its own, so whatever structure the inversion puts in came from the
    data.
    """
    depth_below = grid.z[:, None] - grid.surface[None, :]
    frac = np.clip(depth_below / max(grid.z[-1] - grid.surface.min(), grid.dz), 0, 1)
    return v_top + (v_bottom - v_top) * frac


def estimate_gradient(table: pd.DataFrame) -> tuple[float, float]:
    """``(v_top, v_bottom)`` for a starting gradient, read off the picks.

    ``v_top`` is the median apparent velocity (offset over time) of the
    nearest tenth of the picks -- close to the direct wave. ``v_bottom`` is
    the median over the farthest tenth, which on a refraction line is close
    to the fastest refractor the spread sees. Both are rough on purpose: a
    starting model should be simple and in the right range, not right.
    """
    off = np.abs(table["receiver_x"] - table["source_x"]).to_numpy(float)
    t = table["time"].to_numpy(float)
    ok = (off > 0) & (t > 0) & np.isfinite(t)
    off, t = off[ok], t[ok]
    n = max(3, off.size // 10)
    order = np.argsort(off)
    v_top = float(np.median(off[order[:n]] / t[order[:n]]))
    far = order[-n:]
    v_far = float(np.median(off[far] / t[far]))
    return v_top, max(v_far * 1.1, 1.5 * v_top)


# ---------------------------------------------------------------------------
# Forward modelling
# ---------------------------------------------------------------------------


def _link_ids(net: dict, a: np.ndarray, b: np.ndarray, n_nodes: int) -> np.ndarray:
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo.astype(np.int64) * n_nodes + hi
    pos = np.searchsorted(net["key_sorted"], key)
    return net["key_order"][pos]


def forward(
    grid: TomographyGrid,
    velocity: np.ndarray,
    source_x,
    receiver_x,
    *,
    jacobian: bool = True,
    rays: bool = False,
) -> dict[str, Any]:
    """First-arrival travel times through a velocity grid.

    Parameters
    ----------
    velocity
        ``(nz, nx)`` node velocities, m/s.
    source_x, receiver_x
        Along-line positions of each source-receiver pair (equal length).
        Each is snapped to the nearest surface node.

    Returns
    -------
    dict with ``time`` (s, one per pair), ``J`` (sparse, pairs x nodes: the
    derivative of each time with respect to each node's slowness, i.e. the
    ray length attributed to that node) when ``jacobian``, ``rays`` (list of
    ``(x, z)`` arrays) when ``rays``, and ``snap`` (m, how far each source
    or receiver was moved to reach a node).
    """
    velocity = np.asarray(velocity, dtype=float)
    if velocity.shape != grid.shape:
        raise ValueError(f"velocity must have shape {grid.shape}, got {velocity.shape}")
    net = grid.network()
    slowness = (1.0 / np.where(grid.active, velocity, np.inf)).ravel()
    slowness[~grid.active.ravel()] = 0.0
    cost = net["W"] @ slowness
    G = sparse.csr_matrix((cost, (net["a"], net["b"])), shape=(grid.n_nodes,) * 2)

    src_node, src_snap = grid.surface_node(source_x)
    rec_node, rec_snap = grid.surface_node(receiver_x)
    uniq, inv = np.unique(src_node, return_inverse=True)
    dist, pred = dijkstra(G, directed=False, indices=uniq, return_predecessors=True)
    time = dist[inv, rec_node]

    out: dict[str, Any] = {"time": time, "snap": np.maximum(src_snap, rec_snap)}
    if not (jacobian or rays):
        return out
    rows, cols, paths = [], [], []
    X, Z = grid.mesh()
    Xf, Zf = X.ravel(), Z.ravel()
    for k in range(time.size):
        p = pred[inv[k]]
        node = rec_node[k]
        nodes = [node]
        while node != src_node[k] and node >= 0:
            node = p[node]
            nodes.append(node)
        nodes = np.asarray(nodes)
        if nodes[-1] < 0:  # unreachable
            continue
        if nodes.size > 1:
            links = _link_ids(net, nodes[:-1], nodes[1:], grid.n_nodes)
            rows.append(np.full(links.size, k))
            cols.append(links)
        if rays:
            paths.append(np.column_stack([Xf[nodes[::-1]], Zf[nodes[::-1]]]))
    if jacobian:
        n_links = net["W"].shape[0]
        S = sparse.csr_matrix(
            (np.ones(sum(r.size for r in rows)),
             (np.concatenate(rows) if rows else [], np.concatenate(cols) if cols else [])),
            shape=(time.size, n_links),
        )
        out["J"] = (S @ net["W"]).tocsr()
    if rays:
        out["rays"] = paths
    return out


def synthetic_traveltimes(
    grid: TomographyGrid,
    velocity: np.ndarray,
    source_x,
    receiver_x,
    *,
    noise: float = 0.0,
    seed: int | None = 0,
) -> pd.DataFrame:
    """A travel-time table computed through a known model -- for testing.

    Every source is paired with every receiver (except where they coincide),
    and Gaussian noise of standard deviation ``noise`` seconds is added. The
    result has the same columns as a picked table, so it goes straight into
    :func:`traveltime_tomography`. Inverting data from a model you chose is
    the only way to see what the method can and cannot recover.
    """
    sx = np.atleast_1d(np.asarray(source_x, dtype=float))
    rx = np.atleast_1d(np.asarray(receiver_x, dtype=float))
    S, R = np.meshgrid(sx, rx, indexing="ij")
    keep = ~np.isclose(S, R)
    s, r = S[keep], R[keep]
    t = forward(grid, velocity, s, r, jacobian=False)["time"]
    rng = np.random.default_rng(seed)
    if noise:
        t = t + rng.normal(0, noise, t.size)
    shot = np.searchsorted(sx, s) if np.all(np.diff(sx) > 0) else None
    return pd.DataFrame({
        "shot": [f"syn{k + 1:02d}" for k in (shot if shot is not None
                                              else np.unique(s, return_inverse=True)[1])],
        "source_x": s,
        "receiver_x": r,
        "offset": np.abs(r - s),
        "time": t,
    })


# ---------------------------------------------------------------------------
# Quality control before inverting
# ---------------------------------------------------------------------------


def reciprocity(table: pd.DataFrame, *, tol: float = 0.25) -> pd.DataFrame:
    """Reciprocal pairs in a travel-time table, and how well they agree.

    A wave takes the same time from A to B as from B to A. When a shot was
    fired at a geophone position and another geophone was a source too,
    the two picks for that pair should be equal; the difference is a direct
    measurement of picking error, free of any model. Tomography cannot fit
    the data better than this, so it sets the error level to invert to.

    Positions within ``tol`` metres are treated as the same place. Returns
    one row per reciprocal pair with both times and ``dt = t_ab - t_ba``;
    ``dt.attrs`` carries nothing -- summarise with
    ``pairs["dt"].abs().median()``. A systematic sign per shot means a
    trigger delay on that shot, not a picking error.
    """
    sx = table["source_x"].to_numpy(float)
    rx = table["receiver_x"].to_numpy(float)
    t = table["time"].to_numpy(float)
    shot = table["shot"].to_numpy() if "shot" in table else np.full(t.size, "")
    def q(v):
        return np.round(v / tol).astype(np.int64)

    fwd = pd.DataFrame({"a": q(sx), "b": q(rx), "i": np.arange(t.size)})
    rev = pd.DataFrame({"a": q(rx), "b": q(sx), "j": np.arange(t.size)})
    m = fwd.merge(rev, on=["a", "b"])
    m = m[m["i"] < m["j"]]
    m = m[sx[m["i"]] != rx[m["i"]]]
    i, j = m["i"].to_numpy(), m["j"].to_numpy()
    return pd.DataFrame({
        "shot_ab": shot[i], "shot_ba": shot[j],
        "x_a": sx[i], "x_b": rx[i],
        "offset": np.abs(rx[i] - sx[i]),
        "t_ab": t[i], "t_ba": t[j], "dt": t[i] - t[j],
    }).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Inversion
# ---------------------------------------------------------------------------


def _roughness(grid: TomographyGrid, z_weight: float) -> sparse.csr_matrix:
    """First differences between neighbouring active nodes, vertical ones scaled."""
    nz, nx = grid.shape
    idx = np.arange(grid.n_nodes).reshape(nz, nx)
    act = grid.active
    blocks = []
    for (a, b), w in (((idx[:, :-1], idx[:, 1:]), 1.0), ((idx[:-1, :], idx[1:, :]), z_weight)):
        ok = act.ravel()[a.ravel()] & act.ravel()[b.ravel()]
        a, b = a.ravel()[ok], b.ravel()[ok]
        m = a.size
        rows = np.repeat(np.arange(m), 2)
        cols = np.column_stack([a, b]).ravel()
        vals = np.tile([-w, w], m)
        blocks.append(sparse.csr_matrix((vals, (rows, cols)), shape=(m, grid.n_nodes)))
    return sparse.vstack(blocks).tocsr()


@dataclass
class TomographyResult:
    """What :func:`traveltime_tomography` returns.

    ``velocity`` and ``start`` are ``(nz, nx)`` on ``grid``; ``coverage`` is
    the total ray length through each node's neighbourhood (metres), zero
    where no ray went. ``table`` is the input table with ``predicted`` and
    ``residual`` columns (seconds). ``history`` has one row per iteration.
    """

    grid: TomographyGrid
    velocity: np.ndarray
    start: np.ndarray
    coverage: np.ndarray
    table: pd.DataFrame
    history: pd.DataFrame
    settings: dict[str, Any]
    rays: list | None = None

    @property
    def chi2(self) -> float:
        return float(self.history["chi2"].iloc[-1])

    @property
    def rms(self) -> float:
        """RMS travel-time residual, seconds."""
        return float(np.sqrt(np.mean(self.table["residual"] ** 2)))

    def predict(self, source_x, receiver_x) -> np.ndarray:
        """First-arrival times (s) through the final model for any pairs.

        Used to guide re-picking: predict every trace, including the ones
        that were dropped, and look for the first break near the prediction
        (:meth:`PickingSession.guided_pick`).
        """
        return forward(self.grid, self.velocity, source_x, receiver_x,
                       jacobian=False)["time"]

    def outliers(self, k: float = 3.0, *, floor: float = 0.001) -> pd.DataFrame:
        """Picks whose residual is far outside the rest.

        A pick is flagged when ``|residual|`` exceeds both ``k`` robust
        standard deviations (1.4826 x the median absolute deviation) and
        ``floor`` seconds. On field data these are nearly always mispicks --
        a later phase, or noise -- rather than geology: the model cannot
        bend enough to fit one trace without its neighbours.
        """
        r = self.table["residual"]
        sd = 1.4826 * float(np.median(np.abs(r - r.median())))
        bad = (r.abs() > k * sd) & (r.abs() > floor)
        return self.table[bad]

    def covered(self, min_coverage: float | None = None) -> np.ndarray:
        """Nodes with enough ray coverage to believe, as a bool ``(nz, nx)``.

        The default threshold is one grid spacing of ray length -- at least
        one ray passes through the node's cell.
        """
        thresh = self.grid.dx if min_coverage is None else min_coverage
        return self.coverage >= thresh

    def depth_of_investigation(self) -> np.ndarray:
        """Deepest covered node below the ground surface at each x (NaN if none)."""
        cov = self.covered()
        zz = np.where(cov, self.grid.z[:, None] - self.grid.surface[None, :], np.nan)
        out = np.full(zz.shape[1], np.nan)
        have = np.isfinite(zz).any(axis=0)
        out[have] = np.nanmax(zz[:, have], axis=0)
        return out

    def profile(self, x: float) -> pd.DataFrame:
        """Velocity against depth at one position, with the coverage there."""
        i = int(np.argmin(np.abs(self.grid.x - x)))
        return pd.DataFrame({
            "depth": self.grid.z - self.grid.surface[i],
            "velocity": self.velocity[:, i],
            "start": self.start[:, i],
            "coverage": self.coverage[:, i],
        })

    def summary(self) -> pd.Series:
        return pd.Series({
            "backend": self.settings.get("backend"),
            "picks": len(self.table),
            "shots": self.table["shot"].nunique() if "shot" in self.table else np.nan,
            "iterations": int(self.history["iteration"].iloc[-1]),
            "lam": self.settings.get("lam"),
            "z_weight": self.settings.get("z_weight"),
            "chi2": self.chi2,
            "rms_ms": 1e3 * self.rms,
            "v_min": float(np.nanmin(np.where(self.covered(), self.velocity, np.nan))),
            "v_max": float(np.nanmax(np.where(self.covered(), self.velocity, np.nan))),
            "max_depth_m": float(np.nanmax(self.depth_of_investigation())),
        })

    # -- plots ---------------------------------------------------------------

    def plot(self, ax=None, *, field: str = "velocity", mask: bool = True,
             rays: bool = False, contours=None, vmin=None, vmax=None,
             cmap: str = "viridis", colorbar: bool = True, sensors: bool = True):
        """The velocity image, faded where no ray has been.

        Parameters
        ----------
        field
            ``"velocity"``, ``"start"``, ``"coverage"`` or ``"change"``
            (percent change from the starting model).
        mask
            Fade the nodes that no ray crosses. Keep this on: the colours
            there are the smoothness constraint, not the ground.
        rays
            Overlay the ray paths through the final model.
        contours
            Velocity values to contour, e.g. ``[1000, 1500, 2000]``.
        """
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        g = self.grid
        if field == "velocity":
            values = self.velocity
        elif field == "start":
            values = self.start
        elif field == "coverage":
            values = self.coverage
        elif field == "change":
            values = 100 * (self.velocity - self.start) / self.start
        else:
            raise ValueError(f"unknown field {field!r}")
        values = np.where(g.active, values, np.nan)
        if field == "change":
            lim = np.nanmax(np.abs(values)) if vmin is None else None
            vmin, vmax = (-lim, lim) if lim is not None else (vmin, vmax)
            cmap = "RdBu_r" if cmap == "viridis" else cmap
        half = g.dx / 2
        extent = (g.x[0] - half, g.x[-1] + half, g.z[-1] + half, g.z[0] - half)
        norm = None
        if field == "coverage":
            from matplotlib.colors import LogNorm

            positive = values[np.isfinite(values) & (values > 0)]
            lo_ = max(float(np.nanmin(positive)) if positive.size else 1e-3, 1e-3 * g.dx)
            norm = LogNorm(vmin=vmin or lo_, vmax=vmax or float(np.nanmax(values)))
            values = np.where(values > 0, values, np.nan)
            vmin = vmax = None
            cmap = "magma" if cmap == "viridis" else cmap
        im = ax.imshow(values, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm,
                       aspect="equal", interpolation="bilinear", origin="upper")
        if mask and field != "coverage":
            # Fade to white in proportion to how far a node falls short of
            # one ray's worth of coverage, so the edge of the constrained
            # region is gradual rather than a staircase of grid cells.
            fade = 1 - np.clip(self.coverage / g.dx, 0, 1)
            rgba = np.zeros(g.shape + (4,))
            rgba[..., :3] = 1.0
            rgba[..., 3] = np.where(g.active, 0.85 * fade, 0.0)
            ax.imshow(rgba, extent=extent, aspect="equal", interpolation="bilinear",
                      origin="upper")
        if contours is not None and field in ("velocity", "start"):
            X, Z = g.mesh()
            cs = ax.contour(X, Z, np.where(self.covered(), values, np.nan), levels=contours,
                            colors="w", linewidths=0.9)
            ax.clabel(cs, fmt="%.0f", fontsize=7)
        if rays and self.rays:
            for p in self.rays:
                ax.plot(p[:, 0], p[:, 1], color="w", lw=0.3, alpha=0.35)
        if sensors:
            rx = np.unique(self.table["receiver_x"])
            sx = np.unique(self.table["source_x"])
            ax.plot(rx, np.interp(rx, g.x, g.surface), "v", color="k", ms=4,
                    clip_on=False, label="geophones")
            ax.plot(sx, np.interp(sx, g.x, g.surface) - g.dx, "*", color="#eb6834",
                    ms=8, mec="k", mew=0.4, clip_on=False, label="shots")
        ax.set_xlabel("position along line (m)")
        ax.set_ylabel("depth (m)")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3] - g.dx * 1.5)
        if colorbar:
            label = {"velocity": "velocity (m/s)", "start": "velocity (m/s)",
                     "coverage": "ray length (m)", "change": "change from start (%)"}[field]
            ax.figure.colorbar(im, ax=ax, label=label, shrink=0.8, pad=0.02)
        return ax

    def plot_fit(self, axes=None):
        """Observed against predicted times, and the residuals by offset and shot."""
        import matplotlib.pyplot as plt

        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        tab = self.table
        a0, a1, a2 = axes
        lim = 1.05e3 * max(tab["time"].max(), tab["predicted"].max())
        a0.plot(1e3 * tab["time"], 1e3 * tab["predicted"], ".", ms=4, color="#256abf")
        a0.plot([0, lim], [0, lim], "k-", lw=0.8)
        a0.set_xlim(0, lim)
        a0.set_ylim(0, lim)
        a0.set_aspect("equal")
        a0.set_xlabel("observed (ms)")
        a0.set_ylabel("predicted (ms)")
        a0.set_title(f"RMS {1e3 * self.rms:.2f} ms, chi² {self.chi2:.2f}", fontsize=10)
        off = np.abs(tab["receiver_x"] - tab["source_x"])
        a1.plot(off, 1e3 * tab["residual"], ".", ms=4, color="#256abf")
        a1.axhline(0, color="k", lw=0.8)
        a1.set_xlabel("offset (m)")
        a1.set_ylabel("observed − predicted (ms)")
        a1.grid(alpha=0.3)
        if "shot" in tab:
            order = tab.groupby("shot")["source_x"].first().sort_values()
            data = [1e3 * tab.loc[tab["shot"] == s, "residual"].to_numpy() for s in order.index]
            a2.boxplot(data, positions=order.to_numpy(), widths=0.4 * max(
                np.min(np.diff(np.unique(order.to_numpy()))) if order.nunique() > 1 else 1, 0.1),
                manage_ticks=False, flierprops={"ms": 3})
            a2.axhline(0, color="k", lw=0.8)
            a2.set_xlabel("source position (m)")
            a2.set_ylabel("residual (ms)")
            a2.set_title("a shot sitting off zero has a timing problem", fontsize=9)
            a2.grid(alpha=0.3)
        return axes


def _prepare(table: pd.DataFrame, error, rel_error) -> tuple[pd.DataFrame, np.ndarray]:
    need = {"source_x", "receiver_x", "time"}
    missing = need - set(table.columns)
    if missing:
        raise ValueError(f"table is missing column(s) {sorted(missing)}")
    tab = table.copy()
    if "use" in tab:
        tab = tab[tab["use"].astype(bool)]
    tab = tab[tab["time"].notna() & (tab["time"] > 0)
              & ~np.isclose(tab["source_x"], tab["receiver_x"])].reset_index(drop=True)
    if "shot" not in tab:
        tab["shot"] = tab["source_x"].map(lambda v: f"{v:.2f}")
    n_shots = tab["shot"].nunique()
    if n_shots < MIN_SHOTS:
        raise ValueError(
            f"tomography needs rays crossing from several directions: at least "
            f"{MIN_SHOTS} shots into the spread, ideally one every one or two "
            f"geophones. This table has {n_shots}. For sparse data use "
            "shallowgeo.refraction.fit_layers."
        )
    if "error" in tab and error is None:
        sigma = tab["error"].to_numpy(float)
    else:
        sigma = (5e-4 if error is None else error) + rel_error * tab["time"].to_numpy(float)
    return tab, sigma


def traveltime_tomography(
    table: pd.DataFrame,
    *,
    grid: TomographyGrid | None = None,
    start: np.ndarray | tuple[float, float] | None = None,
    lam: float = 30.0,
    z_weight: float = 0.3,
    error: float | None = None,
    rel_error: float = 0.03,
    max_iter: int = 12,
    target_chi2: float = 1.0,
    v_limits: tuple[float, float] = (100.0, 8000.0),
    backend: str = "native",
    keep_rays: bool = True,
    verbose: bool = False,
    **grid_kwargs,
) -> TomographyResult:
    """Invert first-break picks for a 2-D velocity image.

    Parameters
    ----------
    table
        Travel-time table with ``source_x``, ``receiver_x``, ``time`` (s),
        and preferably ``shot``: the output of
        :meth:`PickingSession.table` or :func:`traveltime_table`. Rows with
        ``use`` False are dropped, as are zero-offset traces.
    grid
        From :func:`make_grid`; built from the table if omitted, with any
        extra keyword arguments passed to it (``depth=``, ``dx=`` ...).
    start
        Starting velocity: an ``(nz, nx)`` array, a ``(v_top, v_bottom)``
        gradient, or ``None`` for a gradient from :func:`estimate_gradient`.
    lam
        Regularisation strength. Larger is smoother and fits worse; smaller
        fits better and grows artefacts. Use :func:`lambda_sweep` to choose.
    z_weight
        Weight of vertical relative to horizontal smoothing. Below 1 allows
        sharper change with depth than along the line.
    error, rel_error
        Pick error model, ``sigma = error + rel_error * t`` (seconds).
        ``error`` defaults to 0.5 ms, or to a table ``error`` column when
        there is one. :func:`reciprocity` measures what it should be.
    max_iter, target_chi2
        Stop after ``max_iter`` Gauss-Newton steps, or once the error-
        weighted misfit reaches ``target_chi2``, or when it stops falling.
    v_limits
        Velocities are kept inside these bounds (m/s).
    backend
        ``"native"`` (shortest-path rays, SciPy only) or ``"pygimli"``.

    Returns
    -------
    :class:`TomographyResult`
    """
    tab, sigma = _prepare(table, error, rel_error)
    if grid is None:
        grid = make_grid(tab, **grid_kwargs)
    elif grid_kwargs:
        raise TypeError(f"grid given, so {sorted(grid_kwargs)} cannot be used")
    if start is None:
        start = estimate_gradient(tab)
    if isinstance(start, tuple):
        start_v = gradient_velocity(grid, *start)
    else:
        start_v = np.asarray(start, dtype=float)
    settings = {"backend": backend, "lam": lam, "z_weight": z_weight,
                "error": error, "rel_error": rel_error, "max_iter": max_iter,
                "v_limits": v_limits}

    if backend == "pygimli":
        return _invert_pygimli(tab, sigma, grid, start_v, settings, verbose=verbose)
    if backend != "native":
        raise ValueError(f"backend must be 'native' or 'pygimli', got {backend!r}")

    sx = tab["source_x"].to_numpy(float)
    rx = tab["receiver_x"].to_numpy(float)
    t_obs = tab["time"].to_numpy(float)
    act = grid.active.ravel()
    R = _roughness(grid, z_weight)[:, act]
    Wd = 1.0 / sigma

    lo, hi = np.log(1 / v_limits[1]), np.log(1 / v_limits[0])
    m = np.log(1.0 / start_v.ravel()[act])

    def velocity_of(mv):
        v = np.full(grid.n_nodes, np.nan)
        v[act] = np.exp(-mv)
        return v.reshape(grid.shape)

    def objective(mv):
        fw = forward(grid, velocity_of(mv), sx, rx)
        r = (t_obs - fw["time"]) * Wd
        phi_d = float(r @ r)
        rm = R @ mv
        return fw, phi_d, phi_d + lam * float(rm @ rm)

    fw, phi_d, phi = objective(m)
    rows = [{"iteration": 0, "chi2": phi_d / t_obs.size,
             "rms_ms": 1e3 * np.sqrt(np.mean((t_obs - fw["time"]) ** 2)), "phi": phi}]
    if verbose:
        print(f"start: chi2 {rows[0]['chi2']:.2f}, RMS {rows[0]['rms_ms']:.2f} ms")
    sq = np.sqrt(lam)
    for it in range(1, max_iter + 1):
        s = np.exp(m)
        Jm = fw["J"][:, act].multiply(s[None, :]).tocsr()  # d t / d ln s
        A = sparse.vstack([sparse.diags(Wd) @ Jm, sq * R]).tocsr()
        b = np.concatenate([(t_obs - fw["time"]) * Wd, -sq * (R @ m)])
        dm = lsqr(A, b, atol=1e-8, btol=1e-8, iter_lim=2000)[0]
        step = 1.0
        for _ in range(6):
            m_try = np.clip(m + step * dm, lo, hi)
            fw_try, phi_d_try, phi_try = objective(m_try)
            if phi_try < phi:
                break
            step *= 0.5
        else:
            break
        rel = (phi - phi_try) / phi
        m, fw, phi_d, phi = m_try, fw_try, phi_d_try, phi_try
        rows.append({"iteration": it, "chi2": phi_d / t_obs.size,
                     "rms_ms": 1e3 * np.sqrt(np.mean((t_obs - fw["time"]) ** 2)),
                     "phi": phi, "step": step})
        if verbose:
            print(f"iter {it}: chi2 {rows[-1]['chi2']:.2f}, "
                  f"RMS {rows[-1]['rms_ms']:.2f} ms, step {step:g}")
        if rows[-1]["chi2"] <= target_chi2 or rel < 0.01:
            break

    velocity = velocity_of(m)
    final = forward(grid, velocity, sx, rx, rays=keep_rays)
    tab["predicted"] = final["time"]
    tab["residual"] = t_obs - final["time"]
    tab["sigma"] = sigma
    # Column sums of J: the ray length attributed to each node by the
    # bilinear weights, i.e. how much ray passes through its cell.
    coverage = np.asarray(final["J"].sum(axis=0)).reshape(grid.shape)
    return TomographyResult(grid=grid, velocity=velocity, start=start_v,
                            coverage=np.where(grid.active, coverage, 0.0),
                            table=tab, history=pd.DataFrame(rows),
                            settings=settings, rays=final.get("rays"))


def refine_picks(session, *, rounds: int = 3, k: float = 3.0, window: float = 0.004,
                 guided: bool = True, verbose: bool = False, **kwargs) -> TomographyResult:
    """Clean a :class:`PickingSession` against tomography, and return the final model.

    1. Invert the picks in use; mark the ones whose residual is an outlier
       (:meth:`TomographyResult.outliers`) as not used; repeat until none
       are left or ``rounds`` is reached.
    2. If ``guided``, re-pick every trace within ``window`` seconds of the
       clean model's prediction (:meth:`PickingSession.guided_pick`), which
       recovers traces the first pass had locked onto a later arrival, and
       repeat step 1 on the new picks.

    This does automatically what the residual plot asks a person to do. It
    is not a substitute for looking: plot the shots afterwards, and treat
    any trace it dropped as a question about the record. Keyword arguments
    go to :func:`traveltime_tomography`.
    """
    keep_rays = kwargs.pop("keep_rays", True)

    def clean(tag):
        res = traveltime_tomography(session.table(), keep_rays=False, **kwargs)
        for i in range(rounds):
            bad = res.outliers(k)
            if verbose:
                print(f"{tag} round {i}: {len(res.table)} picks, chi2 {res.chi2:.2f}, "
                      f"RMS {1e3 * res.rms:.2f} ms, {len(bad)} outliers")
            if bad.empty:
                break
            session.exclude(bad)
            res = traveltime_tomography(session.table(), keep_rays=False, **kwargs)
        return res

    res = clean("clean")
    if guided:
        session.guided_pick(res, window=window)
        clean("guided")
    # One more inversion of exactly the picks now in use, with the rays kept
    # for plotting unless asked otherwise.
    return traveltime_tomography(session.table(), keep_rays=keep_rays, **kwargs)


def lambda_sweep(table: pd.DataFrame, lams=(3, 10, 30, 100, 300), **kwargs) -> pd.DataFrame:
    """Invert at several ``lam`` values and tabulate fit against roughness.

    The usual choice is the smoothest model that still fits the data to the
    noise level (chi-squared near 1). Returns one row per ``lam`` with
    ``chi2``, ``rms_ms``, ``roughness`` (``|R m|``) and the result object in
    ``result``.
    """
    rows = []
    grid = kwargs.pop("grid", None) or make_grid(
        _prepare(table, kwargs.get("error"), kwargs.get("rel_error", 0.03))[0],
        **{k: kwargs.pop(k) for k in list(kwargs) if k in
           ("x", "depth", "dx", "nodes_per_spacing", "radius", "elevation")})
    for lam in lams:
        res = traveltime_tomography(table, grid=grid, lam=lam, keep_rays=False, **kwargs)
        act = grid.active.ravel()
        m = np.log(1 / res.velocity.ravel()[act])
        rough = float(np.linalg.norm(_roughness(grid, res.settings["z_weight"])[:, act] @ m))
        rows.append({"lam": lam, "chi2": res.chi2, "rms_ms": 1e3 * res.rms,
                     "roughness": rough, "result": res})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# pyGIMLi backend
# ---------------------------------------------------------------------------


def _invert_pygimli(tab, sigma, grid, start_v, settings, *, verbose=False):
    try:
        import pygimli as pg
        from pygimli.physics import TravelTimeManager
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "backend='pygimli' needs pyGIMLi: `pixi run -e full ...`, "
            "`conda install -c gimli -c conda-forge pygimli`, or on Linux/Colab "
            "`pip install pygimli`. The native backend needs nothing extra."
        ) from exc

    sx = tab["source_x"].to_numpy(float)
    rx = tab["receiver_x"].to_numpy(float)
    pos = np.unique(np.round(np.concatenate([sx, rx]), 4))
    surf = lambda x: -np.interp(x, grid.x, grid.surface)  # elevation, positive up
    data = pg.DataContainer()
    data.registerSensorIndex("s")
    data.registerSensorIndex("g")
    for p in pos:
        data.createSensor([float(p), float(surf(p))])
    data.resize(len(tab))
    data.set("s", np.searchsorted(pos, np.round(sx, 4)).astype(float))
    data.set("g", np.searchsorted(pos, np.round(rx, 4)).astype(float))
    data.set("t", tab["time"].to_numpy(float))
    data.set("err", sigma)
    data.markValid(data("t") > 0)

    v_top = float(np.nanmedian(start_v[0]))
    v_bot = float(np.nanmedian(start_v[-1]))
    depth = float(grid.z[-1] - grid.surface.min())
    mgr = TravelTimeManager()
    mgr.invert(
        data,
        secNodes=3,
        paraDX=0.5,
        paraDepth=depth,
        paraMaxCellSize=(grid.dx * 2) ** 2,
        lam=settings["lam"],
        zWeight=settings["z_weight"],
        vTop=v_top,
        vBottom=v_bot,
        limits=list(settings["v_limits"]),
        maxIter=settings["max_iter"],
        verbose=verbose,
    )
    mesh = mgr.paraDomain
    vel_cells = np.asarray(mgr.model)
    # Summed ray length in each cell; as a density it transfers to grid cells
    # of a different size.
    ray_density = np.asarray(mgr.rayCoverage())[: mesh.cellCount()] / np.asarray(
        mesh.cellSizes())
    X, Z = grid.mesh()
    pts = np.column_stack([X.ravel(), -Z.ravel()])
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    centres = np.array([[c.center()[0], c.center()[1]] for c in mesh.cells()])
    lin = LinearNDInterpolator(centres, vel_cells)
    near = NearestNDInterpolator(centres, vel_cells)
    v = lin(pts)
    v = np.where(np.isfinite(v), v, near(pts)).reshape(grid.shape)
    cov = LinearNDInterpolator(centres, ray_density, fill_value=0.0)(pts)
    cov = cov.reshape(grid.shape) * grid.dx ** 2
    cov = np.where(grid.active, cov, 0.0)

    resp = np.asarray(mgr.inv.response)
    tab = tab.copy()
    tab["predicted"] = resp
    tab["residual"] = tab["time"].to_numpy(float) - resp
    tab["sigma"] = sigma
    chi2 = float(mgr.inv.chi2())
    history = pd.DataFrame({"iteration": [int(mgr.inv.inv.iter())], "chi2": [chi2],
                            "rms_ms": [1e3 * float(np.sqrt(np.mean(tab["residual"] ** 2)))]})
    return TomographyResult(grid=grid, velocity=np.where(grid.active, v, np.nan),
                            start=start_v, coverage=cov, table=tab, history=history,
                            settings={**settings, "pygimli_version": pg.__version__})
