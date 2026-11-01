"""2.5-D resistivity forward modelling on a rectangular finite-element mesh.

The ground is two-dimensional -- resistivity varies along the line and with
depth, not across it -- but the current source is a point, so the potential
it makes is three-dimensional. The **2.5-D** trick (Dey & Morrison, 1979)
takes a Fourier cosine transform across the line, which turns the 3-D
problem into a set of independent 2-D ones, one per wavenumber ``k``:

    -div(sigma grad V~) + k^2 sigma V~ = (I / 2) delta(x - x_s) delta(z)

and the potential on the line is ``V = (2 / pi) * sum_j w_j V~(k_j)``. The
wavenumbers and weights are chosen by least squares so that the same sum
reproduces ``1 / r`` -- a uniform half-space -- to a fraction of a percent
over every distance in the survey.

Each 2-D problem is solved with bilinear finite elements on a rectangular
mesh: fine (a quarter of an electrode spacing) under the electrodes, cells
growing geometrically outside so that the far boundaries are many line
lengths away. The ground surface is insulating (no current leaves into the
air); on the other three sides a mixed boundary condition lets the potential
decay as it would to infinity.

Accuracy and the geometric factor
---------------------------------
A point source is a singularity no mesh resolves, so the raw potentials are
a few percent off near the electrodes. The standard remedy, used here, is to
run the same mesh with a uniform 1 ohm-m ground and take the *numerical*
geometric factor ``k_num = 1 / V_num``: ``rho_a = k_num V(rho)`` is then
exact for a uniform ground by construction, and the mesh error cancels to
well under 1 % for realistic structure (the tests check a two-layer earth
against its analytic image-series solution).

Sensitivities
-------------
Because the mesh equations are linear in each cell's conductivity, the
derivative of every measurement with respect to every cell follows from the
potentials alone (the adjoint-state, or reciprocity, method): one solve per
electrode per wavenumber gives the whole Jacobian. That is what makes
Gauss-Newton inversion affordable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear
from scipy.sparse.linalg import splu
from scipy.special import k0, k0e, k1e

from .data import ERTData

__all__ = ["ERTMesh", "make_mesh", "wavenumbers", "forward", "sensitivity",
           "apparent_resistivity"]


# ---------------------------------------------------------------------------
# Wavenumbers
# ---------------------------------------------------------------------------


def wavenumbers(r_min: float, r_max: float, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Wavenumbers ``k_j`` and weights ``w_j`` for the inverse cosine transform.

    Chosen so that ``(2/pi) sum_j w_j K0(k_j r)`` equals ``1/r`` -- the
    transform of a point source -- to within a fraction of a percent for
    ``r_min <= r <= r_max`` (about 0.01 % for ten wavenumbers over a
    two-decade range). Wavenumbers are log-spaced; the weights come from a
    non-negative least-squares fit of the relative error.
    """
    r = np.geomspace(r_min, r_max, 200)
    best = None
    for lo in (0.3, 0.1, 0.05):
        for hi in (3.0, 5.0, 8.0):
            k = np.geomspace(lo / r_max, hi / r_min, n)
            G = k0(np.outer(r, k)) * (2 / np.pi) * r[:, None]  # relative to 1/r
            w = lsq_linear(G, np.ones_like(r), bounds=(0, np.inf), method="bvls").x
            err = np.abs(G @ w - 1).max()
            if best is None or err < best[0]:
                best = (err, k, w)
    _, k, w = best
    return k, w


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

# Bilinear element on a rectangle, local nodes (ix, iz) = (0,0) (1,0) (0,1) (1,1):
_S1 = np.array([[1.0, -1.0], [-1.0, 1.0]])
_M1 = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0
_KX = np.kron(_M1, _S1)  # int dNi/dx dNj/dx  * hx/hz
_KZ = np.kron(_S1, _M1)  # int dNi/dz dNj/dz  * hz/hx
_KM = np.kron(_M1, _M1)  # int Ni Nj          / (hx hz)
_EDGES = {"left": (0, 2), "right": (1, 3), "bottom": (2, 3)}


@dataclass
class ERTMesh:
    """Finite-element mesh and the inversion grid that sits inside it.

    The **model grid** (``x_edges``, ``z_edges``) holds the resistivities
    being solved for, as ``(nz, nx)`` arrays of cells, and covers the part of
    the ground the data can see. The **FE mesh** (``xn``, ``zn`` nodes)
    subdivides each model cell ``refine`` times and adds padding cells that
    grow outwards; each padding cell takes the resistivity of the nearest
    model cell. ``zn`` and ``z_edges`` are depths (positive down) below a
    flat surface.
    """

    x_edges: np.ndarray
    z_edges: np.ndarray
    xn: np.ndarray
    zn: np.ndarray
    cell_map: np.ndarray  # FE cell -> model cell, (nz_fe, nx_fe)
    electrodes: np.ndarray
    electrode_nodes: np.ndarray
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def shape(self) -> tuple[int, int]:
        """Model grid shape ``(nz, nx)``."""
        return (self.z_edges.size - 1, self.x_edges.size - 1)

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.shape))

    @property
    def x(self) -> np.ndarray:
        """Model cell centres along the line."""
        return 0.5 * (self.x_edges[1:] + self.x_edges[:-1])

    @property
    def z(self) -> np.ndarray:
        """Model cell centres in depth."""
        return 0.5 * (self.z_edges[1:] + self.z_edges[:-1])

    @property
    def cell_area(self) -> np.ndarray:
        return np.outer(np.diff(self.z_edges), np.diff(self.x_edges))

    def __repr__(self) -> str:
        return (f"<ERTMesh: model {self.shape[0]} x {self.shape[1]} cells to "
                f"{self.z_edges[-1]:.1f} m; FE {self.zn.size} x {self.xn.size} nodes>")

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """``X, Z`` model-cell centres, each ``(nz, nx)``."""
        return np.meshgrid(self.x, self.z)

    def uniform(self, rho: float) -> np.ndarray:
        return np.full(self.shape, float(rho))

    # -- FE pieces, built once ------------------------------------------------

    def _fe(self) -> dict:
        if "fe" in self._cache:
            return self._cache["fe"]
        nxn, nzn = self.xn.size, self.zn.size
        hx, hz = np.diff(self.xn), np.diff(self.zn)
        HX, HZ = np.meshgrid(hx, hz)
        HX, HZ = HX.ravel(), HZ.ravel()
        jj, ii = np.meshgrid(np.arange(nzn - 1), np.arange(nxn - 1), indexing="ij")
        n0 = (jj * nxn + ii).ravel()
        nodes = np.column_stack([n0, n0 + 1, n0 + nxn, n0 + nxn + 1])  # (ncell, 4)
        stiff = (HZ / HX)[:, None, None] * _KX + (HX / HZ)[:, None, None] * _KZ
        mass = (HX * HZ)[:, None, None] * _KM
        # Mixed-boundary edges: element, local node pair, edge length,
        # distance and direction cosine from the line centre.
        xc = 0.5 * (self.electrodes[:, 0].min() + self.electrodes[:, 0].max())
        X, Z = np.meshgrid(self.xn, self.zn)
        Xf, Zf = X.ravel(), Z.ravel()
        bc = []
        ncx = nxn - 1
        cells = np.arange(nodes.shape[0]).reshape(nzn - 1, ncx)
        for side, cols in (("left", cells[:, 0]), ("right", cells[:, -1]),
                           ("bottom", cells[-1, :])):
            p, q = _EDGES[side]
            for c in cols:
                a, b = nodes[c, p], nodes[c, q]
                mx, mz = 0.5 * (Xf[a] + Xf[b]), 0.5 * (Zf[a] + Zf[b])
                length = np.hypot(Xf[a] - Xf[b], Zf[a] - Zf[b])
                normal = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "bottom": (0.0, 1.0)}[side]
                r = np.hypot(mx - xc, mz)
                cos = ((mx - xc) * normal[0] + mz * normal[1]) / r
                bc.append((c, p, q, length, r, cos))
        bc = np.array(bc)
        rows = np.repeat(nodes, 4, axis=1).ravel()
        cols = np.tile(nodes, (1, 4)).ravel()
        self._cache["fe"] = {"nodes": nodes, "stiff": stiff, "mass": mass, "bc": bc,
                             "rows": rows, "cols": cols, "n": nxn * nzn}
        return self._cache["fe"]

    def element_matrices(self, k: float) -> np.ndarray:
        """``(ncell, 4, 4)`` element matrices for unit conductivity at wavenumber ``k``."""
        fe = self._fe()
        Ke = fe["stiff"] + k * k * fe["mass"]
        bc = fe["bc"]
        c = bc[:, 0].astype(int)
        p, q = bc[:, 1].astype(int), bc[:, 2].astype(int)
        beta = k * k1e(k * bc[:, 4]) / k0e(k * bc[:, 4]) * bc[:, 5]
        edge = (beta * bc[:, 3])[:, None, None] * _M1[None]
        Ke = Ke.copy()
        np.add.at(Ke, (c, p, p), edge[:, 0, 0])
        np.add.at(Ke, (c, p, q), edge[:, 0, 1])
        np.add.at(Ke, (c, q, p), edge[:, 1, 0])
        np.add.at(Ke, (c, q, q), edge[:, 1, 1])
        return Ke

    def fe_conductivity(self, rho: np.ndarray) -> np.ndarray:
        """Conductivity of every FE cell from a model-grid resistivity."""
        rho = np.asarray(rho, dtype=float)
        if rho.shape != self.shape:
            raise ValueError(f"resistivity must have shape {self.shape}, got {rho.shape}")
        return 1.0 / rho.ravel()[self.cell_map.ravel()]


def _grow(start: float, step: float, factor: float, distance: float) -> np.ndarray:
    """Positions from ``start`` outward with steps growing by ``factor`` up to ``distance``."""
    out, pos, h = [], start, step
    while abs(pos - start) < distance:
        h *= factor
        pos += np.sign(distance) * h if distance else h
        out.append(pos)
    return np.array(out)


def make_mesh(data: ERTData, *, depth: float | None = None, dx: float | None = None,
              dz0: float | None = None, z_factor: float = 1.1, refine: int = 2,
              margin: float | None = None, padding: float = 5.0) -> ERTMesh:
    """A mesh sized to a survey.

    Parameters
    ----------
    data
        The survey; only the electrode positions and pseudo-depths are used.
    depth
        Depth of the model grid. Default: 1.5 times the deepest median depth
        of investigation, which is about as deep as the data can see.
    dx
        Model cell width. Default: half the electrode spacing, so electrodes
        sit on cell edges.
    dz0, z_factor
        Thickness of the top layer of cells (default a quarter of the
        spacing) and how much each layer below thickens.
    refine
        FE cells per model cell in each direction.
    margin
        How far the model grid extends beyond the outermost electrodes.
        Default: two cell widths.
    padding
        How far the FE mesh extends beyond the model grid, in line lengths.
    """
    if data.metadata.get("topography") or np.any(data.electrodes[:, 1] != 0):
        raise NotImplementedError("the ERT mesh assumes a flat surface; topography is not "
                                  "supported yet")
    used = data.used_electrodes()
    ex = data.electrodes[used, 0]
    spacing = data.spacing
    dx = spacing / 2 if dx is None else dx
    dz0 = spacing / 4 if dz0 is None else dz0
    margin = 2 * dx if margin is None else margin
    if depth is None:
        depth = 1.5 * float(data.data["depth"].max())
    x0 = ex.min() - np.ceil(margin / dx - 1e-9) * dx
    x1 = ex.max() + np.ceil(margin / dx - 1e-9) * dx
    x_edges = x0 + dx * np.arange(int(round((x1 - x0) / dx)) + 1)
    z_edges = [0.0]
    h = dz0
    while z_edges[-1] < depth - 1e-9:
        z_edges.append(z_edges[-1] + h)
        h *= z_factor
    z_edges = np.array(z_edges)

    def subdivide(edges):
        f = np.linspace(0, 1, refine + 1)[:-1]
        return np.append((edges[:-1, None] + np.diff(edges)[:, None] * f).ravel(), edges[-1])

    xc, zc = subdivide(x_edges), subdivide(z_edges)
    length = max(x1 - x0, 1.0)
    pad = padding * length
    left = x0 - _grow(0, xc[1] - xc[0], 1.4, pad)[::-1]
    right = x1 + _grow(0, xc[-1] - xc[-2], 1.4, pad)
    below = zc[-1] + _grow(0, zc[-1] - zc[-2], 1.4, pad)
    xn = np.concatenate([left, xc, right])
    zn = np.concatenate([zc, below])
    # FE cell -> model cell: model cells by index, padding takes the nearest.
    xmid, zmid = 0.5 * (xn[1:] + xn[:-1]), 0.5 * (zn[1:] + zn[:-1])
    ix = np.clip(np.searchsorted(x_edges, xmid) - 1, 0, x_edges.size - 2)
    iz = np.clip(np.searchsorted(z_edges, zmid) - 1, 0, z_edges.size - 2)
    cell_map = iz[:, None] * (x_edges.size - 1) + ix[None, :]
    enode = np.full(data.n_electrodes, -1)
    for i in used:
        j = int(np.argmin(np.abs(xn - data.electrodes[i, 0])))
        if abs(xn[j] - data.electrodes[i, 0]) > 1e-6 * spacing:
            raise ValueError(f"electrode {i} at x={data.electrodes[i, 0]:g} is not on a node; "
                             "choose dx so electrodes fall on cell edges")
        enode[i] = j  # surface row, z = 0
    return ERTMesh(x_edges, z_edges, xn, zn, cell_map, data.electrodes.copy(), enode)


# ---------------------------------------------------------------------------
# Forward solve
# ---------------------------------------------------------------------------


def _wavenumbers_for(mesh: ERTMesh, data: ERTData):
    key = ("k", round(data.spacing, 9))
    if key not in mesh._cache:
        ex = data.electrodes[data.used_electrodes(), 0]
        r_min = 0.5 * data.spacing
        r_max = 2.0 * max(np.ptp(ex), data.spacing)
        mesh._cache[key] = wavenumbers(r_min, r_max)
    return mesh._cache[key]


def _potentials(mesh: ERTMesh, sigma_fe: np.ndarray, electrodes: np.ndarray, kw):
    """Unit-current potentials ``(n_k, n_nodes, n_used)`` for every used electrode."""
    fe = mesh._fe()
    ks = kw[0]
    n = fe["n"]
    rhs = np.zeros((n, electrodes.size))
    rhs[mesh.electrode_nodes[electrodes], np.arange(electrodes.size)] = 0.5
    U = np.empty((ks.size, n, electrodes.size))
    for j, k in enumerate(ks):
        Ke = mesh.element_matrices(k) * sigma_fe[:, None, None]
        A = sparse.csc_matrix((Ke.ravel(), (fe["rows"], fe["cols"])), shape=(n, n))
        U[j] = splu(A).solve(rhs)
    return U


def forward(mesh: ERTMesh, rho: np.ndarray, data: ERTData, *,
            jacobian: bool = False) -> dict[str, Any]:
    """Apparent resistivities for a resistivity model, and their sensitivities.

    Parameters
    ----------
    rho
        ``(nz, nx)`` model-cell resistivities (ohm-m) on ``mesh``.
    data
        The measurements to predict (only electrode indices are used).
    jacobian
        Also return ``J``, ``(n_data, n_cells)``: the derivative of
        ``ln rho_a`` with respect to ``ln rho`` of each model cell.

    Returns
    -------
    dict with ``rhoa`` (one per measurement), ``v`` (potential difference
    for unit current, ohm), ``k_num`` (numerical geometric factor) and, if
    asked, ``J``.
    """
    sigma = mesh.fe_conductivity(rho)
    kw = _wavenumbers_for(mesh, data)
    used = data.used_electrodes()
    col = np.full(data.n_electrodes + 1, -1)
    col[used] = np.arange(used.size)
    idx = data.data[list("abmn")].to_numpy()

    def transfer(U):
        """Integrated potentials at the electrodes -> V for each measurement."""
        ws = kw[1]
        Ve = (2 / np.pi) * np.tensordot(ws, U[:, mesh.electrode_nodes[used], :], axes=1)
        # Ve[i, j]: potential at electrode used[i] from a unit source at used[j]
        Ve = np.vstack([np.hstack([Ve, np.zeros((used.size, 1))]), np.zeros(used.size + 1)])
        a, b, m, n = (col[np.where(idx[:, c] < 0, data.n_electrodes, idx[:, c])] for c in range(4))
        return Ve[m, a] - Ve[n, a] - Ve[m, b] + Ve[n, b]

    key = ("k_num", data.n_electrodes, tuple(map(tuple, idx)))
    if key not in mesh._cache:
        U1 = _potentials(mesh, np.ones_like(sigma), used, kw)
        mesh._cache[key] = 1.0 / transfer(U1)
    k_num = mesh._cache[key]
    U = _potentials(mesh, sigma, used, kw)
    v = transfer(U)
    out = {"rhoa": k_num * v, "v": v, "k_num": k_num,
           "_state": (sigma, U, kw, idx, col, data.n_electrodes)}
    if jacobian:
        out["J"] = sensitivity(mesh, out)
    return out


def sensitivity(mesh: ERTMesh, fw: dict) -> np.ndarray:
    """``d ln rho_a / d ln rho`` for each model cell, from a :func:`forward` result.

    By reciprocity: the derivative of the voltage at M due to a source at A
    with respect to one element's conductivity is minus the product of the
    potentials of unit sources at A and at M through that element's matrix.
    It reuses the potentials of the forward solve, so it costs no extra
    solves.
    """
    sigma, U, (ks, ws), idx, col, n_e = fw["_state"]
    v = fw["v"]
    fe = mesh._fe()
    nodes = fe["nodes"]
    n_nodes = U.shape[1]
    src = [col[np.where(idx[:, c] < 0, n_e, idx[:, c])] for c in range(4)]
    J_fe = np.zeros((nodes.shape[0], idx.shape[0]))
    for j, k in enumerate(ks):
        Uj = np.hstack([U[j], np.zeros((n_nodes, 1))])  # last column: remote
        uS = (Uj[:, src[0]] - Uj[:, src[1]])[nodes]  # current dipole, (ncell, 4, nd)
        uR = (Uj[:, src[2]] - Uj[:, src[3]])[nodes]  # potential dipole (reciprocity)
        acc = np.einsum("cpd,cpd->cd", uS, mesh.element_matrices(k) @ uR)
        # dV/dsigma_e = -2 (2/pi) w_j uS^T K_e uR ; the 2 is from the I/2 source
        J_fe -= (4 / np.pi) * ws[j] * acc
    # d ln rho_a / d ln rho_c = -(1/V) sum_{e in c} sigma_e dV/dsigma_e
    J_fe *= -sigma[:, None] / v[None, :]
    P = sparse.csr_matrix((np.ones(nodes.shape[0]), (mesh.cell_map.ravel(),
                                                     np.arange(nodes.shape[0]))),
                          shape=(mesh.n_cells, nodes.shape[0]))
    return np.asarray((P @ J_fe).T)


def apparent_resistivity(data: ERTData, rho, *, mesh: ERTMesh | None = None,
                         noise: float = 0.0, seed: int | None = 0) -> ERTData:
    """A synthetic survey: ``data``'s measurements over a resistivity model.

    ``rho`` is a ``(nz, nx)`` array on ``mesh`` or a function ``rho(X, Z)``
    of cell-centre positions. ``noise`` adds Gaussian noise of that relative
    size (0.03 = 3 %). Returns a copy of ``data`` with ``rhoa`` replaced.
    """
    mesh = mesh or make_mesh(data)
    if callable(rho):
        X, Z = mesh.mesh()
        rho = np.broadcast_to(rho(X, Z), mesh.shape).astype(float)
    out = data.copy()
    pred = forward(mesh, rho, data)["rhoa"]
    if noise:
        rng = np.random.default_rng(seed)
        pred = pred * (1 + noise * rng.standard_normal(pred.size))
    out.data["rhoa"] = pred
    out.data["r"] = pred / out.data["k"]
    out.provenance.record("synthetic", noise=noise)
    return out
