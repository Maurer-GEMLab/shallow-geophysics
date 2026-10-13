"""Regularised linear inversion of gravity or magnetic data on a 2-D mesh.

The forward problem is linear, ``d = G m``, and badly underdetermined: a
few dozen stations, thousands of cells. The inversion picks, out of all the
models that fit the data to within their noise, the one that is smallest and
smoothest:

    minimise  phi_d + beta phi_m
    phi_d = || (G m - d) / sigma ||^2
    phi_m = alpha_s || w m ||^2 + alpha_x || d(w m)/dx ||^2 + alpha_z || d(w m)/dz ||^2

(Li and Oldenburg, 1996, 1998). ``w`` is a weight per cell. Without it the
smallest model puts everything at the surface, because a shallow cell can
explain a datum with a much smaller value than a deep one. The cure is a
depth weight that falls off as the square root of the kernel,

    w(z) = (z + z0)^(-nu)

with ``nu = 1/2`` for 2-D gravity (kernel ~ 1/r) and ``nu = 1`` for 2-D
magnetics (kernel ~ 1/r^2). In 3-D the kernels fall off one power faster and
so do the weights.

``beta`` is chosen so that ``phi_d`` equals the number of data -- each
datum fit to about one standard deviation, the *discrepancy principle*.

Without bounds the minimiser has a closed form. With ``B = (L^T L)^-1``
(``L`` the regularisation operator, so ``phi_m = ||L m||^2``) it is

    m = B G^T (G B G^T + beta Sigma)^-1 d,       Sigma = diag(sigma^2)

which needs one sparse factorisation for all ``beta``, after which each
``beta`` costs a solve the size of the data. Bounds (a void can only be
lighter than rock) are imposed with L-BFGS-B on the same objective.

A smooth model is the most conservative answer, not the most likely one: it
spreads a compact body over a wide blur of small values.
:meth:`LinearInversion.solve_compact` asks instead for the model with the
fewest cells away from zero, which is the better guess for a tunnel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import brentq, minimize
from scipy.sparse.linalg import splu

from .mesh import Mesh2D

__all__ = ["DEPTH_EXPONENT", "InversionResult", "LinearInversion", "depth_weights"]

#: Depth-weighting exponents that balance the decay of 2-D kernels.
DEPTH_EXPONENT = {"gravity": 0.5, "magnetics": 1.0}


def depth_weights(mesh: Mesh2D, exponent: float, z0: float = 0.0) -> np.ndarray:
    """``(z + z0)^(-exponent)`` at every cell centre, scaled to a maximum of 1.

    ``z0`` is the station height, so the weight follows the distance from the
    sensor rather than from the ground.
    """
    _, z = mesh.centers()
    w = (z + z0) ** (-float(exponent))
    return w / w.max()


def _difference_operators(mesh: Mesh2D) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """First differences along x and z, scaled so ``||D m||^2`` approximates
    the integral of the squared gradient over the section."""
    nz, nx = mesh.shape
    idx = np.arange(mesh.n_cells).reshape(nz, nx)
    dxc = np.diff(mesh.x_centers)
    dzc = np.diff(mesh.z_centers)

    # along x: cells (j, i) and (j, i+1)
    a, b = idx[:, :-1].ravel(), idx[:, 1:].ravel()
    s = np.sqrt(np.outer(mesh.dz, 1 / dxc)).ravel()
    rows = np.arange(a.size)
    Dx = sparse.csr_matrix((np.r_[-s, s], (np.r_[rows, rows], np.r_[a, b])),
                           shape=(a.size, mesh.n_cells))
    # along z: cells (j, i) and (j+1, i)
    a, b = idx[:-1, :].ravel(), idx[1:, :].ravel()
    s = np.sqrt(np.outer(1 / dzc, mesh.dx)).ravel()
    rows = np.arange(a.size)
    Dz = sparse.csr_matrix((np.r_[-s, s], (np.r_[rows, rows], np.r_[a, b])),
                           shape=(a.size, mesh.n_cells))
    return Dx, Dz


@dataclass
class InversionResult:
    """One recovered model and how well it does."""

    model: np.ndarray
    predicted: np.ndarray
    beta: float
    phi_d: float
    phi_m: float
    n_data: int
    bounds: tuple | None = None
    info: dict = field(default_factory=dict, repr=False)

    @property
    def chi2(self) -> float:
        """``phi_d`` per datum: about 1 when the data are fit to their noise."""
        return self.phi_d / self.n_data


class LinearInversion:
    """Everything that does not change with ``beta``, set up once.

    Parameters
    ----------
    mesh, G
        The mesh and the forward matrix for it (``gravity_kernel`` or
        ``magnetic_kernel``).
    data, std
        Observed data and their standard deviations, same units as ``G``.
    length
        Smoothing length in metres: ``alpha_s = 1 / length^2`` against
        ``alpha_x = alpha_z = 1``. Longer gives smoother models.
    z_smooth
        Multiplies ``alpha_z``. Below 1 lets the model change faster with
        depth than along the profile.
    weighting
        ``"gravity"`` or ``"magnetics"`` for the depth weight that suits that
        kernel (see :func:`depth_weights`), ``"none"``, or an array of one
        weight per cell.
    z0
        Station height for the depth weight, metres.
    """

    def __init__(
        self,
        mesh: Mesh2D,
        G: np.ndarray,
        data,
        std,
        *,
        length: float = 2.0,
        z_smooth: float = 1.0,
        weighting="gravity",
        z0: float = 0.0,
    ) -> None:
        self.mesh = mesh
        self.G = np.asarray(G, dtype=float)
        self.data = np.asarray(data, dtype=float)
        self.std = np.broadcast_to(np.asarray(std, dtype=float), self.data.shape).copy()
        if self.G.shape != (self.data.size, mesh.n_cells):
            raise ValueError(f"G must be (n_data, n_cells) = {(self.data.size, mesh.n_cells)}, "
                             f"got {self.G.shape}")
        if np.any(self.std <= 0):
            raise ValueError("std must be positive")

        if isinstance(weighting, str):
            if weighting in DEPTH_EXPONENT:
                w = depth_weights(mesh, DEPTH_EXPONENT[weighting], z0)
            elif weighting == "none":
                w = np.ones(mesh.n_cells)
            else:
                raise ValueError("weighting must be 'gravity', 'magnetics', 'none' or an array")
        else:
            w = np.asarray(weighting, dtype=float)
            if w.shape != (mesh.n_cells,):
                raise ValueError("a weighting array needs one value per cell")
        self.weights = w

        self.length = float(length)
        self.z_smooth = float(z_smooth)
        self._Dx, self._Dz = _difference_operators(mesh)
        self._set_operator(self._operator())

    def _operator(self, support=None, smooth: float = 1.0) -> sparse.csr_matrix:
        """``L`` with ``phi_m = ||L m||^2``. ``support`` reweights the
        smallness term cell by cell (compact inversion); ``smooth`` scales
        the two gradient terms."""
        W = sparse.diags(self.weights)
        s = np.sqrt(self.mesh.area) / self.length
        if support is not None:
            s = s * support
        return sparse.vstack([
            sparse.diags(s) @ W,
            smooth * self._Dx @ W,
            smooth * np.sqrt(self.z_smooth) * self._Dz @ W,
        ]).tocsr()

    def _set_operator(self, L: sparse.csr_matrix) -> None:
        self.L = L
        self._LtL = (L.T @ L).tocsc()
        # B G^T, and the data-space matrix G B G^T, shared by every beta.
        self._BGt = splu(self._LtL).solve(np.ascontiguousarray(self.G.T))
        self._K = self.G @ self._BGt

    @property
    def n_data(self) -> int:
        return self.data.size

    # -- objective ----------------------------------------------------------

    def phi_d(self, m: np.ndarray) -> float:
        return float(np.sum(((self.G @ m - self.data) / self.std) ** 2))

    def phi_m(self, m: np.ndarray) -> float:
        return float(np.sum((self.L @ m) ** 2))

    def _result(self, m, beta, bounds=None, **info) -> InversionResult:
        return InversionResult(model=m, predicted=self.G @ m, beta=float(beta),
                               phi_d=self.phi_d(m), phi_m=self.phi_m(m),
                               n_data=self.n_data, bounds=bounds, info=info)

    # -- solving ------------------------------------------------------------

    def _unbounded(self, beta: float) -> np.ndarray:
        A = self._K + beta * np.diag(self.std**2)
        return self._BGt @ np.linalg.solve(A, self.data)

    def _bounded(self, beta: float, lower, upper, m0=None, maxiter=20_000):
        Ginv = self.G / self.std[:, None]
        dw = self.data / self.std
        LtL = self._LtL
        # Scale so the objective is O(1) for L-BFGS-B's tolerances.
        scale = 1.0 / self.n_data

        def fun(m):
            r = Ginv @ m - dw
            Lm = LtL @ m
            f = r @ r + beta * (m @ Lm)
            g = 2 * (Ginv.T @ r) + 2 * beta * Lm
            return scale * f, scale * g

        lo = np.broadcast_to(lower, (self.mesh.n_cells,))
        hi = np.broadcast_to(upper, (self.mesh.n_cells,))
        if m0 is None:
            m0 = self._unbounded(beta)
        m0 = np.clip(m0, lo, hi)
        res = minimize(fun, m0, jac=True, method="L-BFGS-B",
                       bounds=list(zip(lo, hi)),
                       options={"maxiter": maxiter, "maxfun": 2 * maxiter,
                                "ftol": 1e-13, "gtol": 1e-10, "maxcor": 30})
        return res.x, {"iterations": res.nit, "converged": bool(res.success),
                       "message": str(res.message)}

    def solve(self, beta="target", bounds=None) -> InversionResult:
        """Recover a model.

        ``beta`` is a number, or ``"target"`` to find the one that fits the
        data to their noise (``phi_d`` = number of data). ``bounds`` is
        ``(lower, upper)``, scalars or one value per cell; ``None`` means
        unbounded.
        """
        if bounds is None:
            if beta == "target":
                beta = self.target_beta()
            return self._result(self._unbounded(beta), beta)

        lower, upper = bounds
        if beta != "target":
            m, info = self._bounded(beta, lower, upper)
            return self._result(m, beta, bounds, **info)

        # phi_d grows with beta for the bounded problem too: bisect in log beta,
        # warm-starting each solve from the last.
        state = {"m": None}

        def misfit(log_beta):
            m, _ = self._bounded(10**log_beta, lower, upper, m0=state["m"])
            state["m"] = m
            return np.log10(self.phi_d(m) / self.n_data)

        b0 = np.log10(self.target_beta())
        lo_b, hi_b = b0 - 1, b0 + 1
        while misfit(lo_b) > 0:
            lo_b -= 1
        while misfit(hi_b) < 0:
            hi_b += 1
        log_beta = brentq(misfit, lo_b, hi_b, xtol=0.02)
        m, info = self._bounded(10**log_beta, lower, upper, m0=state["m"])
        return self._result(m, 10**log_beta, bounds, **info)

    def solve_compact(
        self,
        eps: float | None = None,
        bounds=None,
        *,
        n_iter: int = 12,
        smooth: float = 0.1,
    ) -> InversionResult:
        """A compact model: as few cells as possible away from zero.

        Minimum-support regularisation (Last and Kubik, 1983; Portniaguine
        and Zhdanov, 1999) by iterative reweighting. Starting from the
        smooth model, each pass makes the smallness term of every cell
        ``eps^2 / (m^2 + eps^2)`` times as strong: cells already far from
        zero become cheap and grow, cells near zero become expensive and
        shrink, and the body tightens. ``eps`` is the size below which a
        value counts as "zero" -- a few percent of the contrast expected,
        or by default a quarter of the largest value in the smooth model.
        ``smooth`` keeps a little of the gradient penalty so the result is
        not a scatter of single cells. ``beta`` is nudged after every pass
        to keep ``phi_d`` at the number of data.

        The setup of this object is restored afterwards, so :meth:`solve`
        still gives the smooth model.
        """
        start = self.solve(bounds=bounds)
        m = start.model
        if eps is None:
            eps = 0.25 * float(np.max(np.abs(m)))
        if eps <= 0:
            raise ValueError("eps must be positive")
        original = self.L
        history = []
        try:
            beta = None
            for k in range(n_iter):
                support = eps / np.sqrt(m**2 + eps**2)
                self._set_operator(self._operator(support, smooth))
                if beta is None:
                    beta = self.target_beta()
                else:
                    beta *= float(np.clip(self.n_data / self.phi_d(m), 0.5, 2.0))
                if bounds is None:
                    m = self._unbounded(beta)
                else:
                    m, _ = self._bounded(beta, *bounds, m0=m)
                history.append({"iteration": k + 1, "beta": beta,
                                "chi2": self.phi_d(m) / self.n_data,
                                "min": float(m.min()), "max": float(m.max())})
            result = self._result(m, beta, bounds, eps=eps,
                                  history=pd.DataFrame(history))
        finally:
            self._set_operator(original)
        # phi_m of the compact model under the smooth regulariser, for comparison.
        result.phi_m = self.phi_m(result.model)
        return result

    def target_beta(self, chi2: float = 1.0) -> float:
        """The ``beta`` at which the unbounded model has ``phi_d = chi2 * n_data``."""
        target = chi2 * self.n_data
        if self.phi_d(self._unbounded(1e-12 * self._scale())) > target:
            raise ValueError("the data cannot be fit to their noise even with almost "
                             "no regularisation; check std")

        def g(log_beta):
            return np.log(self.phi_d(self._unbounded(10**log_beta)) / target)

        lo, hi = np.log10(self._scale()) - 8, np.log10(self._scale()) + 4
        while g(hi) < 0:
            hi += 2
        return 10 ** brentq(g, lo, hi, xtol=1e-3)

    def _scale(self) -> float:
        """A beta of the right order of magnitude: trace(K Sigma^-1) / n."""
        return float(np.trace(self._K / self.std[:, None] ** 2) / self.n_data)

    def sweep(self, betas=None) -> pd.DataFrame:
        """``phi_d`` and ``phi_m`` over a range of ``beta`` (unbounded) -- the
        trade-off curve from which ``beta`` is chosen."""
        if betas is None:
            b0 = np.log10(self.target_beta())
            betas = np.logspace(b0 - 4, b0 + 3, 36)
        rows = []
        for b in np.asarray(betas, dtype=float):
            m = self._unbounded(b)
            rows.append({"beta": b, "phi_d": self.phi_d(m), "phi_m": self.phi_m(m)})
        df = pd.DataFrame(rows)
        df["chi2"] = df["phi_d"] / self.n_data
        return df
