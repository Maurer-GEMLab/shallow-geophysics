"""2-D resistivity inversion: a resistivity image from apparent resistivities.

The model is the logarithm of resistivity in every cell of the
:class:`~shallowgeo.ert.forward.ERTMesh` model grid, which keeps
resistivities positive and treats a factor of two the same at 10 and at
1000 ohm-m. Each Gauss-Newton step solves

    minimise  || (ln rho_a,obs - ln rho_a(m)) / e ||^2
              +  lam ( || R m ||^2  +  alpha_s || m - m_ref ||^2 )

where ``e`` is the relative data error, ``R`` takes differences between
neighbouring cells -- vertical ones weighted by ``z_weight`` -- and the
optional ``alpha_s`` term pulls cells the data do not see back towards a
reference model. ``lam`` trades fit against smoothness and is the number a
student has to choose; :func:`lambda_sweep` draws the trade-off. Iterations
stop when the misfit reaches the noise level (chi-squared of about 1) or
stops improving.

This is the same smoothness-constrained (Occam-style) least squares that
RES2DINV, R2/ResIPy and pyGIMLi solve. Details differ -- their meshes,
their choice of ``lam`` at each iteration, robust (L1) options -- so the
images agree in their large features and differ in detail; see
``docs/ert-validation.md`` for a comparison with pyGIMLi and ResIPy's R2.

What the image means
--------------------
Sensitivity falls off quickly with depth and towards the ends of the line,
and where it is small the cells take whatever the smoothing gives them. The
plots fade those cells, using the error-weighted cumulative sensitivity
(:attr:`ERTResult.coverage`), and :func:`doi_index` gives the stricter
Oldenburg & Li (1999) depth-of-investigation test: invert twice from very
different reference models and see where the answers disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .data import ERTData, _log_colorbar_ticks
from .forward import ERTMesh, forward, make_mesh, sensitivity

__all__ = ["ERTResult", "resistivity_tomography", "lambda_sweep", "doi_index"]


def _roughness(mesh: ERTMesh, z_weight: float) -> sparse.csr_matrix:
    """First differences between neighbouring model cells, vertical ones scaled."""
    nz, nx = mesh.shape
    idx = np.arange(nz * nx).reshape(nz, nx)
    blocks = []
    for (a, b), w in (((idx[:, :-1], idx[:, 1:]), 1.0), ((idx[:-1, :], idx[1:, :]), z_weight)):
        a, b = a.ravel(), b.ravel()
        m = a.size
        rows = np.repeat(np.arange(m), 2)
        vals = np.tile([-w, w], m)
        blocks.append(sparse.csr_matrix((vals, (rows, np.column_stack([a, b]).ravel())),
                                        shape=(m, nz * nx)))
    return sparse.vstack(blocks).tocsr()


@dataclass
class ERTResult:
    """What :func:`resistivity_tomography` returns.

    ``resistivity`` and ``start`` are ``(nz, nx)`` on ``mesh`` (ohm-m).
    ``coverage`` is the error-weighted cumulative sensitivity per unit area,
    scaled to 1 at its maximum. ``data`` is the input survey with
    ``predicted``, ``residual`` (percent, ``100 (obs - pred) / pred``) and
    ``error`` columns added. ``history`` has one row per iteration.
    """

    mesh: ERTMesh
    resistivity: np.ndarray
    start: np.ndarray
    coverage: np.ndarray
    data: ERTData
    history: pd.DataFrame
    settings: dict[str, Any]
    J: np.ndarray | None = None

    @property
    def chi2(self) -> float:
        return float(self.history["chi2"].iloc[-1])

    @property
    def rms(self) -> float:
        """RMS misfit in percent, as RES2DINV and ResIPy report it."""
        return float(self.history["rms_pct"].iloc[-1])

    def predict(self, data: ERTData) -> np.ndarray:
        """Apparent resistivities through the final model for any measurements."""
        return forward(make_mesh(data) if data.n_electrodes != self.data.n_electrodes
                       else self.mesh, self.resistivity, data)["rhoa"]

    def outliers(self, k: float = 3.0, *, floor: float = 5.0) -> pd.DataFrame:
        """Measurements whose misfit is far outside the rest.

        A measurement is flagged when ``|residual|`` exceeds both ``k``
        robust standard deviations (1.4826 x MAD) and ``floor`` percent. On
        field data these are usually bad electrode contacts or noisy
        readings: the model cannot bend enough to fit one measurement
        without its neighbours.
        """
        r = self.data.data["residual"]
        sd = 1.4826 * float(np.median(np.abs(r - r.median())))
        return self.data.data[(r.abs() > k * sd) & (r.abs() > floor)]

    def covered(self, threshold: float = 0.01) -> np.ndarray:
        """Cells whose relative coverage is at least ``threshold``, ``(nz, nx)`` bool."""
        return self.coverage >= threshold

    def depth_of_investigation(self, threshold: float = 0.01) -> np.ndarray:
        """Deepest covered cell centre at each model column (NaN if none)."""
        cov = self.covered(threshold)
        zz = np.where(cov, self.mesh.z[:, None], np.nan)
        out = np.full(zz.shape[1], np.nan)
        have = np.isfinite(zz).any(axis=0)
        out[have] = np.nanmax(zz[:, have], axis=0)
        return out

    def profile(self, x: float) -> pd.DataFrame:
        """Resistivity against depth at one position, with the coverage there."""
        i = int(np.argmin(np.abs(self.mesh.x - x)))
        return pd.DataFrame({"depth": self.mesh.z, "resistivity": self.resistivity[:, i],
                             "start": self.start[:, i], "coverage": self.coverage[:, i]})

    def summary(self) -> pd.Series:
        cov = self.covered()
        return pd.Series({
            "measurements": len(self.data),
            "iterations": int(self.history["iteration"].iloc[-1]),
            "lam": self.settings.get("lam"), "z_weight": self.settings.get("z_weight"),
            "chi2": self.chi2, "rms_pct": self.rms,
            "rho_min": float(self.resistivity[cov].min()),
            "rho_max": float(self.resistivity[cov].max()),
            "max_depth_m": float(np.nanmax(self.depth_of_investigation())),
        })

    def to_frame(self) -> pd.DataFrame:
        """One row per model cell: centre ``x``, ``depth``, ``resistivity``, ``coverage``."""
        X, Z = self.mesh.mesh()
        return pd.DataFrame({"x": X.ravel(), "depth": Z.ravel(),
                             "resistivity": self.resistivity.ravel(),
                             "coverage": self.coverage.ravel()})

    # -- plots ---------------------------------------------------------------

    def plot(self, ax=None, *, field: str = "resistivity", mask: bool | np.ndarray = True,
             threshold: float = 0.01, contours=None, vmin=None, vmax=None,
             cmap: str = "turbo", colorbar: bool = True, electrodes: bool = True,
             max_depth: float | None = None):
        """The resistivity image, faded where the data do not reach.

        Parameters
        ----------
        field
            ``"resistivity"``, ``"start"``, ``"coverage"`` or ``"change"``
            (percent change from the starting model).
        mask
            Fade cells whose coverage is below ``threshold`` (or pass a bool
            ``(nz, nx)`` array of cells to keep, such as ``doi < 0.2``). Keep
            it on: the colours there are the smoothness constraint, not the
            ground.
        contours
            Resistivity values to contour, e.g. ``[100, 300, 1000]``.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm, Normalize

        ax = ax or plt.gca()
        m = self.mesh
        if field == "resistivity":
            values, label = self.resistivity, "resistivity (Ω·m)"
        elif field == "start":
            values, label = self.start, "resistivity (Ω·m)"
        elif field == "coverage":
            values, label = self.coverage, "relative coverage"
        elif field == "change":
            values, label = 100 * (self.resistivity - self.start) / self.start, "change (%)"
        else:
            raise ValueError(f"unknown field {field!r}")
        if field == "change":
            lim = np.nanmax(np.abs(values))
            norm = Normalize(vmin=-lim if vmin is None else vmin, vmax=lim if vmax is None else vmax)
            cmap = "RdBu_r" if cmap == "turbo" else cmap
        else:
            pos = values[values > 0]
            norm = LogNorm(vmin=vmin or pos.min(), vmax=vmax or pos.max())
            if field == "coverage":
                cmap = "magma" if cmap == "turbo" else cmap
        im = ax.pcolormesh(m.x_edges, m.z_edges, values, cmap=cmap, norm=norm,
                           shading="flat", rasterized=True)
        if mask is not False and field != "coverage":
            keep = self.covered(threshold) if mask is True else np.asarray(mask, dtype=bool)
            if mask is True:
                fade = 1 - np.clip(np.log10(np.maximum(self.coverage, 1e-12) / threshold) + 1,
                                   0, 1)
            else:
                fade = (~keep).astype(float)
            rgba = np.zeros(m.shape + (4,))
            rgba[..., :3] = 1.0
            rgba[..., 3] = 0.85 * fade
            ax.pcolormesh(m.x_edges, m.z_edges, rgba[..., 3], cmap=_white_alpha(),
                          vmin=0, vmax=1, shading="flat", rasterized=True)
        if contours is not None and field in ("resistivity", "start"):
            X, Z = m.mesh()
            keep = self.covered(threshold)
            cs = ax.contour(X, Z, np.where(keep, values, np.nan), levels=contours,
                            colors="k", linewidths=0.7)
            ax.clabel(cs, fmt="%.0f", fontsize=7)
        if electrodes:
            ex = self.data.electrodes[self.data.used_electrodes(), 0]
            ax.plot(ex, np.zeros_like(ex), "v", color="k", ms=3, clip_on=False)
        ax.set_xlim(m.x_edges[0], m.x_edges[-1])
        bottom = m.z_edges[-1] if max_depth is None else max_depth
        ax.set_ylim(bottom, 0)
        ax.set_aspect("equal")
        ax.set_xlabel("position along line (m)")
        ax.set_ylabel("depth (m)")
        if colorbar:
            cb = ax.figure.colorbar(im, ax=ax, label=label, shrink=0.8, pad=0.02)
            if field != "change":
                _log_colorbar_ticks(cb)
        return ax

    def plot_pseudosections(self, axes=None, **kwargs):
        """Observed and predicted apparent resistivity, and the misfit between them."""
        import matplotlib.pyplot as plt

        if axes is None:
            _, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
        d = self.data
        obs = d.data["rhoa"].to_numpy()
        pred = d.data["predicted"].to_numpy()
        lo, hi = min(obs.min(), pred.min()), max(obs.max(), pred.max())
        d.plot_pseudosection(axes[0], vmin=lo, vmax=hi, **kwargs)
        axes[0].set_title("measured apparent resistivity", fontsize=10)
        d.plot_pseudosection(axes[1], values=pred, vmin=lo, vmax=hi, **kwargs)
        axes[1].set_title("calculated from the model", fontsize=10)
        res = d.data["residual"].to_numpy()
        d.plot_pseudosection(axes[2], values=res, log=False, cmap="RdBu_r",
                             label="misfit (%)", **kwargs)
        axes[2].set_title(f"misfit: RMS {self.rms:.1f} %, chi² {self.chi2:.2f}", fontsize=10)
        return axes

    def plot_fit(self, axes=None):
        """Observed against predicted, the misfit histogram, and the convergence."""
        import matplotlib.pyplot as plt

        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(14, 4))
        d = self.data.data
        a0, a1, a2 = axes
        a0.loglog(d["rhoa"], d["predicted"], ".", ms=4, color="#256abf")
        lim = [min(d["rhoa"].min(), d["predicted"].min()) * 0.9,
               max(d["rhoa"].max(), d["predicted"].max()) * 1.1]
        a0.plot(lim, lim, "k-", lw=0.8)
        a0.set_xlabel("measured ρa (Ω·m)")
        a0.set_ylabel("calculated ρa (Ω·m)")
        a0.set_aspect("equal")
        a1.hist(d["residual"], bins=30, color="#256abf")
        a1.set_xlabel("misfit (%)")
        a1.set_ylabel("measurements")
        h = self.history
        a2.semilogy(h["iteration"], h["chi2"], "o-")
        a2.axhline(1, color="k", lw=0.8, ls="--")
        a2.set_xlabel("iteration")
        a2.set_ylabel("chi²")
        return axes


def _white_alpha():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("fade", [(1, 1, 1, 0), (1, 1, 1, 1)])


def _start_model(mesh: ERTMesh, data: ERTData, start) -> np.ndarray:
    if start is None:
        return mesh.uniform(float(np.exp(np.median(np.log(data.data["rhoa"])))))
    if np.isscalar(start):
        return mesh.uniform(float(start))
    start = np.asarray(start, dtype=float)
    if start.shape != mesh.shape:
        raise ValueError(f"start must have shape {mesh.shape}, got {start.shape}")
    return start


def resistivity_tomography(
    data: ERTData,
    *,
    mesh: ERTMesh | None = None,
    start=None,
    lam: float = 20.0,
    z_weight: float = 0.5,
    rel_error: float = 0.03,
    reference=None,
    alpha_s: float = 0.0,
    max_iter: int = 20,
    target_chi2: float = 1.0,
    rho_limits: tuple[float, float] = (0.1, 1e6),
    keep_jacobian: bool = False,
    verbose: bool = False,
    **mesh_kwargs,
) -> ERTResult:
    """Invert apparent resistivities for a 2-D resistivity image.

    Parameters
    ----------
    data
        An :class:`ERTData`, e.g. from :func:`read_res2dinv` -- trimmed to
        the electrodes actually in the ground. Measurements with
        ``rhoa <= 0`` are dropped. An ``error`` column (relative, 0.05 =
        5 %) overrides ``rel_error`` per measurement.
    mesh
        From :func:`make_mesh`; built from the data if omitted, with any
        extra keyword arguments passed to it (``depth=``, ``dx=`` ...).
    start
        Starting resistivity: an ``(nz, nx)`` array, a number, or ``None``
        for a uniform ground at the median apparent resistivity.
    lam
        Regularisation strength. Larger is smoother and fits worse; smaller
        fits better and grows artefacts. Use :func:`lambda_sweep` to choose.
    z_weight
        Weight of vertical relative to horizontal smoothing. Below 1 lets
        resistivity change faster with depth than along the line.
    rel_error
        Relative error of each measurement (0.03 = 3 %). SuperSting repeat
        errors or reciprocal measurements say what it should be; without
        them, 2-5 % is usual.
    reference, alpha_s
        Reference model (array or number; default the starting model) and
        the weight of the term pulling the model towards it. ``alpha_s = 0``
        is a pure smoothness inversion.
    max_iter, target_chi2
        Stop after ``max_iter`` Gauss-Newton steps, once chi-squared reaches
        ``target_chi2``, or when it stops falling.
    rho_limits
        Resistivities are kept inside these bounds (ohm-m).

    Returns
    -------
    :class:`ERTResult`
    """
    data = data.drop_nonpositive() if (data.data["rhoa"] <= 0).any() else data
    if mesh is None:
        mesh = make_mesh(data, **mesh_kwargs)
    elif mesh_kwargs:
        raise TypeError(f"mesh given, so {sorted(mesh_kwargs)} cannot be used")
    start_rho = _start_model(mesh, data, start)
    ref = start_rho if reference is None else _start_model(mesh, data, reference)
    settings = {"lam": lam, "z_weight": z_weight, "rel_error": rel_error,
                "alpha_s": alpha_s, "max_iter": max_iter, "rho_limits": rho_limits}

    d_obs = np.log(data.data["rhoa"].to_numpy(float))
    err = (data.data["error"].to_numpy(float) if "error" in data.data
           else np.full(d_obs.size, rel_error))
    Wd = 1.0 / err
    R = _roughness(mesh, z_weight)
    RtR = (R.T @ R).toarray()
    if alpha_s:
        RtR += alpha_s * np.eye(mesh.n_cells)
    m_ref = np.log(ref.ravel())
    lo, hi = np.log(rho_limits[0]), np.log(rho_limits[1])
    m = np.log(start_rho.ravel())

    def model_reg(mv):
        rm = R @ mv
        out = float(rm @ rm)
        if alpha_s:
            dm = mv - m_ref
            out += alpha_s * float(dm @ dm)
        return out

    def evaluate(mv, jac):
        fw = forward(mesh, np.exp(mv).reshape(mesh.shape), data, jacobian=jac)
        if np.any(fw["rhoa"] <= 0):  # an extreme trial model: reject it
            return fw, np.inf, np.inf
        r = (d_obs - np.log(fw["rhoa"])) * Wd
        phi_d = float(r @ r)
        return fw, phi_d, phi_d + lam * model_reg(mv)

    def row(it, fw, phi_d, phi, **extra):
        pct = 100 * (data.data["rhoa"].to_numpy() - fw["rhoa"]) / fw["rhoa"]
        return {"iteration": it, "chi2": phi_d / d_obs.size,
                "rms_pct": float(np.sqrt(np.mean(pct ** 2))), "phi": phi, **extra}

    fw, phi_d, phi = evaluate(m, True)
    rows = [row(0, fw, phi_d, phi)]
    if verbose:
        print(f"start: chi2 {rows[0]['chi2']:.2f}, RMS {rows[0]['rms_pct']:.1f} %")
    for it in range(1, max_iter + 1):
        WJ = fw["J"] * Wd[:, None]
        r = (d_obs - np.log(fw["rhoa"])) * Wd
        g = WJ.T @ r - lam * (R.T @ (R @ m) + (alpha_s * (m - m_ref) if alpha_s else 0))
        H = WJ.T @ WJ + lam * RtR
        dm = np.linalg.solve(H, g)
        step = 1.0
        for _ in range(6):
            m_try = np.clip(m + step * dm, lo, hi)
            fw_try, phi_d_try, phi_try = evaluate(m_try, False)
            if phi_try < phi:
                break
            step *= 0.5
        else:
            break
        rel = (phi - phi_try) / phi
        m, fw, phi_d, phi = m_try, fw_try, phi_d_try, phi_try
        fw["J"] = sensitivity(mesh, fw)
        rows.append(row(it, fw, phi_d, phi, step=step))
        if verbose:
            print(f"iter {it}: chi2 {rows[-1]['chi2']:.2f}, "
                  f"RMS {rows[-1]['rms_pct']:.1f} %, step {step:g}")
        if rows[-1]["chi2"] <= target_chi2 or rel < 0.01:
            break

    rho = np.exp(m).reshape(mesh.shape)
    out = data.copy()
    out.data["predicted"] = fw["rhoa"]
    out.data["residual"] = 100 * (out.data["rhoa"] - fw["rhoa"]) / fw["rhoa"]
    out.data["error"] = err
    cov = (np.abs(fw["J"]) * Wd[:, None]).sum(axis=0).reshape(mesh.shape) / mesh.cell_area
    cov = cov / cov.max()
    out.provenance.record("resistivity_tomography", **{k: v for k, v in settings.items()
                                                       if not isinstance(v, tuple)})
    return ERTResult(mesh=mesh, resistivity=rho, start=start_rho, coverage=cov, data=out,
                     history=pd.DataFrame(rows), settings=settings,
                     J=fw["J"] if keep_jacobian else None)


def lambda_sweep(data: ERTData, lams=(3, 10, 30, 100, 300), **kwargs) -> pd.DataFrame:
    """Invert at several ``lam`` values and tabulate fit against roughness.

    The usual choice is the smoothest model that still fits the data to the
    noise level (chi-squared near 1), or the corner of the L-curve when the
    noise level is not known. Returns one row per ``lam`` with ``chi2``,
    ``rms_pct``, ``roughness`` (``|R m|``) and the result in ``result``.
    """
    mesh = kwargs.pop("mesh", None) or make_mesh(
        data, **{k: kwargs.pop(k) for k in list(kwargs)
                 if k in ("depth", "dx", "dz0", "z_factor", "refine", "margin", "padding")})
    rows = []
    for lam in lams:
        res = resistivity_tomography(data, mesh=mesh, lam=lam, **kwargs)
        m = np.log(res.resistivity.ravel())
        rough = float(np.linalg.norm(_roughness(mesh, res.settings["z_weight"]) @ m))
        rows.append({"lam": lam, "chi2": res.chi2, "rms_pct": res.rms,
                     "roughness": rough, "result": res})
    return pd.DataFrame(rows)


def doi_index(data: ERTData, *, factor: float = 10.0, alpha_s: float = 0.01,
              **kwargs) -> tuple[np.ndarray, ERTResult, ERTResult]:
    """Oldenburg & Li (1999) depth-of-investigation index.

    Inverts twice with reference (and starting) models ``factor`` times
    above and below the median apparent resistivity, each with a weak pull
    ``alpha_s`` towards its reference. Where the data constrain the model
    both inversions arrive at the same resistivity and the index is near 0;
    where they do not, each stays near its own reference and the index
    approaches 1. Values below about 0.1-0.2 are usually taken as resolved.

    Returns ``(doi, low, high)``: the ``(nz, nx)`` index and both results.
    """
    mesh = kwargs.pop("mesh", None) or make_mesh(data)
    base = float(np.exp(np.median(np.log(data.data["rhoa"]))))
    lo_ref, hi_ref = base / factor, base * factor
    low = resistivity_tomography(data, mesh=mesh, start=lo_ref, reference=lo_ref,
                                 alpha_s=alpha_s, **kwargs)
    high = resistivity_tomography(data, mesh=mesh, start=hi_ref, reference=hi_ref,
                                  alpha_s=alpha_s, **kwargs)
    doi = np.abs(np.log(high.resistivity) - np.log(low.resistivity)) / np.log(hi_ref / lo_ref)
    return doi, low, high
