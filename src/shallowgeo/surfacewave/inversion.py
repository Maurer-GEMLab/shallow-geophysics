"""Layered Vs inversion of a dispersion curve.

A deliberately plain approach: fixed layer thicknesses (chosen from the
wavelength range of the data), log-parameterised shear velocities, and
``scipy.optimize.least_squares`` with an optional roughness penalty. It is
transparent enough to teach from and fast enough to run in a notebook. When
the class needs a global search or a joint Vp/thickness inversion,
``evodcinv`` (same author as ``disba``) is the drop-in upgrade and this
module's :class:`InversionResult` is the shape to return from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .dispersion import DispersionCurve
from .models import LayeredModel, forward_dispersion


@dataclass
class InversionResult:
    model: LayeredModel
    observed: DispersionCurve
    predicted: DispersionCurve
    initial: LayeredModel
    rms: float
    n_evaluations: int
    success: bool
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def residuals(self) -> np.ndarray:
        pred = self.predicted.resample(self.observed.frequency)
        obs = self.observed.resample(pred.frequency)
        return pred.velocity - obs.velocity

    def plot(self, axes=None):
        """Two panels: dispersion fit and Vs profile."""
        import matplotlib.pyplot as plt

        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        ax_d, ax_m = axes
        self.observed.plot(ax=ax_d, label="observed", color="k")
        ax_d.plot(self.predicted.frequency, self.predicted.velocity, "-",
                  color="firebrick", label=f"model, rms {self.rms:.1f} m/s")
        ax_d.legend()
        self.initial.plot(ax=ax_m, ls="--", color="0.6", label="initial")
        self.model.plot(ax=ax_m, color="firebrick", label="inverted")
        ax_m.legend()
        return axes


def initial_model(
    curve: DispersionCurve,
    n_layers: int = 5,
    *,
    max_depth: float | None = None,
    vs_vr: float = 1.1,
    depth_fraction: float = 0.33,
    vp_vs: float | None = None,
) -> LayeredModel:
    """Starting model from the wavelength--depth rule of thumb.

    A Rayleigh wave of wavelength λ samples to roughly λ/3 (some use λ/2),
    and Vs is about 1.1 times the phase velocity. Layer boundaries are spaced
    geometrically between the shallowest and deepest resolved depths, which
    gives thin layers near the surface where the data have resolution and
    thick ones at depth where they do not.
    """
    if n_layers < 2:
        raise ValueError("need at least two layers (one over a half-space)")
    lam = curve.wavelength
    z_min = depth_fraction * lam.min()
    z_max = max_depth or depth_fraction * lam.max()
    if z_max <= z_min:
        raise ValueError("max_depth must exceed the shallowest resolved depth")

    # Bottoms of the n-1 finite layers, geometrically spaced.
    if n_layers == 2:
        bottoms = np.array([np.sqrt(z_min * z_max)])
    else:
        bottoms = np.geomspace(z_min, z_max, n_layers - 1)
    tops = np.concatenate([[0.0], bottoms[:-1]])
    thickness = np.diff(np.concatenate([[0.0], bottoms]))

    # Vs of each layer: phase velocity of the wavelength whose λ/3 is the
    # layer midpoint; the half-space takes a depth below the last interface.
    order = np.argsort(lam)
    z_of_lam = depth_fraction * lam[order]
    v_of_lam = curve.velocity[order]
    mids = np.concatenate([0.5 * (tops + bottoms), [1.5 * bottoms[-1]]])
    vs = vs_vr * np.interp(mids, z_of_lam, v_of_lam)
    kwargs = {} if vp_vs is None else {"vp_vs": vp_vs}
    return LayeredModel(thickness, vs, metadata={"from": "initial_model"}, **kwargs)


def invert_dispersion(
    curve: DispersionCurve,
    initial: LayeredModel | int = 5,
    *,
    vs_bounds: tuple[float, float] = (50.0, 4000.0),
    smoothing: float = 0.0,
    invert_thickness: bool = False,
    thickness_bounds: tuple[float, float] = (0.25, 200.0),
    max_evaluations: int = 200,
    mode: int | None = None,
    wave: str | None = None,
) -> InversionResult:
    """Fit a layered Vs model to *curve*.

    Parameters
    ----------
    initial
        A starting :class:`LayeredModel`, or a layer count to hand to
        :func:`initial_model`.
    smoothing
        Weight on the second difference of log-Vs between adjacent layers.
        Zero fits the data alone; ~0.1-1 suppresses oscillating profiles
        when there are more layers than the data can support.
    invert_thickness
        Also solve for layer thicknesses. Off by default: with a single
        fundamental-mode curve the trade-off between thickness and velocity
        is strong, and a fixed-thickness fit is what students should see
        first.
    """
    if isinstance(initial, int):
        initial = initial_model(curve, initial)
    mode = curve.mode if mode is None else mode
    wave = curve.wave if wave is None else wave

    obs_f = curve.frequency
    obs_v = curve.velocity
    sigma = (
        np.ones_like(obs_v) if curve.velocity_std is None
        else np.maximum(curve.velocity_std, 1e-3)
    )
    # Relative weighting: without formal errors, treat each point equally in
    # log space so the slow, high-frequency part does not vanish under the fast part.
    if curve.velocity_std is None:
        sigma = 0.05 * obs_v

    n = initial.n_layers
    log_vs0 = np.log(initial.vs)
    log_h0 = np.log(initial.thickness)
    x0 = np.concatenate([log_vs0, log_h0]) if invert_thickness else log_vs0
    lb = [np.log(vs_bounds[0])] * n
    ub = [np.log(vs_bounds[1])] * n
    if invert_thickness:
        lb += [np.log(thickness_bounds[0])] * (n - 1)
        ub += [np.log(thickness_bounds[1])] * (n - 1)

    n_eval = {"count": 0}

    def unpack(x: np.ndarray) -> LayeredModel:
        vs = np.exp(x[:n])
        h = np.exp(x[n:]) if invert_thickness else initial.thickness
        return initial.with_vs(vs, h)

    def residual(x: np.ndarray) -> np.ndarray:
        n_eval["count"] += 1
        model = unpack(x)
        try:
            pred = forward_dispersion(model, obs_f, mode=mode, wave=wave)
        except RuntimeError:
            return np.full(obs_f.size + max(n - 2, 0), 1e3)
        # disba may drop frequencies where the mode vanishes; interpolate the
        # found part and penalise the missing part heavily.
        v_pred = np.interp(obs_f, pred.frequency, pred.velocity, left=np.nan, right=np.nan)
        missing = np.isnan(v_pred)
        data_res = np.where(missing, 10.0, (v_pred - obs_v) / sigma)
        if n > 2 and smoothing > 0:
            rough = np.sqrt(smoothing) * np.diff(x[:n], n=2)
        else:
            rough = np.zeros(max(n - 2, 0))
        return np.concatenate([data_res, rough])

    sol = least_squares(
        residual, x0, bounds=(lb, ub), max_nfev=max_evaluations,
        method="trf", x_scale="jac", diff_step=1e-3,
    )
    model = unpack(sol.x)
    predicted = forward_dispersion(model, obs_f, mode=mode, wave=wave)
    v_pred = np.interp(obs_f, predicted.frequency, predicted.velocity)
    rms = float(np.sqrt(np.mean((v_pred - obs_v) ** 2)))
    return InversionResult(
        model=model,
        observed=curve,
        predicted=predicted,
        initial=initial,
        rms=rms,
        n_evaluations=n_eval["count"],
        success=bool(sol.success),
        message=str(sol.message),
        metadata={"smoothing": smoothing, "invert_thickness": invert_thickness,
                  "cost": float(sol.cost)},
    )
