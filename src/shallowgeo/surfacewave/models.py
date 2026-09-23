"""1D layered earth for surface-wave modelling.

Forward dispersion is delegated to ``disba`` (Luu, 2021), a numba port of
Herrmann's Computer Programs in Seismology. It is the mature solver for this
problem and matches the project's rule of wrapping, not reimplementing.
``disba`` is an optional extra so the readers stay installable without numba.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .dispersion import DispersionCurve

#: Vp/Vs for a Poisson solid; the default when no Vp is known.
DEFAULT_VP_VS = 3.0 ** 0.5
DEFAULT_DENSITY = 1900.0  # kg/m^3, a soil-to-weathered-rock middle value


def _need_disba():
    try:
        import disba  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "surface-wave forward modelling needs disba: "
            "pip install 'shallow-geophysics[masw]'"
        ) from exc
    return disba


@dataclass
class LayeredModel:
    """Horizontal layers over a half-space.

    Parameters
    ----------
    thickness
        Thickness of each layer *above* the half-space, metres. Length
        ``n_layers - 1``.
    vs
        Shear velocity per layer including the half-space, m/s. Length
        ``n_layers``.
    vp, density
        Optional, same length as ``vs``. Filled from ``vp_vs`` and a
        constant density when omitted; Rayleigh dispersion is weakly
        sensitive to both, which is why MASW resolves Vs and little else.
    """

    thickness: np.ndarray
    vs: np.ndarray
    vp: np.ndarray | None = None
    density: np.ndarray | None = None
    vp_vs: float = DEFAULT_VP_VS
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.thickness = np.atleast_1d(np.asarray(self.thickness, dtype=float))
        self.vs = np.atleast_1d(np.asarray(self.vs, dtype=float))
        n = self.vs.size
        if self.thickness.size == n:
            # Tolerate a half-space "thickness" entry and drop it.
            self.thickness = self.thickness[:-1]
        if self.thickness.size != n - 1:
            raise ValueError(
                f"thickness has {self.thickness.size} entries; expected "
                f"{n - 1} for {n} layers (the half-space has none)"
            )
        if np.any(self.thickness <= 0):
            raise ValueError("layer thicknesses must be positive")
        if np.any(self.vs <= 0):
            raise ValueError("shear velocities must be positive")
        self.vp = (
            self.vs * self.vp_vs if self.vp is None
            else np.atleast_1d(np.asarray(self.vp, dtype=float))
        )
        self.density = (
            np.full(n, DEFAULT_DENSITY) if self.density is None
            else np.atleast_1d(np.asarray(self.density, dtype=float))
        )
        if self.vp.size != n or self.density.size != n:
            raise ValueError("vp and density must have one entry per layer")

    @property
    def n_layers(self) -> int:
        return self.vs.size

    @property
    def depth_top(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(self.thickness)])

    @property
    def depth_bottom(self) -> np.ndarray:
        """Bottom of each layer; the half-space gets ``inf``."""
        return np.concatenate([np.cumsum(self.thickness), [np.inf]])

    def to_disba(self) -> np.ndarray:
        """``(n_layers, 4)`` array in disba units: km, km/s, km/s, g/cm^3."""
        thick_km = np.concatenate([self.thickness, [0.0]]) / 1000.0
        return np.column_stack(
            [thick_km, self.vp / 1000.0, self.vs / 1000.0, self.density / 1000.0]
        )

    def profile(self, zmax: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Step arrays ``(depth, vs)`` for plotting, down to ``zmax``."""
        zmax = zmax or 1.5 * self.depth_top[-1] + 5.0
        z = np.repeat(np.concatenate([self.depth_top, [zmax]]), 2)[1:-1]
        v = np.repeat(self.vs, 2)
        return z, v

    def with_vs(self, vs: np.ndarray, thickness: np.ndarray | None = None) -> LayeredModel:
        return LayeredModel(
            self.thickness if thickness is None else thickness,
            vs,
            vp=None if self.metadata.get("vp_from_ratio", True) else self.vp,
            density=self.density,
            vp_vs=self.vp_vs,
            metadata=dict(self.metadata),
        )

    def plot(self, ax=None, *, zmax: float | None = None, **kwargs):
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        z, v = self.profile(zmax)
        ax.plot(v, z, **kwargs)
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.set_xlabel("Vs (m/s)")
        ax.set_ylabel("depth (m)")
        return ax

    def __repr__(self) -> str:
        vs = ", ".join(f"{v:.0f}" for v in self.vs)
        h = ", ".join(f"{t:.1f}" for t in self.thickness)
        return f"<LayeredModel vs=[{vs}] m/s thickness=[{h}] m>"


def forward_dispersion(
    model: LayeredModel,
    frequencies: np.ndarray,
    *,
    mode: int = 0,
    wave: str = "rayleigh",
    algorithm: str = "dunkin",
    dc: float = 0.005,
) -> DispersionCurve:
    """Phase-velocity dispersion curve of *model* at *frequencies*.

    Frequencies at which the requested mode does not exist are dropped from
    the returned curve, so its length can be shorter than the input. Callers
    that need a fixed length (the inversion) should compare on the returned
    ``frequency`` array.
    """
    disba = _need_disba()
    f = np.sort(np.asarray(frequencies, dtype=float))
    f = f[f > 0]
    periods = np.sort(1.0 / f)
    pd_ = disba.PhaseDispersion(*model.to_disba().T, algorithm=algorithm, dc=dc)
    try:
        result = pd_(periods, mode=mode, wave=wave)
    except Exception as exc:  # disba raises a bare Exception on failure
        raise RuntimeError(f"disba failed on {model!r}: {exc}") from exc
    if result.period.size == 0:
        raise RuntimeError(f"mode {mode} not found at any requested frequency")
    return DispersionCurve(
        1.0 / result.period,
        result.velocity * 1000.0,
        mode=mode,
        wave=wave,
        metadata={"model": model, "algorithm": algorithm},
    )
