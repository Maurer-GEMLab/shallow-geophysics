"""Dispersion imaging and curve extraction.

The phase-shift method (Park, Miller & Xia, 1998) is used rather than f-k or
tau-p because it needs no regular receiver spacing, handles the short spreads
typical of engineering surveys gracefully, and is the method students will
meet in the commercial packages. For each frequency the trace spectra are
amplitude-normalised, phase-shifted by the travel time a wave of trial
velocity ``c`` would need to reach each offset, and summed: energy stacks
coherently only at the true phase velocity.

Resolution limits are made explicit through :func:`spread_wavelength_limits`
so a picked curve can be clipped to what the array can actually resolve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DispersionCurve:
    """Phase velocity against frequency for one mode.

    ``velocity_std`` is an optional per-point uncertainty. When it comes from
    :meth:`DispersionImage.pick` it is the half-width of the energy peak,
    which is the honest resolution of the image rather than a formal error.
    """

    frequency: np.ndarray
    velocity: np.ndarray
    velocity_std: np.ndarray | None = None
    mode: int = 0
    wave: str = "rayleigh"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frequency = np.asarray(self.frequency, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
        if self.frequency.shape != self.velocity.shape:
            raise ValueError("frequency and velocity must have the same shape")
        if self.velocity_std is not None:
            self.velocity_std = np.asarray(self.velocity_std, dtype=float)
        order = np.argsort(self.frequency)
        self.frequency = self.frequency[order]
        self.velocity = self.velocity[order]
        if self.velocity_std is not None:
            self.velocity_std = self.velocity_std[order]

    @property
    def n_points(self) -> int:
        return self.frequency.size

    @property
    def wavelength(self) -> np.ndarray:
        return self.velocity / self.frequency

    @property
    def period(self) -> np.ndarray:
        return 1.0 / self.frequency

    def clip_wavelength(self, lambda_min: float, lambda_max: float) -> DispersionCurve:
        """Keep only points the spread can resolve (see :func:`spread_wavelength_limits`)."""
        keep = (self.wavelength >= lambda_min) & (self.wavelength <= lambda_max)
        return self._subset(keep, clipped_to=(lambda_min, lambda_max))

    def clip_frequency(self, fmin: float, fmax: float) -> DispersionCurve:
        keep = (self.frequency >= fmin) & (self.frequency <= fmax)
        return self._subset(keep)

    def _subset(self, keep: np.ndarray, **meta) -> DispersionCurve:
        return DispersionCurve(
            self.frequency[keep],
            self.velocity[keep],
            None if self.velocity_std is None else self.velocity_std[keep],
            mode=self.mode,
            wave=self.wave,
            metadata={**self.metadata, **meta},
        )

    def resample(self, frequencies: np.ndarray) -> DispersionCurve:
        """Linear interpolation onto new frequencies, within the curve's range."""
        f = np.asarray(frequencies, dtype=float)
        inside = (f >= self.frequency.min()) & (f <= self.frequency.max())
        f = f[inside]
        v = np.interp(f, self.frequency, self.velocity)
        std = None
        if self.velocity_std is not None:
            std = np.interp(f, self.frequency, self.velocity_std)
        return DispersionCurve(f, v, std, mode=self.mode, wave=self.wave,
                               metadata=dict(self.metadata))

    def to_dataframe(self) -> pd.DataFrame:
        out = pd.DataFrame({"frequency": self.frequency, "velocity": self.velocity,
                            "wavelength": self.wavelength})
        if self.velocity_std is not None:
            out["velocity_std"] = self.velocity_std
        return out

    @classmethod
    def from_dataframe(cls, frame: pd.DataFrame, **kwargs) -> DispersionCurve:
        std = frame["velocity_std"].to_numpy() if "velocity_std" in frame else None
        return cls(frame["frequency"].to_numpy(), frame["velocity"].to_numpy(), std, **kwargs)

    def plot(self, ax=None, *, x="frequency", **kwargs):
        """Quick look; ``x`` may be ``"frequency"`` or ``"wavelength"``."""
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        xs = self.frequency if x == "frequency" else self.wavelength
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("ms", 3)
        if self.velocity_std is not None:
            ax.errorbar(xs, self.velocity, yerr=self.velocity_std, ls="none", **kwargs)
        else:
            ax.plot(xs, self.velocity, ls="none", **kwargs)
        ax.set_xlabel("frequency (Hz)" if x == "frequency" else "wavelength (m)")
        ax.set_ylabel("phase velocity (m/s)")
        return ax

    def __repr__(self) -> str:
        return (f"<DispersionCurve {self.wave} mode={self.mode} n={self.n_points} "
                f"f={self.frequency.min():.1f}-{self.frequency.max():.1f} Hz>")


def spread_wavelength_limits(offsets: np.ndarray) -> tuple[float, float]:
    """Rule-of-thumb resolvable wavelength range for a linear spread.

    Shortest: twice the receiver spacing (spatial Nyquist). Longest: the
    spread length -- a wavelength longer than the array cannot be measured
    reliably, and several practitioners argue for half that. Both are guides,
    not laws; the point is to make students state them.
    """
    x = np.sort(np.asarray(offsets, dtype=float))
    dx = np.median(np.diff(x))
    return float(2.0 * dx), float(x[-1] - x[0])


@dataclass
class DispersionImage:
    """Normalised energy on a (frequency, phase velocity) grid."""

    frequency: np.ndarray
    velocity: np.ndarray
    image: np.ndarray
    offsets: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.image.shape != (self.frequency.size, self.velocity.size):
            raise ValueError("image must be (n_frequencies, n_velocities)")

    def pick(
        self,
        *,
        fmin: float | None = None,
        fmax: float | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        method: str = "track",
        max_jump: float = 0.15,
        min_quality: float = 0.0,
        mode: int = 0,
    ) -> DispersionCurve:
        """Extract the fundamental-mode ridge.

        Parameters
        ----------
        method
            ``"max"`` takes the energy maximum at every frequency
            independently. ``"track"`` (default) starts at the frequency with
            the sharpest peak and follows the ridge outwards, never letting
            the velocity jump more than ``max_jump`` (fraction) between
            adjacent frequencies -- which is what keeps the pick from hopping
            to a higher mode or to aliasing energy at the band edges.
        min_quality
            Drop points whose peak is less than this many times the mean
            energy at that frequency. Around 2-3 is a sensible floor.
        """
        f, v, img = self.frequency, self.velocity, self.image
        fsel = np.ones(f.size, bool)
        if fmin is not None:
            fsel &= f >= fmin
        if fmax is not None:
            fsel &= f <= fmax
        vsel = np.ones(v.size, bool)
        if vmin is not None:
            vsel &= v >= vmin
        if vmax is not None:
            vsel &= v <= vmax
        if not fsel.any() or vsel.sum() < 3:
            raise ValueError("pick window contains no image samples")

        fi = np.flatnonzero(fsel)
        vi = np.flatnonzero(vsel)
        sub = img[np.ix_(fi, vi)]
        vv = v[vi]

        # Peak sharpness: peak over row mean. Used both as the quality metric
        # and to choose where ridge tracking starts.
        row_mean = sub.mean(axis=1) + 1e-30
        quality = sub.max(axis=1) / row_mean

        if method == "max":
            idx = sub.argmax(axis=1)
        elif method == "track":
            idx = np.empty(fi.size, dtype=int)
            start = int(np.argmax(quality))
            idx[start] = int(sub[start].argmax())
            for direction in (1, -1):
                prev = idx[start]
                rng = range(start + 1, fi.size) if direction > 0 else range(start - 1, -1, -1)
                for k in rng:
                    lo, hi = vv[prev] * (1 - max_jump), vv[prev] * (1 + max_jump)
                    window = (vv >= lo) & (vv <= hi)
                    if not window.any():
                        window[prev] = True
                    cand = np.where(window, sub[k], -np.inf)
                    idx[k] = int(cand.argmax())
                    prev = idx[k]
        else:
            raise ValueError(f"unknown pick method {method!r}")

        picked_v = vv[idx]
        std = _peak_halfwidth(sub, idx, vv)
        keep = quality >= min_quality
        return DispersionCurve(
            f[fi][keep], picked_v[keep], std[keep], mode=mode,
            metadata={"quality": quality[keep], "method": method,
                      "offsets": self.offsets},
        )

    def plot(self, ax=None, *, curve: DispersionCurve | None = None, **kwargs):
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        kwargs.setdefault("cmap", "viridis")
        kwargs.setdefault("shading", "auto")
        ax.pcolormesh(self.frequency, self.velocity, self.image.T, **kwargs)
        if curve is not None:
            ax.plot(curve.frequency, curve.velocity, "w.", ms=4)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("phase velocity (m/s)")
        return ax


def _peak_halfwidth(sub: np.ndarray, idx: np.ndarray, vv: np.ndarray) -> np.ndarray:
    """Half-width at half-maximum of each row's peak, in velocity units."""
    out = np.empty(idx.size)
    for k, j in enumerate(idx):
        row = sub[k]
        half = 0.5 * row[j]
        lo = j
        while lo > 0 and row[lo] > half:
            lo -= 1
        hi = j
        while hi < row.size - 1 and row[hi] > half:
            hi += 1
        out[k] = 0.5 * (vv[hi] - vv[lo])
    return out


def phase_shift(
    data: np.ndarray,
    offsets: np.ndarray,
    sample_interval: float,
    *,
    fmin: float = 2.0,
    fmax: float = 100.0,
    vmin: float = 50.0,
    vmax: float = 1500.0,
    dv: float = 5.0,
    nfft: int | None = None,
    normalize: bool = True,
) -> DispersionImage:
    """Phase-shift transform of a shot gather.

    Parameters
    ----------
    data
        ``(n_traces, n_samples)``.
    offsets
        Absolute source-receiver distance per trace, metres. Need not be
        sorted or regular.
    nfft
        Zero-pad the FFT to this length for a finer frequency axis. Defaults
        to the next power of two at least four times the record length,
        because refraction-length records (100-200 ms) otherwise give a
        frequency step of 5-10 Hz, far too coarse to pick.
    normalize
        Scale each frequency row to unit maximum, the usual display.
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    x = np.asarray(offsets, dtype=float)
    if x.size != data.shape[0]:
        raise ValueError("offsets must have one entry per trace")
    if np.ptp(x) == 0:
        raise ValueError("all offsets are equal; the gather has no geometry")

    n_samples = data.shape[1]
    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(4 * n_samples)))
    freqs_all = np.fft.rfftfreq(nfft, d=sample_interval)
    spectra = np.fft.rfft(data, n=nfft, axis=1)

    fsel = (freqs_all >= fmin) & (freqs_all <= fmax)
    if not fsel.any():
        raise ValueError("no FFT bins in the requested frequency band")
    f = freqs_all[fsel]
    U = spectra[:, fsel]
    amp = np.abs(U)
    U = np.where(amp > 0, U / np.where(amp > 0, amp, 1.0), 0.0)

    velocities = np.arange(vmin, vmax + dv / 2, dv)
    # phase term: exp(+i 2 pi f x / c) undoes the propagation delay x / c
    # shape (n_freq, n_vel, n_traces) is small for engineering gathers.
    k = 2.0 * np.pi * f[:, None, None] / velocities[None, :, None]
    phase = np.exp(1j * k * x[None, None, :])
    image = np.abs(np.einsum("fvx,xf->fv", phase, U)) / x.size

    if normalize:
        rowmax = image.max(axis=1, keepdims=True)
        image = image / np.where(rowmax > 0, rowmax, 1.0)

    return DispersionImage(
        frequency=f, velocity=velocities, image=image, offsets=x,
        metadata={"method": "phase_shift", "nfft": nfft,
                  "sample_interval": sample_interval},
    )


def dispersion_image(survey, **kwargs) -> DispersionImage:
    """:func:`phase_shift` applied to a ``SeismicSurvey`` shot gather.

    Uses the survey's own offsets, so the geometry checks (units, source
    position) have already happened in the driver.
    """
    if survey.is_passive:
        raise ValueError("dispersion_image needs an active-source gather")
    img = phase_shift(survey.data, survey.offsets(), survey.sample_interval, **kwargs)
    img.metadata["source_file"] = survey.metadata.get("source_file")
    img.metadata["wavelength_limits"] = spread_wavelength_limits(survey.offsets())
    # Below about two cycles per record the transform sees a truncated wave
    # and the low-frequency picks are not trustworthy. A 128 ms refraction
    # record puts this floor near 16 Hz.
    img.metadata["frequency_floor"] = 2.0 / survey.duration
    return img
