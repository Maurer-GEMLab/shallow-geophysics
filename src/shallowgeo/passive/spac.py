"""Spatial autocorrelation (SPAC) and the checks that say whether it applies.

SPAC (Aki, 1957) reads phase velocity out of ambient noise without knowing
where the noise comes from. Average the complex coherency between two stations
a distance ``r`` apart over enough time, and if the noise arrives from all
azimuths equally the imaginary part averages away and the real part is

.. math:: \\rho(f, r) = J_0\\!\\left(\\frac{2 \\pi f r}{c(f)}\\right)

which inverts for ``c(f)`` one frequency at a time. Okada (2003) is the
standard monograph; Chavez-Garcia et al. (2005) showed it still works with
very few stations, which is what makes it the right choice for a five-node
student deployment where beamforming has nothing to work with.

Two assumptions do the heavy lifting, and both fail quietly:

*The noise must be azimuthally distributed.* If it all comes from one
direction, a pair broadside to it sees no phase difference at all and
:math:`\\rho \\to 1` at every frequency.

*The coherent part must dominate.* Real records are a coherent wavefield plus
whatever each node is doing on its own -- wind on the case, a loose spike.
Writing the coherent fraction as :math:`\\alpha`, what is actually measured is
:math:`\\rho = \\alpha J_0(2 \\pi f r / c)`, and a small :math:`\\alpha`
depresses every separation by the same factor.

Both failures produce a coherency that does not decay with separation, and
both then invert to a phase velocity proportional to ``r`` -- different
station pairs disagreeing by exactly the ratio of their separations. That is
what :func:`separation_test` looks for, and it is the single most useful
number in this module: it is the difference between a dispersion curve and a
plot of the array's own geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import j0

from ..core.survey import SeismicSurvey
from ..surfacewave.dispersion import DispersionCurve
from .deployment import ArrayLayout, array_layout

#: First zero of J0. Past this the Bessel inversion stops being single-valued.
J0_FIRST_ZERO = 2.404825557695773


def konno_ohmachi(frequency: np.ndarray, values: np.ndarray, b: float = 40.0):
    """Konno-Ohmachi smoothing: constant width in log frequency.

    Used in preference to a boxcar because coherency has to be smoothed over
    several octaves at once, and a linear-width window that is reasonable at
    20 Hz erases the entire useful band at 2 Hz.
    """
    out = np.empty_like(values, dtype=float)
    positive = frequency > 0
    for i, centre in enumerate(frequency):
        if centre <= 0:
            out[i] = values[i]
            continue
        ratio = np.ones_like(frequency)
        ratio[positive] = frequency[positive] / centre
        with np.errstate(divide="ignore", invalid="ignore"):
            arg = b * np.log10(ratio)
            weight = np.where(arg == 0, 1.0, (np.sin(arg) / arg) ** 4)
        weight[~positive] = 0.0
        out[i] = np.sum(weight * values) / np.sum(weight)
    return out


@dataclass
class SpacResult:
    """Averaged coherency per station pair.

    ``coherency`` is ``(n_pairs, n_freq)``, real part of the complex
    coherency -- the quantity that equals :math:`J_0` under the SPAC
    assumptions. ``coherency_abs`` and ``phase`` are kept because they are how
    the assumptions get tested: a propagating field on a line array shows a
    phase that grows with frequency, a broadside or common-mode one does not.
    """

    frequency: np.ndarray
    distance: np.ndarray
    coherency: np.ndarray
    coherency_abs: np.ndarray
    phase: np.ndarray
    pairs: pd.DataFrame
    n_windows: int
    n_windows_rejected: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def at(self, frequency: float) -> pd.DataFrame:
        """Coherency of every pair at the nearest frequency bin."""
        k = int(np.argmin(np.abs(self.frequency - frequency)))
        return pd.DataFrame(
            {
                "node_a": self.pairs["node_a"],
                "node_b": self.pairs["node_b"],
                "distance": self.distance,
                "coherency": self.coherency[:, k],
                "abs": self.coherency_abs[:, k],
                "phase_deg": np.degrees(self.phase[:, k]),
            }
        ).sort_values("distance")

    def __repr__(self) -> str:
        return (
            f"SpacResult({len(self.distance)} pairs, "
            f"{self.frequency[0]:.2f}-{self.frequency[-1]:.1f} Hz, "
            f"{self.n_windows} windows kept, {self.n_windows_rejected} rejected)"
        )


def spac_coherency(
    survey: SeismicSurvey,
    *,
    window: float = 30.0,
    overlap: float = 0.5,
    fmin: float = 0.5,
    fmax: float = 60.0,
    anti_trigger: tuple[float, float] | None = (0.3, 3.0),
    smoothing: float | None = 40.0,
    layout: ArrayLayout | None = None,
) -> SpacResult:
    """Average the complex coherency of every station pair over time windows.

    ``anti_trigger`` rejects windows whose RMS on any node falls outside the
    given multiples of that node's median. This is the standard Geopsy-style
    amplitude gate and on real records it is not optional: a handful of
    windows containing a vehicle or a footfall next to one node carry orders
    of magnitude more energy than the ambient field, and because that energy
    is *local to one node* it enters the cross-spectrum as noise while
    dominating both auto-spectra. On the September 2026 test deployment,
    leaving the gate off drove the measured coherency from 0.53 to 0.03 --
    it does not merely add scatter, it erases the signal.

    Pass ``anti_trigger=None`` to keep every window, which is what you want
    when the transients *are* the signal (see :mod:`shallowgeo.passive.events`).
    """
    layout = layout or array_layout(survey)
    fs = survey.sample_rate
    nperseg = int(round(window * fs))
    if nperseg < 16:
        raise ValueError(f"window of {window} s is only {nperseg} samples")
    step = max(1, int(round(nperseg * (1.0 - overlap))))

    data = survey.data
    starts = np.arange(0, data.shape[1] - nperseg + 1, step)
    if starts.size == 0:
        raise ValueError(
            f"record is {survey.duration:.0f} s, shorter than one {window:g} s window"
        )

    segments = np.stack([data[:, s : s + nperseg] for s in starts], axis=1)
    finite = np.isfinite(segments).all(axis=(0, 2))
    rms = np.zeros(segments.shape[:2])
    rms[:, finite] = segments[:, finite, :].std(axis=2)

    keep = finite & (rms > 0).all(axis=0)
    if anti_trigger is not None:
        lo, hi = anti_trigger
        median = np.median(np.where(rms > 0, rms, np.nan), axis=1, keepdims=True)
        keep &= ((rms > lo * median) & (rms < hi * median)).all(axis=0)
    if keep.sum() < 5:
        raise ValueError(
            f"only {keep.sum()} windows survived the amplitude gate; "
            "loosen anti_trigger or use a shorter window"
        )

    taper = np.hanning(nperseg)
    kept = segments[:, keep, :]
    kept = (kept - kept.mean(axis=2, keepdims=True)) * taper
    spectra = np.fft.rfft(kept, axis=2)
    freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
    band = (freqs >= fmin) & (freqs <= fmax)
    spectra, freqs = spectra[:, :, band], freqs[band]

    power = np.mean(np.abs(spectra) ** 2, axis=1)
    pairs = layout.separations
    real, absolute, phase = [], [], []
    for _, row in pairs.iterrows():
        i, j = int(row["index_a"]), int(row["index_b"])
        cross = np.mean(spectra[i] * np.conj(spectra[j]), axis=0)
        gamma = cross / np.sqrt(power[i] * power[j] + 1e-30)
        rho = np.real(gamma)
        if smoothing:
            rho = konno_ohmachi(freqs, rho, smoothing)
        real.append(rho)
        absolute.append(np.abs(gamma))
        phase.append(np.angle(gamma))

    return SpacResult(
        frequency=freqs,
        distance=pairs["distance"].to_numpy(float),
        coherency=np.array(real),
        coherency_abs=np.array(absolute),
        phase=np.array(phase),
        pairs=pairs.reset_index(drop=True),
        n_windows=int(keep.sum()),
        n_windows_rejected=int((~keep).sum()),
        metadata={
            "window": window,
            "overlap": overlap,
            "anti_trigger": anti_trigger,
            "smoothing": smoothing,
        },
    )


# -- validity ---------------------------------------------------------------


@dataclass
class SeparationTest:
    """Does the coherency actually decay with station separation?

    ``slope`` regresses coherency on separation, per frequency, over the pairs
    where :math:`J_0` should still be falling. A propagating wavefield gives a
    clearly negative slope. A slope near zero means the coherency is the same
    at 1 m and 50 m, which no travelling wave can produce.

    ``velocity_ratio`` is the sharper test. It inverts each pair separately
    and takes the spread of the answers. Under the failure modes in the module
    docstring the inverted velocity is exactly proportional to ``r``, so this
    ratio reproduces the ratio of the largest to smallest separation
    (``geometry_ratio``) instead of collapsing towards 1.
    """

    frequency: np.ndarray
    slope: np.ndarray
    velocity_ratio: np.ndarray
    geometry_ratio: float
    verdict: str
    usable: np.ndarray

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "frequency": self.frequency,
                "slope_per_m": self.slope,
                "velocity_ratio": self.velocity_ratio,
                "usable": self.usable,
            }
        )

    def __repr__(self) -> str:
        return f"SeparationTest({self.verdict}, {int(self.usable.sum())} usable bins)"


def separation_rings(distances: np.ndarray, *, tolerance: float = 0.05) -> np.ndarray:
    """Label pairs that share a separation, to within ``tolerance`` relatively.

    A circular array has four pairs at the same radius. They are four samples
    of one measurement, not four independent ones, and treating them as
    independent is what makes the agreement test below vacuous.
    """
    order = np.argsort(distances)
    label = np.zeros(distances.size, dtype=int)
    current, reference = 0, distances[order[0]]
    for index in order[1:]:
        if distances[index] > reference * (1.0 + tolerance):
            current += 1
            reference = distances[index]
        label[index] = current
    label[order[0]] = 0
    return label


def separation_test(
    result: SpacResult, *, tolerance: float = 0.35, ring_tolerance: float = 0.05
) -> SeparationTest:
    """Test the SPAC assumptions against the measured coherency.

    A frequency is usable when the coherency falls with separation *and* two
    or more distinct separations invert to the same velocity.

    Requiring **distinct separations**, rather than simply two pairs, is what
    makes the test mean anything. Past the first zero of :math:`J_0` the
    coherency comes back up into the second lobe, and inverting it on the
    first lobe returns a velocity that is badly wrong but perfectly stable --
    every pair at that radius agrees with every other, because they are all
    measuring the same thing incorrectly. Only a *different* separation, whose
    own second lobe falls elsewhere, disagrees and gives the error away.

    So a frequency at which only one ring of the array still inverts is
    reported unusable, however tidy its numbers look.
    """
    distances = result.distance
    if distances.size < 2:
        raise ValueError("need at least two station pairs to test separation decay")
    geometry_ratio = float(distances.max() / distances.min())
    rings = separation_rings(distances, tolerance=ring_tolerance)
    n_rings = int(rings.max()) + 1

    slopes, ratios = [], []
    for k in range(result.frequency.size):
        rho = result.coherency[:, k]
        good = np.isfinite(rho)
        if good.sum() >= 2 and np.ptp(distances[good]) > 0:
            slopes.append(float(np.polyfit(distances[good], rho[good], 1)[0]))
        else:
            slopes.append(np.nan)

        # One velocity per ring, then compare rings against each other.
        per_ring = []
        for ring in range(n_rings):
            members = rings == ring
            values = [
                _invert_bessel(result.frequency[k], value, r)
                for r, value in zip(distances[members], rho[members], strict=True)
            ]
            values = [v for v in values if np.isfinite(v)]
            if values:
                per_ring.append(float(np.median(values)))
        if len(per_ring) >= 2:
            per_ring = np.array(per_ring)
            ratios.append(float(np.std(per_ring) / np.mean(per_ring)))
        else:
            ratios.append(np.nan)

    slopes, ratios = np.array(slopes), np.array(ratios)
    usable = (slopes < 0) & (ratios < tolerance)

    fraction = float(np.nanmean(usable)) if usable.size else 0.0
    if fraction > 0.5:
        verdict = "SPAC assumptions hold over most of the band"
    elif fraction > 0.15:
        verdict = "SPAC valid only in part of the band -- inspect before inverting"
    else:
        verdict = (
            "SPAC assumptions fail: coherency does not decay with separation, so "
            "the inverted velocity tracks station spacing rather than the ground"
        )
    return SeparationTest(
        frequency=result.frequency,
        slope=slopes,
        velocity_ratio=ratios,
        geometry_ratio=geometry_ratio,
        verdict=verdict,
        usable=usable,
    )


@dataclass
class WavelengthTest:
    """Does the inverted curve describe a wave, or just the array?

    A dispersion curve is a statement about wavelength. Over a band spanning a
    factor ``F`` in frequency, a site with constant velocity gives wavelengths
    spanning the same factor ``F``, and a normally dispersive one -- velocity
    falling as frequency rises -- spans more. ``index`` is
    :math:`\\log(\\Lambda) / \\log(F)` for measured wavelength span
    :math:`\\Lambda`, so it is 1 for a uniform half-space and larger for
    normal dispersion.

    An index near zero means the wavelength barely moved while the frequency
    changed by nearly an order of magnitude. No medium does that. It is what
    comes back when the measured coherency is flat in frequency -- a partial
    common-mode correlation between nodes rather than a travelling wave --
    because inverting a constant :math:`\\rho` against
    :math:`J_0(2 \\pi f r / c)` forces ``c`` to rise in step with ``f`` and
    pins the wavelength at whatever value that constant implies.

    The resulting curve looks entirely respectable on a velocity-frequency
    plot, rising smoothly, which is why this is worth testing explicitly
    rather than eyeballing.
    """

    index: float
    wavelength_span: float
    frequency_span: float
    wavelength_min: float
    wavelength_max: float
    plausible: bool
    verdict: str

    def __repr__(self) -> str:
        return f"WavelengthTest(index {self.index:.2f}, {self.verdict})"


def wavelength_test(curve: DispersionCurve, *, threshold: float = 0.5) -> WavelengthTest:
    """Check an inverted curve for a pinned wavelength.

    ``threshold`` is the lowest index treated as physical. Below roughly 0.5
    the wavelength is moving less than half as fast as it would over a uniform
    half-space, which no realistic velocity profile produces.
    """
    f, lam = curve.frequency, curve.wavelength
    good = np.isfinite(f) & np.isfinite(lam) & (f > 0) & (lam > 0)
    if good.sum() < 3:
        raise ValueError("need at least three points to test for wavelength pinning")
    f, lam = f[good], lam[good]

    frequency_span = float(f.max() / f.min())
    wavelength_span = float(lam.max() / lam.min())
    # Over a narrow band every curve looks flat, so the ratio of logarithms is
    # dominated by rounding rather than by the ground. Refuse rather than
    # return a confident-looking number from a two-octave-free curve.
    if frequency_span < 1.5:
        raise ValueError(
            f"curve spans only a factor of {frequency_span:.2f} in frequency; "
            "at least 1.5 is needed to tell dispersion from a flat coherency. "
            "Widen the picked band."
        )
    index = float(np.log(wavelength_span) / np.log(frequency_span))

    plausible = index >= threshold
    if plausible:
        verdict = "wavelength varies with frequency as a real dispersion curve should"
    else:
        verdict = (
            f"wavelength is pinned near {np.median(lam):.0f} m while frequency spans "
            f"a factor of {frequency_span:.0f}; this is a flat coherency inverted "
            "through J0, not a propagating wave"
        )
    return WavelengthTest(
        index=index,
        wavelength_span=wavelength_span,
        frequency_span=frequency_span,
        wavelength_min=float(lam.min()),
        wavelength_max=float(lam.max()),
        plausible=plausible,
        verdict=verdict,
    )


def _invert_bessel(frequency: float, rho: float, distance: float) -> float:
    """Solve ``J0(2 pi f r / c) = rho`` on the first lobe, or return NaN.

    Restricted to the first lobe because past the first zero the equation has
    many roots and nothing in the data chooses between them.
    """
    if frequency <= 0 or distance <= 0 or not np.isfinite(rho):
        return np.nan
    if not (j0(J0_FIRST_ZERO) < rho < 0.999):
        return np.nan
    try:
        argument = brentq(lambda x: j0(x) - rho, 1e-9, J0_FIRST_ZERO)
    except ValueError:
        return np.nan
    return 2.0 * np.pi * frequency * distance / argument


def _fit_bessel(
    frequency: float,
    distances: np.ndarray,
    rho: np.ndarray,
    velocities: np.ndarray,
) -> tuple[float, float, float]:
    r"""Fit ``rho(r) = A * J0(2 pi f r / c)`` over all separations at once.

    Returns ``(c, A, residual)``.

    Fitting the whole curve is strictly better than inverting each separation
    on its own, for two reasons.

    A real record is a coherent wavefield plus whatever each station is doing
    by itself, so what comes back is :math:`\\alpha J_0`, not :math:`J_0`.
    Inverting point by point has no way to know that and charges the whole
    deficit to the velocity; fitting carries ``A`` as a free parameter and the
    velocity comes out unbiased.

    And because the fit sees the *shape* of the decay across several radii, it
    is not confined to the first lobe. A single separation past the first zero
    is ambiguous -- second-lobe coherency inverts to a plausible wrong answer
    -- but the pattern across radii is not, so the usable band extends well
    past where point-wise inversion has to stop.

    ``A`` is solved in closed form for each trial velocity, which leaves a
    one-dimensional search and makes this cheap enough to run per frequency.
    """
    argument = 2.0 * np.pi * frequency * distances[None, :] / velocities[:, None]
    basis = j0(argument)                                  # (n_vel, n_ring)
    denominator = np.sum(basis * basis, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        amplitude = np.where(
            denominator > 1e-12, np.sum(basis * rho[None, :], axis=1) / denominator, 0.0
        )
    amplitude = np.clip(amplitude, 0.0, 1.0)
    residual = np.sqrt(
        np.mean((amplitude[:, None] * basis - rho[None, :]) ** 2, axis=1)
    )
    best = int(np.argmin(residual))
    return float(velocities[best]), float(amplitude[best]), float(residual[best])


def spac_dispersion(
    result: SpacResult,
    *,
    test: SeparationTest | None = None,
    only_usable: bool = True,
    max_r_over_lambda: float = 0.5,
    method: str = "fit",
    vmin: float = 50.0,
    vmax: float = 1200.0,
    dv: float = 2.0,
    min_rings: int = 3,
    max_residual: float = 0.15,
    min_aperture_ratio: float = 0.3,
    max_aliasing_ratio: float = 0.9,
) -> DispersionCurve:
    """Invert averaged coherency to a phase-velocity curve.

    Each pair is inverted independently and the frequency's value is the
    median across pairs, with the spread carried through as ``velocity_std``.
    That spread is the honest uncertainty here: when the pairs disagree the
    SPAC model does not fit, and a mean would hide it.

    With ``only_usable`` the frequencies that :func:`separation_test` rejected
    are dropped rather than returned with a large error bar, because their
    error is systematic rather than random.
    """
    if method not in {"fit", "pointwise"}:
        raise ValueError(f"method must be 'fit' or 'pointwise', got {method!r}")
    test = test if test is not None else separation_test(result)
    rings = separation_rings(result.distance)
    n_rings = int(rings.max()) + 1
    # Pairs sharing a separation are repeats of one measurement, so they are
    # averaged into a single ring value before anything else happens.
    ring_distance = np.array(
        [result.distance[rings == ring].mean() for ring in range(n_rings)]
    )
    trial = np.arange(vmin, vmax + dv / 2, dv)
    frequencies, velocities, spreads, amplitudes = [], [], [], []

    for k, f in enumerate(result.frequency):
        if only_usable and not test.usable[k]:
            continue
        ring_rho = np.array(
            [result.coherency[rings == ring, k].mean() for ring in range(n_rings)]
        )
        if not np.isfinite(ring_rho).all():
            continue

        if method == "fit":
            if n_rings < min_rings:
                raise ValueError(
                    f"the Bessel fit needs at least {min_rings} distinct station "
                    f"separations to solve for velocity and coherent fraction "
                    f"together; this array has {n_rings}. Use method='pointwise' "
                    "to invert each separation on its own, and treat the result "
                    "with corresponding suspicion."
                )
            c, amplitude, residual = _fit_bessel(f, ring_distance, ring_rho, trial)
            if residual > max_residual or amplitude <= 0.05:
                continue
            if c <= vmin + dv or c >= vmax - dv:      # pinned to the search edge
                continue
            # The free amplitude that makes this fit robust to incoherent
            # noise also makes it able to absorb a *flat* coherency: pick a
            # large enough velocity and J0 is near-constant across the array,
            # so A alone matches the data and the velocity means nothing. Two
            # physical bounds keep that from being reported as a measurement.
            if ring_distance.max() * f / c < min_aperture_ratio:
                continue        # wavelength too long for this array to resolve
            if ring_distance.min() * f / c > max_aliasing_ratio:
                # The default 0.9 is the second zero of J0 at r/lambda = 0.878:
                # beyond it even the closest pair is more than two lobes out,
                # and there is not enough Bessel shape left to pin the lobe.
                continue
            frequencies.append(f)
            velocities.append(c)
            spreads.append(residual * c)             # residual scaled to m/s
            amplitudes.append(amplitude)
            continue

        per_ring = []
        for ring in range(n_rings):
            members = rings == ring
            values = []
            for r, rho in zip(
                result.distance[members], result.coherency[members, k], strict=True
            ):
                value = _invert_bessel(f, rho, r)
                if np.isfinite(value) and r * f / value <= max_r_over_lambda:
                    values.append(value)
            if values:
                per_ring.append(float(np.median(values)))
        if len(per_ring) < 2:
            continue
        per_ring = np.asarray(per_ring)
        frequencies.append(f)
        velocities.append(float(np.median(per_ring)))
        spreads.append(float(np.std(per_ring)))

    if not frequencies:
        raise ValueError(
            "no frequency passed the SPAC validity checks, so no dispersion "
            f"curve can be extracted. {test.verdict}."
        )
    curve = DispersionCurve(
        frequency=np.array(frequencies),
        velocity=np.array(velocities),
        velocity_std=np.array(spreads),
        mode=0,
        wave="rayleigh",
        metadata={
            "method": f"spac_{method}",
            "n_windows": result.n_windows,
            "coherent_fraction": (
                float(np.median(amplitudes)) if amplitudes else None
            ),
            "separation_verdict": test.verdict,
        },
    )
    # The separation test can pass on a curve that is still an artefact, so
    # the pinning check runs here too rather than waiting to be called.
    try:
        pinning = wavelength_test(curve)
        curve.metadata["wavelength_index"] = pinning.index
        curve.metadata["wavelength_verdict"] = pinning.verdict
        curve.metadata["plausible"] = pinning.plausible
    except ValueError:
        curve.metadata["plausible"] = None
    return curve
