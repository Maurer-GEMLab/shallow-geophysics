"""Plane-wave fitting: velocity *and* direction from inter-node phase.

SPAC needs the noise to arrive from every azimuth, because the azimuthal
average is what turns the coherency into :math:`J_0`. At a quiet site with one
road, one quarry or one pump, that average never happens. Here the dominant
direction is the assumption rather than the problem: fit one plane wave

.. math:: \\gamma_{ij}(f) \\approx e^{-2 \\pi i f \\, \\mathbf{s} \\cdot (\\mathbf{x}_i - \\mathbf{x}_j)}

to the complex coherency of every station pair, one frequency at a time, and
read the slowness vector :math:`\\mathbf{s}` -- speed and direction together.
A 2-D array of five nodes has ten pairs pointing in several directions, which
over-determines the two unknowns, so the fit can also say how well a single
plane wave explains the data.

The result is a bracket, not a point. What limits it is how much phase the
wave puts across the array. A wavelength several times the aperture puts only
a few degrees across even the longest pair, and position error alone is
worth that much: 1.5 m of uncorrected GPS error on a 30 m pair at 10 Hz and
2 km/s is 3 degrees. So every frequency comes back with ``velocity_min`` and
``velocity_max``, the extremes of the region of slowness space that fits the
phases as well as the noise allows. A ``velocity_max`` of infinity means the
data cannot tell the wave from one arriving at every node at once -- the
velocity is bounded below and that is all. That is the normal state of
affairs on a compact array over fast ground, and it is the reason this
function reports brackets: the alternative is a single number that looks
like a measurement.

Two related diagnostics come out of the same fit. ``single_wave`` is the
fraction of the measured coherency that one plane wave accounts for; near 1,
the field is directional and SPAC's azimuthal average does not hold, however
clean :func:`~shallowgeo.passive.separation_test` looks. ``aliased`` flags a
frequency where the misfit has more than one acceptable minimum, because the
station spacing lets a second slowness explain the phases too.

Every velocity here is *apparent*: the speed at which the wavefront sweeps
across the ground. A wave arriving steeply from below sweeps across at
V/sin(i), faster than it travels, so when body waves contribute the result
is an upper bound on the horizontal surface-wave velocity. At site A in
October 2026 the fit returned more than the rock's own P-wave velocity,
which no surface wave can do (see ``docs/atom-1c-passive-2026.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage

from ..core.survey import SeismicSurvey
from ..surfacewave.dispersion import DispersionCurve
from .deployment import ArrayLayout, array_layout
from .events import EventWindow
from .spac import window_spectra

#: chi-squared increase bounding a joint 68 % region for two parameters.
DELTA_CHI2_68 = 2.30


@dataclass
class PlaneWaveFit:
    """Per-frequency plane-wave fit; ``table`` holds one row per frequency.

    Columns:

    ``velocity``, ``velocity_min``, ``velocity_max``
        Best-fit phase velocity and the 68 % bracket, m/s.
        ``velocity_max`` is ``inf`` when zero slowness fits.
    ``azimuth``, ``azimuth_halfwidth``
        Direction the wave arrives *from*, degrees clockwise from north, and
        half the angular width of the acceptable region (180 when it
        includes zero slowness, i.e. no direction is resolved).
    ``coherence``, ``beam``, ``single_wave``
        Mean coherency magnitude over the pairs used, the plane-wave beam
        power at the best fit, and their ratio.
    ``phase_rms_deg``
        RMS phase residual at the best fit.
    ``wavelength_over_aperture``
        Best-fit wavelength divided by the array aperture.
    ``resolved``, ``aliased``
        The bracket is finite and clear of the search edge; the acceptable
        region is split into separate pieces.
    """

    table: pd.DataFrame
    nodes: list[str]
    n_windows: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_curve(
        self, *, max_relative_width: float = 0.5, max_wavelength_ratio: float = 10.0
    ) -> DispersionCurve:
        """The frequencies whose bracket is narrow enough to call a measurement.

        Keeps rows that are resolved, not aliased, whose bracket is no wider
        than ``max_relative_width`` of the velocity, and whose wavelength is
        at most ``max_wavelength_ratio`` times the aperture. ``velocity_std``
        is half the bracket.

        The wavelength limit is not statistical. Past ten apertures the whole
        array sees a few degrees of phase, the formal bracket rests on the
        assumed ``timing_error`` and ``position_error`` being right, and a
        steeply arriving body wave is indistinguishable from a fast surface
        wave. On 2026-10-01 the compact layout returned 7-16 km/s at 3-10 Hz
        with brackets that looked respectable; none of that is a dispersion
        curve.
        """
        t = self.table
        width = (t["velocity_max"] - t["velocity_min"]) / t["velocity"]
        keep = (
            t["resolved"]
            & ~t["aliased"]
            & (width <= max_relative_width)
            & (t["wavelength_over_aperture"] <= max_wavelength_ratio)
        )
        if not keep.any():
            raise ValueError(
                "no frequency has a finite, unaliased velocity bracket narrower "
                f"than {max_relative_width:.0%} at a wavelength within "
                f"{max_wavelength_ratio:g} apertures. The wavelengths are too long "
                "for this aperture, or too short for this station spacing; the "
                "table still gives the lower bounds."
            )
        rows = t[keep]
        return DispersionCurve(
            frequency=rows["frequency"].to_numpy(float),
            velocity=rows["velocity"].to_numpy(float),
            velocity_std=((rows["velocity_max"] - rows["velocity_min"]) / 2).to_numpy(float),
            mode=0,
            wave="rayleigh",
            metadata={
                "method": "plane_wave_fit",
                "azimuth": rows["azimuth"].to_numpy(float),
                "n_windows": self.n_windows,
                **self.metadata,
            },
        )

    def __repr__(self) -> str:
        t = self.table
        return (
            f"PlaneWaveFit({len(t)} frequencies, {int(t['resolved'].sum())} resolved, "
            f"{int(t['aliased'].sum())} aliased, {self.n_windows} windows)"
        )


def _event_spectra(survey: SeismicSurvey, events: list[EventWindow]):
    """Per-node-normalised, tapered spectra of event windows."""
    fs = survey.sample_rate
    data = np.nan_to_num(survey.data)
    n = int(round(events[0].duration * fs))
    blocks = [
        data[:, e.start_sample : e.start_sample + n]
        for e in events
        if data[:, e.start_sample : e.start_sample + n].shape[1] == n
    ]
    if not blocks:
        raise ValueError("no complete event window could be transformed")
    block = np.stack(blocks, axis=1)
    block = block - block.mean(axis=2, keepdims=True)
    # Normalise each node in each event, so that neither the loudest event
    # nor the node nearest the source decides the answer on its own.
    scale = block.std(axis=2, keepdims=True)
    block = block / np.where(scale > 0, scale, 1.0) * np.hanning(n)
    return np.fft.rfft(block, axis=2), np.fft.rfftfreq(n, 1.0 / fs)


def _circular_halfwidth(degrees: np.ndarray) -> float:
    """Half the smallest arc containing every angle given."""
    values = np.unique(np.round(np.asarray(degrees) % 360.0, 6))
    if values.size <= 1:
        return 0.0
    gaps = np.diff(np.r_[values, values[0] + 360.0])
    return float((360.0 - gaps.max()) / 2.0)


def plane_wave_fit(
    survey: SeismicSurvey,
    frequencies=None,
    *,
    events: list[EventWindow] | None = None,
    window: float = 20.0,
    overlap: float = 0.5,
    anti_trigger: tuple[float, float] | None = (0.3, 3.0),
    fmin: float = 1.0,
    fmax: float = 40.0,
    n_frequencies: int = 30,
    bandwidth: float = 0.1,
    vmin: float = 50.0,
    vmax: float = 20000.0,
    position_error: float = 1.5,
    timing_error: float = 1e-4,
    min_coherence: float = 0.3,
    n_velocity: int = 600,
    azimuth_step: float = 2.0,
    layout: ArrayLayout | None = None,
) -> PlaneWaveFit:
    """Fit one plane wave per frequency to the inter-node coherency.

    Works on any array shape; a 2-D one is what makes direction and velocity
    separable. Ambient noise is windowed and gated exactly as in
    :func:`~shallowgeo.passive.spac_coherency`. Pass ``events`` (from
    :func:`~shallowgeo.passive.detect_events` or
    :func:`~shallowgeo.passive.shot_windows`) to fit transients instead --
    the 2-D counterpart of :func:`~shallowgeo.passive.event_dispersion_image`.
    A shot inside or beside the array is not a plane wave, so treat those
    velocities as apparent.

    ``frequencies`` defaults to ``n_frequencies`` log-spaced between ``fmin``
    and ``fmax``; the cross-spectra are averaged over a relative
    ``bandwidth`` around each. ``position_error`` is the one-sigma error of
    each node position in metres -- about 1.5 m for the ATOM-1C's own GPS,
    a few centimetres for a taped layout -- and ``timing_error`` the
    one-sigma clock error per node in seconds; both are carried into the
    phase uncertainty. The trial velocities run geometrically from ``vmin``
    to ``vmax`` (steps of about 1 % by default) plus infinite velocity, so
    the bracket has the same relative resolution at 100 m/s and 3 km/s; a
    ``velocity_max`` of ``inf`` means "faster than ``vmax``". Pairs
    whose coherency falls below ``min_coherence`` are left out; with fewer
    than three left, the frequency is reported as NaN.
    """
    layout = layout or array_layout(survey)
    if events is not None:
        if not events:
            raise ValueError("no event windows given; lower the detection threshold")
        spectra, freqs = _event_spectra(survey, events)
        n_windows = spectra.shape[1]
    else:
        spectra, freqs, keep = window_spectra(
            survey, window=window, overlap=overlap, anti_trigger=anti_trigger,
            fmin=0.0, fmax=np.inf,
        )
        n_windows = int(keep.sum())

    if frequencies is None:
        frequencies = np.geomspace(fmin, fmax, n_frequencies)
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))

    east, north = layout.east, layout.north
    first, second = np.triu_indices(len(east), 1)
    dx = east[first] - east[second]
    dy = north[first] - north[second]

    # Index increases with slowness, zero slowness is row 0. Geometric in
    # velocity from vmax down to vmin, so the relative step is the same
    # everywhere; below 1/vmax, a short linear run down to zero, because a
    # single jump from vmax to infinite velocity would leave the best fit
    # of a near-zero phase between two grid points and outside the bracket.
    slowness = np.r_[
        np.linspace(0.0, 1.0 / vmax, 20, endpoint=False),
        1.0 / np.geomspace(vmax, vmin, n_velocity),
    ]
    azimuth = np.arange(0.0, 360.0, azimuth_step)            # propagation direction
    sx = slowness[:, None] * np.sin(np.radians(azimuth))[None, :]
    sy = slowness[:, None] * np.cos(np.radians(azimuth))[None, :]

    rows = []
    for fc in frequencies:
        band = np.abs(freqs - fc) <= 0.5 * bandwidth * fc
        if not band.any():
            band = np.zeros_like(freqs, dtype=bool)
            band[int(np.argmin(np.abs(freqs - fc)))] = True
        x = spectra[:, :, band]
        cross = np.einsum("iwf,jwf->ij", x, x.conj())
        power = np.real(np.diag(cross))
        gamma = cross / np.sqrt(np.outer(power, power) + 1e-30)
        g = gamma[first, second]
        magnitude = np.abs(g)
        use = magnitude >= min_coherence
        n_eff = n_windows * int(band.sum())

        empty = {
            "frequency": fc, "velocity": np.nan, "velocity_min": np.nan,
            "velocity_max": np.nan, "azimuth": np.nan, "azimuth_halfwidth": np.nan,
            "coherence": float(magnitude.mean()), "beam": np.nan,
            "single_wave": np.nan, "phase_rms_deg": np.nan, "n_pairs": int(use.sum()),
            "wavelength_over_aperture": np.nan, "resolved": False, "aliased": False,
        }
        if use.sum() < 3:
            rows.append(empty)
            continue

        g, m = g[use], magnitude[use]
        steer = np.exp(
            2j * np.pi * fc * (sx[..., None] * dx[use] + sy[..., None] * dy[use])
        )
        aligned = g * steer                                    # (n_s, n_az, n_pairs)
        beam = np.real(aligned).mean(axis=-1)

        # Pass 1: the beam maximum, which needs no error model.
        i_s, i_a = np.unravel_index(int(np.argmax(beam)), beam.shape)
        # Pass 2: weighted phase misfit. Coherence sets the statistical phase
        # error (Bendat & Piersol); position error adds a term proportional
        # to the slowness, held at the pass-1 value so the weights do not
        # reward slow trial velocities for inflating their own error bars.
        sigma2 = (1.0 - m**2) / (2.0 * n_eff * m**2) + 2.0 * (
            2.0 * np.pi * fc * slowness[i_s] * position_error
        ) ** 2 + 2.0 * (2.0 * np.pi * fc * timing_error) ** 2
        sigma2 = np.maximum(sigma2, np.radians(0.5) ** 2)
        residual = np.angle(aligned)
        chi2 = np.sum(residual**2 / sigma2, axis=-1)
        i_s, i_a = np.unravel_index(int(np.argmin(chi2)), chi2.shape)
        dof = max(int(use.sum()) - 2, 1)
        scale = max(chi2[i_s, i_a] / dof, 1.0)       # inflate if the model misfits
        region = chi2 / scale <= chi2[i_s, i_a] / scale + DELTA_CHI2_68

        # Zero slowness is one point, not a row of n_az separate ones.
        labels, _ = ndimage.label(region)
        if region[0].any():
            zero = np.unique(labels[0][region[0]])
            for lab in zero[1:]:
                labels[labels == lab] = zero[0]
        # ...and azimuth wraps.
        for row in range(region.shape[0]):
            a, b = labels[row, 0], labels[row, -1]
            if a and b and a != b:
                labels[labels == b] = a
        pieces = len(np.unique(labels[labels > 0]))

        s_region = np.broadcast_to(slowness[:, None], region.shape)[region]
        az_region = np.broadcast_to(azimuth[None, :], region.shape)[region]
        best_s = slowness[i_s]
        velocity = np.inf if best_s == 0 else 1.0 / best_s
        v_min = 1.0 / s_region.max() if s_region.max() > 0 else np.inf
        # Anything faster than vmax is "unbounded" by construction.
        v_max = np.inf if s_region.min() < 1.0 / vmax else 1.0 / s_region.min()
        at_edge = bool(region[-1].any())
        coherence = float(m.mean())

        rows.append(
            {
                **empty,
                "velocity": velocity,
                "velocity_min": v_min,
                "velocity_max": v_max,
                "azimuth": float((azimuth[i_a] + 180.0) % 360.0),
                "azimuth_halfwidth": (
                    180.0 if s_region.min() == 0 else _circular_halfwidth(az_region)
                ),
                "coherence": coherence,
                "beam": float(beam[i_s, i_a]),
                "single_wave": float(beam[i_s, i_a] / coherence),
                "phase_rms_deg": float(np.degrees(np.sqrt(np.mean(residual[i_s, i_a] ** 2)))),
                "wavelength_over_aperture": (
                    velocity / (fc * layout.aperture) if layout.aperture > 0 else np.nan
                ),
                "resolved": bool(np.isfinite(v_max) and not at_edge),
                "aliased": pieces > 1,
            }
        )

    return PlaneWaveFit(
        table=pd.DataFrame(rows),
        nodes=list(layout.nodes),
        n_windows=n_windows,
        metadata={
            "source": "events" if events is not None else "ambient",
            "position_error": position_error,
            "timing_error": timing_error,
            "aperture": layout.aperture,
            "vmin": vmin,
            "vmax": vmax,
        },
    )
