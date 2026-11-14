r"""Frequency-domain EM response of a layered ground, and what it implies.

Everything here is for a small loop-loop sensor -- a transmitter and a
receiver coil a distance :math:`s` apart, both at height :math:`h` above a
flat, horizontally layered ground -- in the quasi-static limit (no
displacement currents, fine below 100 kHz in soil). Readings are the
secondary field as a fraction of the free-space primary field at the
receiver, in parts per million, with time dependence
:math:`e^{i\omega t}`: in-phase is the real part, quadrature the imaginary.

Two coil orientations:

HCP, horizontal coplanar (vertical dipoles; the GEM-2 ski lying flat)
    .. math:: \frac{H_s}{H_p} = -s^3 \int_0^\infty r_{TE}(\lambda)\,
              \lambda^2 e^{-2\lambda h} J_0(\lambda s)\, d\lambda

VCP, vertical coplanar (horizontal dipoles; the ski on its edge)
    .. math:: \frac{H_s}{H_p} = -s^2 \int_0^\infty r_{TE}(\lambda)\,
              \lambda\, e^{-2\lambda h} J_1(\lambda s)\, d\lambda

with :math:`r_{TE}` the plane-wave reflection coefficient of the layered
ground at horizontal wavenumber :math:`\lambda` (Ward & Hohmann 1988).
The integrals are evaluated by removing the large-:math:`\lambda`
asymptote of :math:`r_{TE}`, which integrates in closed form, and
integrating the rapidly decaying remainder numerically; the result is
checked against Wait's closed-form half-space solution in the tests.

**Low induction number (LIN).** When the coil separation is small compared
with the skin depth, :math:`\delta = \sqrt{2/(\omega\mu_0\sigma)}`, the
quadrature reading is proportional to conductivity (McNeill 1980):

.. math:: Q \approx \frac{\omega\mu_0\sigma s^2}{4}\, R(h/s),
   \qquad R_{HCP}(z) = \frac{1}{\sqrt{4z^2+1}},
   \qquad R_{VCP}(z) = \sqrt{4z^2+1} - 2z

and :math:`R` is also McNeill's cumulative response: the fraction of the
reading that comes from ground deeper than :math:`z\,s` below the coils.
Two consequences matter for a small multi-frequency sensor like the GEM-2:
in LIN the depth the sensor sees is set by the coil separation and the
height, **not by the frequency**, and the reading scales directly with
:math:`R(h/s)`, so the carrying height is part of every number.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j0, j1

MU0 = 4e-7 * np.pi

# Wavenumber grid for the numerical part of the Hankel integrals: log-spaced
# near zero (where r_TE varies on the scale of 1/skin depth, which can be
# tiny) and linear above, finely enough to resolve J0(lambda*s) for s ~ 1-4 m.
_LAM = np.unique(np.concatenate([
    np.logspace(-7, 0, 400),
    np.arange(1.0, 120.0, 0.01),
]))


def _check_orientation(orientation: str) -> str:
    o = orientation.upper()
    if o not in ("HCP", "VCP"):
        raise ValueError(f"orientation must be 'HCP' or 'VCP', got {orientation!r}")
    return o


def skin_depth(conductivity, frequency):
    """Plane-wave skin depth, metres: :math:`\\sqrt{2/(\\omega\\mu_0\\sigma)}`."""
    sigma = np.asarray(conductivity, float)
    f = np.asarray(frequency, float)
    with np.errstate(divide="ignore"):
        return np.sqrt(2.0 / (2 * np.pi * f * MU0 * sigma))


def induction_number(conductivity, frequency, coil_separation):
    """:math:`B = s/\\delta`. LIN formulas hold to a few percent for B < ~0.3."""
    return coil_separation / skin_depth(conductivity, frequency)


def lin_height_factor(height, coil_separation, orientation="HCP"):
    """McNeill's :math:`R(h/s)`: the LIN reading at height *h* over that at h = 0."""
    o = _check_orientation(orientation)
    z = np.asarray(height, float) / coil_separation
    if o == "HCP":
        return 1.0 / np.sqrt(4 * z**2 + 1)
    return np.sqrt(4 * z**2 + 1) - 2 * z


def cumulative_response(depth, coil_separation, orientation="HCP", height=0.0):
    """Fraction of the LIN reading that comes from ground deeper than *depth*.

    *depth* is measured down from the ground surface, the sensor being
    *height* above it. 1 at the surface, falling towards 0.
    """
    h = np.asarray(height, float)
    d = np.asarray(depth, float)
    return (lin_height_factor(h + d, coil_separation, orientation)
            / lin_height_factor(h, coil_separation, orientation))


def depth_of_investigation(coil_separation, orientation="HCP", height=0.0,
                           fraction=0.7):
    """Depth below the ground above which *fraction* of the LIN reading arises.

    With the default 0.7 this is the usual "effective depth": 1.6 s for HCP
    and 0.76 s for VCP with the coils on the ground (McNeill rounds them to
    1.5 s and 0.75 s). Lifting the sensor
    moves it *deeper* below ground (3.3 m for a GEM-2 carried HCP at 1 m,
    7.3 m at 3 m), not because the sensor sees farther but because the
    topsoil no longer dominates a reading that has itself become much
    weaker: at 3 m the whole ground signal is 27 % of what it is at the
    surface.
    """
    o = _check_orientation(orientation)
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    c = (1 - fraction) * lin_height_factor(height, coil_separation, o)
    if o == "HCP":
        z = np.sqrt((1 / c**2 - 1) / 4)
    else:
        z = (1 - c**2) / (4 * c)
    return z * coil_separation - np.asarray(height, float)


def lin_quadrature(conductivity, frequency, coil_separation, height=0.0,
                   orientation="HCP"):
    """LIN quadrature reading, ppm, over uniform ground."""
    w = 2 * np.pi * np.asarray(frequency, float)
    q = (w * MU0 * np.asarray(conductivity, float) * coil_separation**2 / 4
         * lin_height_factor(height, coil_separation, orientation))
    return q * 1e6


def lin_conductivity(quadrature, frequency, coil_separation, height=0.0,
                     orientation="HCP"):
    """Apparent conductivity, S/m, from the quadrature (ppm) by the LIN formula.

    Includes the height factor. With ``height=0`` this is McNeill's
    familiar :math:`\\sigma_a = 4Q/(\\omega\\mu_0 s^2)`.
    """
    w = 2 * np.pi * np.asarray(frequency, float)
    return (4 * np.asarray(quadrature, float) * 1e-6
            / (w * MU0 * coil_separation**2
               * lin_height_factor(height, coil_separation, orientation)))


def lin_susceptibility_factor(height, coil_separation, orientation="HCP"):
    """In-phase ppm per unit (SI) susceptibility of a uniform half-space.

    Static limit, small susceptibility. For HCP it is
    :math:`-\\tfrac12 (8z^2-1)/(4z^2+1)^{5/2}`, which **changes sign** at
    :math:`z = h/s = 0.354`: below that height magnetic soil reads positive,
    above it negative. For VCP it is :math:`-\\tfrac12 (4z^2+1)^{-3/2}`.
    """
    o = _check_orientation(orientation)
    z = np.asarray(height, float) / coil_separation
    if o == "HCP":
        f = -0.5 * (8 * z**2 - 1) / (4 * z**2 + 1) ** 2.5
    else:
        f = -0.5 / (4 * z**2 + 1) ** 1.5
    return f * 1e6


def lin_susceptibility(inphase, coil_separation, height=0.0, orientation="HCP"):
    """Apparent susceptibility (SI) from the in-phase reading (ppm).

    Only as good as the in-phase zero level, which on a GEM-2 is set by
    calibration and drifts with temperature; see the notebook. Undefined
    for HCP near the sign-change height ``0.354 s``.
    """
    return np.asarray(inphase, float) / lin_susceptibility_factor(
        height, coil_separation, orientation)


# -- full solution --------------------------------------------------------------

def _rte(lam, omega, conductivities, thicknesses, mur):
    """Reflection coefficient of a layered ground, shape ``(..., n_lambda)``.

    ``conductivities`` and ``mur`` have shape ``(..., n_layers)``;
    ``thicknesses`` ``(..., n_layers - 1)``.
    """
    sig = np.asarray(conductivities, complex)
    mur = np.asarray(mur, float)
    lam = lam[..., :]
    k2 = 1j * omega * MU0 * sig[..., None, :] * mur[..., None, :]  # (..., 1, nl)
    lam_ = lam[:, None]                                           # (L, 1)
    u = np.sqrt(lam_**2 + k2)                                     # (..., L, nl)
    y = u / mur[..., None, :]
    Y = y[..., -1]
    n = sig.shape[-1]
    t = np.asarray(thicknesses, float)
    for j in range(n - 2, -1, -1):
        th = np.tanh(u[..., j] * t[..., j][..., None])
        Y = y[..., j] * (Y + y[..., j] * th) / (y[..., j] + Y * th)
    return (lam - Y) / (lam + Y)


def _hankel(r, r_inf, c, coil_separation, height, orientation):
    """Normalised secondary field for reflection coefficient *r* on ``_LAM``."""
    s = float(coil_separation)
    p = 2.0 * float(height)
    lam = _LAM
    ex = np.exp(-p * lam)
    r_inf = np.asarray(r_inf)[..., None]
    c = np.asarray(c)[..., None]
    if orientation == "HCP":
        rem = (r * lam**2 - r_inf * lam**2 - c) * ex * j0(lam * s)
        num = np.trapezoid(rem, lam, axis=-1)
        a0 = (2 * p**2 - s**2) / (p**2 + s**2) ** 2.5
        a2 = 1.0 / np.sqrt(p**2 + s**2)
        total = num + r_inf[..., 0] * a0 + c[..., 0] * a2
        return -s**3 * total
    rem = (r * lam - r_inf * lam - c / lam) * ex * j1(lam * s)
    num = np.trapezoid(rem, lam, axis=-1)
    b0 = s / (p**2 + s**2) ** 1.5
    b2 = (np.sqrt(p**2 + s**2) - p) / s
    total = num + r_inf[..., 0] * b0 + c[..., 0] * b2
    return -s**2 * total


def layered_response(conductivities, thicknesses, frequency, coil_separation,
                     height=0.0, orientation="HCP", susceptibilities=None):
    """Complex response (in-phase + i quadrature), ppm, of a layered ground.

    Parameters
    ----------
    conductivities
        S/m, top layer first; the last is the half-space. A leading
        dimension (several models at once) is allowed: ``(n_models, n_layers)``.
    thicknesses
        Metres, one fewer than the layers.
    frequency
        Hz, scalar or 1-D; the result gains a trailing frequency axis when
        it is 1-D.
    susceptibilities
        SI, per layer. Default zero.
    """
    o = _check_orientation(orientation)
    if height < 0:
        raise ValueError("height must be >= 0")
    sig = np.atleast_1d(np.asarray(conductivities, float))
    t = np.asarray(thicknesses, float)
    if t.shape[-1:] != (sig.shape[-1] - 1,) and not (sig.shape[-1] == 1 and t.size == 0):
        raise ValueError("need one fewer thickness than conductivities")
    t = np.broadcast_to(t, sig.shape[:-1] + (sig.shape[-1] - 1,))
    kappa = (np.zeros_like(sig) if susceptibilities is None
             else np.broadcast_to(np.asarray(susceptibilities, float), sig.shape))
    mur = 1.0 + kappa
    freqs = np.atleast_1d(np.asarray(frequency, float))
    out = []
    for f in freqs:
        w = 2 * np.pi * f
        r = _rte(_LAM, w, sig, t, mur)
        m1 = mur[..., 0]
        k2 = 1j * w * MU0 * sig[..., 0] * m1
        r_inf = (m1 - 1) / (m1 + 1)
        c = -(m1**2) * k2 / (m1 + 1) ** 2
        out.append(_hankel(r, r_inf, c, coil_separation, height, o) * 1e6)
    res = np.stack(out, axis=-1)
    return res if np.ndim(frequency) else res[..., 0]


def halfspace_response(conductivity, frequency, coil_separation, height=0.0,
                       orientation="HCP", susceptibility=0.0):
    """Complex response, ppm, of uniform ground. *conductivity* may be 1-D."""
    sig = np.asarray(conductivity, float)[..., None]
    kap = np.asarray(susceptibility, float)[..., None] * np.ones_like(sig)
    return layered_response(sig, np.zeros(sig.shape[:-1] + (0,)), frequency,
                            coil_separation, height, orientation, kap)


def wait_halfspace(conductivity, frequency, coil_separation, orientation="HCP"):
    """Closed-form response, ppm, with both coils *on* uniform ground (h = 0).

    Wait (1982); also McNeill (1980), TN-6, eqs. 4-5. Used to check
    :func:`halfspace_response`.
    """
    o = _check_orientation(orientation)
    g = np.sqrt(1j * 2 * np.pi * np.asarray(frequency, float) * MU0
                * np.asarray(conductivity, float)) * coil_separation
    if o == "HCP":
        hs = 2 / g**2 * (9 - (9 + 9 * g + 4 * g**2 + g**3) * np.exp(-g)) - 1
    else:
        hs = 2 * (1 - 3 / g**2 + (3 + 3 * g + g**2) * np.exp(-g) / g**2) - 1
    return hs * 1e6


# -- apparent conductivity --------------------------------------------------------

#: Conductivity grid for the half-space lookup, S/m (0.01 mS/m to 30 S/m).
SIGMA_GRID = np.logspace(-5, 1.5, 261)


def _branch(q):
    """Index of the quadrature peak: Q rises with sigma only up to there."""
    return int(np.argmax(q)) + 1


def apparent_conductivity(quadrature, frequency, coil_separation, height=0.0,
                          orientation="HCP", *, method="halfspace"):
    """Conductivity of the uniform ground that would give each quadrature reading.

    Parameters
    ----------
    quadrature
        ppm, any shape.
    height
        Sensor height above ground, metres: one number, or one per reading
        (a drone survey).
    method
        ``"halfspace"`` (default) inverts the full solution, so it stays
        right outside the low-induction-number range; ``"lin"`` is McNeill's
        linear formula with the height factor.

    Returns
    -------
    S/m, NaN where no uniform ground gives the reading: zero or negative
    quadrature (noise, or a zero-level error), or a reading above the
    largest quadrature a half-space can produce at that height.
    """
    q = np.asarray(quadrature, float)
    h = np.broadcast_to(np.asarray(height, float), q.shape)
    if method == "lin":
        sig = lin_conductivity(q, frequency, coil_separation, h, orientation)
        return np.where(q > 0, sig, np.nan)
    if method != "halfspace":
        raise ValueError("method must be 'halfspace' or 'lin'")

    flat_q, flat_h = q.ravel(), h.ravel()
    out = np.full(flat_q.shape, np.nan)
    finite = np.isfinite(flat_h)
    hr = np.round(flat_h[finite], 3)
    uniq = np.unique(hr)
    if len(uniq) > 60:
        grid = np.linspace(uniq.min(), uniq.max(), 41)
    else:
        grid = uniq
    table = np.stack([halfspace_response(SIGMA_GRID, frequency, coil_separation,
                                         hh, orientation).imag for hh in grid])
    log_sig = np.log(SIGMA_GRID)
    idx_finite = np.flatnonzero(finite)
    for i, hh in zip(idx_finite, hr):
        qi = flat_q[i]
        if not np.isfinite(qi) or qi <= 0:
            continue
        if len(grid) == 1:
            curve = table[0]
        else:
            k = int(np.clip(np.searchsorted(grid, hh) - 1, 0, len(grid) - 2))
            wt = (hh - grid[k]) / (grid[k + 1] - grid[k])
            curve = (1 - wt) * table[k] + wt * table[k + 1]
        b = _branch(curve)
        if qi < curve[0] or qi > curve[b - 1]:
            continue
        out[i] = np.exp(np.interp(qi, curve[:b], log_sig[:b]))
    return out.reshape(q.shape)
