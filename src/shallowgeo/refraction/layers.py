"""Horizontal-layer travel times and the intercept-time interpretation.

Forward model
-------------
For ``n`` layers with velocities ``v_0 < v_1 < ... < v_{n-1}`` (the last a
half-space) and thicknesses ``h_0 .. h_{n-2}``, the head wave refracted along
the top of layer ``k`` arrives at

    t_k(x) = x / v_k + sum_{j<k} 2 h_j sqrt(v_k^2 - v_j^2) / (v_j v_k)

and exists only beyond its critical distance. The first arrival is the
minimum over the direct wave (``k = 0``) and every existing head wave.

Inverse
-------
The slope of each travel-time branch gives a velocity, its intercept gives
one equation in the thicknesses above it. Solving those top-down is the
classical intercept-time method; :func:`fit_layers` automates the one step
that is usually done by eye -- where each branch begins and ends -- by
trying every partition of the picks and keeping the one with the smallest
misfit that also has velocities increasing with depth.

Assumptions, stated for the students: planar horizontal interfaces,
homogeneous isotropic layers, velocity increasing with depth, source and
receivers on a straight line at the surface. A dipping interface needs
forward and reverse shots; that is the next thing to add here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


def _check_layers(velocities, thicknesses) -> tuple[np.ndarray, np.ndarray]:
    v = np.atleast_1d(np.asarray(velocities, dtype=float))
    h = np.atleast_1d(np.asarray(thicknesses, dtype=float)) if thicknesses is not None else np.zeros(0)
    if h.size != v.size - 1:
        raise ValueError(
            f"{v.size} velocities need {v.size - 1} thicknesses, got {h.size}"
        )
    if np.any(v <= 0) or np.any(h <= 0):
        raise ValueError("velocities and thicknesses must be positive")
    return v, h


def intercept_times(velocities, thicknesses) -> np.ndarray:
    """Intercept time of each head-wave branch, ``k = 1 .. n-1``, seconds."""
    v, h = _check_layers(velocities, thicknesses)
    out = np.zeros(v.size - 1)
    for k in range(1, v.size):
        if v[k] <= v[:k].max():
            out[k - 1] = np.nan  # no head wave from a slower layer
            continue
        out[k - 1] = np.sum(2 * h[:k] * np.sqrt(v[k] ** 2 - v[:k] ** 2) / (v[:k] * v[k]))
    return out


def critical_distances(velocities, thicknesses) -> np.ndarray:
    """Offset beyond which each head wave exists, ``k = 1 .. n-1``, metres."""
    v, h = _check_layers(velocities, thicknesses)
    out = np.zeros(v.size - 1)
    for k in range(1, v.size):
        if v[k] <= v[:k].max():
            out[k - 1] = np.nan
            continue
        # tan of the angle each ray makes in layer j when it is critical at k
        sin_j = v[:k] / v[k]
        out[k - 1] = np.sum(2 * h[:k] * sin_j / np.sqrt(1 - sin_j ** 2))
    return out


def branch_times(offsets, velocities, thicknesses) -> np.ndarray:
    """``(n_layers, n_offsets)`` travel time of every branch, NaN where absent."""
    v, h = _check_layers(velocities, thicknesses)
    x = np.atleast_1d(np.asarray(offsets, dtype=float))
    t = np.full((v.size, x.size), np.nan)
    t[0] = x / v[0]
    if v.size > 1:
        ti = intercept_times(v, h)
        xc = critical_distances(v, h)
        for k in range(1, v.size):
            if np.isnan(ti[k - 1]):
                continue
            t[k] = np.where(x >= xc[k - 1], x / v[k] + ti[k - 1], np.nan)
    return t


def traveltimes(offsets, velocities, thicknesses=None) -> np.ndarray:
    """First-arrival time at each offset for a horizontal layered model."""
    t = branch_times(offsets, velocities, thicknesses)
    return np.nanmin(t, axis=0)


def crossover_distances(velocities, thicknesses) -> np.ndarray:
    """Offset at which branch ``k`` overtakes branch ``k-1``, ``k = 1 .. n-1``.

    Solved numerically from :func:`branch_times` because for three or more
    layers the overtaking branch is not always the one immediately above.
    """
    v, h = _check_layers(velocities, thicknesses)
    ti = intercept_times(v, h)
    out = np.full(v.size - 1, np.nan)
    for k in range(1, v.size):
        if np.isnan(ti[k - 1]):
            continue
        # crossover with the previous branch (direct or head wave)
        prev = k - 1
        ti_prev = 0.0 if prev == 0 else ti[prev - 1]
        if np.isnan(ti_prev):
            continue
        out[k - 1] = (ti[k - 1] - ti_prev) / (1 / v[prev] - 1 / v[k])
    return out


def depths_from_intercepts(velocities, intercepts) -> np.ndarray:
    """Layer thicknesses from branch velocities and intercept times.

    Solves the intercept-time equations top-down: the first intercept fixes
    ``h_0``; each later intercept, with the shallower thicknesses now known,
    fixes the next.
    """
    v = np.atleast_1d(np.asarray(velocities, dtype=float))
    ti = np.atleast_1d(np.asarray(intercepts, dtype=float))
    if ti.size != v.size - 1:
        raise ValueError("need one intercept per head-wave branch")
    if np.any(np.diff(v) <= 0):
        raise ValueError(
            f"velocities must increase with depth for the intercept-time "
            f"method; got {v.tolist()}"
        )
    h = np.zeros(v.size - 1)
    for k in range(1, v.size):
        known = np.sum(2 * h[: k - 1] * np.sqrt(v[k] ** 2 - v[: k - 1] ** 2) / (v[: k - 1] * v[k]))
        coeff = 2 * np.sqrt(v[k] ** 2 - v[k - 1] ** 2) / (v[k - 1] * v[k])
        h[k - 1] = (ti[k - 1] - known) / coeff
    return h


@dataclass
class LayeredRefractionModel:
    """Result of :func:`fit_layers`."""

    velocities: np.ndarray
    thicknesses: np.ndarray
    intercepts: np.ndarray
    breakpoints: list[int]
    offsets: np.ndarray
    times: np.ndarray
    rms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_layers(self) -> int:
        return self.velocities.size

    @property
    def depths(self) -> np.ndarray:
        """Depth to the top of each refractor."""
        return np.cumsum(self.thicknesses)

    @property
    def crossovers(self) -> np.ndarray:
        return crossover_distances(self.velocities, self.thicknesses)

    def predict(self, offsets) -> np.ndarray:
        return traveltimes(offsets, self.velocities, self.thicknesses)

    def residuals(self) -> np.ndarray:
        return self.times - self.predict(self.offsets)

    def branch_of(self) -> np.ndarray:
        """Which fitted branch each input pick was assigned to."""
        edges = [0, *self.breakpoints, self.offsets.size]
        out = np.empty(self.offsets.size, dtype=int)
        for k in range(len(edges) - 1):
            out[edges[k]:edges[k + 1]] = k
        return out

    def summary(self) -> pd.DataFrame:
        depth_top = np.concatenate([[0.0], self.depths])
        return pd.DataFrame(
            {
                "layer": np.arange(1, self.n_layers + 1),
                "velocity_m_s": self.velocities,
                "thickness_m": np.concatenate([self.thicknesses, [np.nan]]),
                "depth_to_top_m": depth_top,
                "intercept_s": np.concatenate([[0.0], self.intercepts]),
            }
        )

    def plot(self, ax=None, *, model_offsets=None):
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        x = self.offsets if model_offsets is None else np.asarray(model_offsets)
        xx = np.linspace(0, x.max() * 1.02, 400)
        branch = self.branch_of()
        for k in range(self.n_layers):
            sel = branch == k
            ax.plot(self.offsets[sel], 1e3 * self.times[sel], "o", ms=5,
                    label=f"branch {k + 1}: {self.velocities[k]:.0f} m/s")
        bt = branch_times(xx, self.velocities, self.thicknesses)
        for k in range(self.n_layers):
            ax.plot(xx, 1e3 * bt[k], "--", lw=1, color="0.5")
        ax.plot(xx, 1e3 * self.predict(xx), "k-", lw=1.8, label="first arrival")
        ax.set_xlabel("offset (m)")
        ax.set_ylabel("travel time (ms)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        return ax

    def __repr__(self) -> str:
        v = ", ".join(f"{x:.0f}" for x in self.velocities)
        h = ", ".join(f"{x:.1f}" for x in self.thicknesses)
        return (f"<LayeredRefractionModel v=[{v}] m/s h=[{h}] m "
                f"rms={1e3 * self.rms:.2f} ms>")


def _fit_branch(x, t, w, *, through_origin: bool) -> tuple[float, float, float]:
    """Weighted straight line; returns (slope, intercept, weighted SSE)."""
    if through_origin:
        slope = np.sum(w * x * t) / np.sum(w * x * x)
        intercept = 0.0
    else:
        W = np.sum(w)
        xm, tm = np.sum(w * x) / W, np.sum(w * t) / W
        sxx = np.sum(w * (x - xm) ** 2)
        if sxx == 0:
            return np.nan, np.nan, np.inf
        slope = np.sum(w * (x - xm) * (t - tm)) / sxx
        intercept = tm - slope * xm
    resid = t - (slope * x + intercept)
    return slope, intercept, float(np.sum(w * resid ** 2))


def _fit_partition(x, t, w, edges, *, direct_through_origin: bool):
    """Fit one straight line per branch; returns (slopes, intercepts, sse)."""
    slopes, intercepts, sse = [], [], 0.0
    for k in range(len(edges) - 1):
        sl = slice(edges[k], edges[k + 1])
        s, b, e = _fit_branch(
            x[sl], t[sl], w[sl], through_origin=(k == 0 and direct_through_origin)
        )
        slopes.append(s)
        intercepts.append(b)
        sse += e
    return np.asarray(slopes), np.asarray(intercepts), sse


def fit_layers(
    offsets,
    times,
    n_layers: int = 2,
    *,
    weights=None,
    min_points: int = 2,
    direct_through_origin: bool = True,
    crossovers=None,
) -> LayeredRefractionModel:
    """Intercept-time fit of first-break picks to ``n_layers`` horizontal layers.

    Parameters
    ----------
    offsets, times
        Absolute source-receiver distance (m) and first-break time (s), one
        shot or several shots merged (a single horizontal model is assumed
        for all of them).
    n_layers
        1, 2 or 3 are what the method is good for. More is allowed but the
        branch search grows combinatorially and the picks rarely support it.
    min_points
        Fewest picks a branch may have. Two defines a line; three lets the
        misfit say something.
    direct_through_origin
        Constrain the direct-wave branch to zero intercept, which it must
        have physically. Turn off to diagnose a trigger-time error: a
        non-zero direct-wave intercept is exactly what that looks like.
    crossovers
        ``n_layers - 1`` offsets at which one branch takes over from the
        next, fixed by hand instead of searched for. This is the method as
        it is done on paper -- decide by eye where the travel-time curve
        bends, then fit a line to each segment -- and it is the only way to
        overrule a search that has latched onto the wrong bend. Leave it
        ``None`` to search.

    Notes
    -----
    Picks are sorted by offset. Every partition into ``n_layers`` contiguous
    branches with at least ``min_points`` each is fitted; partitions whose
    velocities do not increase with depth are rejected; the smallest
    weighted misfit wins. Thicknesses then follow from
    :func:`depths_from_intercepts`.

    With ``crossovers`` given there is no search and no rejection: the fit
    is reported, and if it is unphysical the error says which velocities
    came out wrong, because that is the answer to "where does the bend go?"
    """
    x = np.asarray(offsets, dtype=float)
    t = np.asarray(times, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    good = np.isfinite(x) & np.isfinite(t) & (x > 0)
    x, t, w = x[good], t[good], w[good]
    order = np.argsort(x)
    x, t, w = x[order], t[order], w[order]
    n = x.size
    if n_layers < 1:
        raise ValueError("n_layers must be at least 1")
    if n < n_layers * min_points:
        raise ValueError(
            f"{n} usable picks cannot support {n_layers} branches of at least "
            f"{min_points} points"
        )

    if crossovers is not None:
        xc = np.atleast_1d(np.asarray(crossovers, dtype=float))
        if xc.size != n_layers - 1:
            raise ValueError(
                f"{n_layers} layers need {n_layers - 1} crossover offsets, "
                f"got {xc.size}"
            )
        if np.any(np.diff(xc) <= 0):
            raise ValueError(f"crossover offsets must increase, got {xc.tolist()}")
        cuts = [int(np.searchsorted(x, c)) for c in xc]
        counts = np.diff([0, *cuts, n])
        if np.any(counts < min_points):
            raise ValueError(
                f"crossovers at {np.round(xc, 1).tolist()} m split {n} picks into "
                f"branches of {counts.tolist()}; each branch needs at least "
                f"{min_points}. Move a crossover, or fit fewer layers."
            )
        slopes, b, sse = _fit_partition(
            x, t, w, [0, *cuts, n], direct_through_origin=direct_through_origin
        )
        if not np.all(np.isfinite(slopes)) or np.any(slopes <= 0):
            raise ValueError(
                "a branch came out with zero or negative slope, which is not a "
                "travel time. Move the crossovers."
            )
        v = 1.0 / slopes
        if np.any(np.diff(v) <= 0):
            raise ValueError(
                f"branch velocities {np.round(v, 0).tolist()} m/s do not increase "
                "with depth, so there is no head wave to interpret. The bend is "
                "in the wrong place, or the ground has a low-velocity layer that "
                "refraction cannot see."
            )
        if n_layers > 1 and np.any(b[1:] <= 0):
            raise ValueError(
                f"head-wave intercept times {np.round(1e3 * b[1:], 2).tolist()} ms "
                "are not all positive, so the interfaces would come out at zero or "
                "negative depth. Move the crossovers outward."
            )
    else:
        best = None
        interior = range(min_points, n - min_points + 1)
        for cuts in combinations(interior, n_layers - 1):
            edges = [0, *cuts, n]
            if any(edges[i + 1] - edges[i] < min_points for i in range(n_layers)):
                continue
            slopes, intercepts, sse = _fit_partition(
                x, t, w, edges, direct_through_origin=direct_through_origin
            )
            if not np.all(np.isfinite(slopes)) or np.any(slopes <= 0):
                continue
            v = 1.0 / slopes
            if np.any(np.diff(v) <= 0):
                continue
            if n_layers > 1 and np.any(intercepts[1:] <= 0):
                continue
            if best is None or sse < best[0]:
                best = (sse, list(cuts), v, intercepts)

        if best is None:
            raise ValueError(
                f"no partition of the picks gives {n_layers} branches with "
                "increasing velocity and positive intercepts. Check the picks, or "
                "try fewer layers."
            )
        sse, cuts, v, b = best
    thick = depths_from_intercepts(v, b[1:]) if n_layers > 1 else np.zeros(0)
    model = LayeredRefractionModel(
        velocities=v,
        thicknesses=thick,
        intercepts=b[1:],
        breakpoints=cuts,
        offsets=x,
        times=t,
        rms=float(np.sqrt(sse / np.sum(w))),
        metadata={"direct_through_origin": direct_through_origin,
                  "direct_intercept_s": float(b[0]),
                  "crossovers_fixed": crossovers is not None},
    )
    return model
