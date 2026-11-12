"""ERT data: electrodes, four-electrode measurements, and the RES2DINV format.

A resistivity measurement drives current ``I`` between two electrodes
(``a``, ``b``) and reads the voltage ``V`` between two others (``m``,
``n``). The ratio ``R = V / I`` depends on the ground and on where the four
electrodes are; the **geometric factor** ``k`` takes the second part out,

    rho_a = k R,      k = 2 pi / (1/AM - 1/BM - 1/AN + 1/BN),

so that over uniform ground ``rho_a`` is the ground's resistivity whatever
the layout. Over real ground it is a weighted average of everything the
current passes through: an *apparent* resistivity, not a resistivity at any
one place. Turning the apparent resistivities into a picture of the ground
is the job of :mod:`~shallowgeo.ert.inversion`.

The RES2DINV ``.dat`` format
----------------------------
What an AGI SuperSting (and most other instruments) exports for RES2DINV.
Six header lines, then one line per measurement::

    Sting/Swift prg: 2610234      title
      0.91440                     smallest electrode spacing
    3                             array type (see ARRAYS)
    323                           number of measurements
    1                             x-location: 0 first electrode, 1 mid-point
    0                             0 = no IP data
      1.37160  0.91440  1.00000  88.39630     x, a, n, rho_a
    ...
    0                             topography flag, then trailing zeros

``a`` is the dipole length and ``n`` the dipole separation in multiples of
``a``. Array type 11 (*general array*) lists the four electrode positions
explicitly instead. :func:`read_res2dinv` reads both and returns an
:class:`ERTData` with the electrodes numbered; :meth:`ERTData.to_res2dinv`
writes one back.

Two problems with SuperSting exports
------------------------------------
**Non-integer n.** The SuperSting computes ``x``, ``a`` and ``n`` from
electrode positions held in single precision, so ``n`` comes out as
``8.00039`` or ``2.99999`` and ``a`` as ``6.40079``. RES2DINV rejects a
non-integer ``n``. Reading snaps every electrode to the nearest multiple of
the unit spacing, which fixes ``a``, ``n`` and ``x`` together and writes
them back clean (the ``convert.py`` fix, done on the electrodes rather than
on the columns).

**Measurements past the end of the cable.** When the command file was made
for more electrodes than were laid out, the SuperSting keeps measuring with
electrode addresses that have nothing on them, and writes those readings to
the file like any other. They look plausible -- they continue the trend of
the real ones -- and they are not data. :meth:`ERTData.trim` drops every
measurement that uses an electrode beyond the number actually in the ground.

And the opposite problem: **readings missing on the cable.** A file can lack
readings that its own measurement scheme makes on the electrodes that were
there (skipped injections, a filtered export, a run stopped early).
:meth:`ERTData.coverage` counts, for each array configuration, the positions
measured out of those possible, and :meth:`ERTData.missing_readings` lists
the gaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.provenance import Provenance

__all__ = [
    "ARRAYS",
    "ERTData",
    "read_res2dinv",
    "read_stg",
    "geometric_factor",
    "median_depth",
    "dipole_dipole",
    "wenner",
    "schlumberger",
]

#: RES2DINV array-type codes.
ARRAYS = {
    1: "wenner", 2: "pole-pole", 3: "dipole-dipole", 4: "wenner-beta",
    5: "wenner-gamma", 6: "pole-dipole", 7: "wenner-schlumberger", 11: "general",
}
_CODES = {v: k for k, v in ARRAYS.items()}

# For each standard array: electrode offsets (A, B, M, N) from the first
# electrode, in units of a, as a function of n. None marks a remote electrode.
# The order follows RES2DINV so that k comes out positive (dipole-dipole is
# laid out C2 C1 P1 P2: B, A, M, N from left to right).
_LAYOUT = {
    1: lambda n: (0.0, 3.0, 1.0, 2.0),
    2: lambda n: (0.0, None, 1.0, None),
    3: lambda n: (1.0, 0.0, n + 1.0, n + 2.0),
    4: lambda n: (1.0, 0.0, 2.0, 3.0),
    5: lambda n: (0.0, 2.0, 1.0, 3.0),
    6: lambda n: (0.0, None, n, n + 1.0),
    7: lambda n: (0.0, 2 * n + 1.0, n, n + 1.0),
}

_SNAP_TOL = 0.02  # an electrode may sit this fraction of a spacing off the grid


def geometric_factor(a, b, m, n) -> np.ndarray:
    """``k`` (m) for electrode positions ``(x, z)`` on a flat half-space.

    Each argument is an ``(N, 2)`` array of positions, or ``(N,)`` along-line
    distances; NaN marks a remote ("infinite") electrode, whose terms drop
    out. ``rho_a = k V / I``.
    """
    pts = [np.atleast_1d(np.asarray(p, dtype=float)) for p in (a, b, m, n)]
    pts = [p[:, None] if p.ndim == 1 else p for p in pts]

    def inv(p, q):
        r = np.sqrt(np.sum((p - q) ** 2, axis=1))
        return np.where(np.isfinite(r), 1.0 / np.where(r > 0, r, np.nan), 0.0)

    A, B, M, N = pts
    g = inv(A, M) - inv(B, M) - inv(A, N) + inv(B, N)
    return 2 * np.pi / g


def _pp_cumulative(z, r):
    """Fraction of a pole-pole pair's signal from above depth ``z`` (Roy & Apparao)."""
    return 1.0 - 1.0 / np.sqrt(1.0 + 4.0 * z ** 2 / r ** 2)


def median_depth(xa, xb, xm, xn) -> np.ndarray:
    """Edwards (1977) median depth of investigation for surface arrays.

    The depth above which half of the measurement's sensitivity lies, over a
    uniform half-space. It is the usual pseudo-depth for plotting a
    pseudosection: about 0.52 a for Wenner and 0.42 a for dipole-dipole with
    n = 1. NaN positions are remote electrodes.
    """
    xs = [np.atleast_1d(np.asarray(v, dtype=float)) for v in (xa, xb, xm, xn)]
    terms = []
    for (p, q, sign) in ((0, 2, 1), (1, 2, -1), (0, 3, -1), (1, 3, 1)):
        r = np.abs(xs[p] - xs[q])
        terms.append((sign, r))

    def cum(z):
        tot = np.zeros_like(z)
        for sign, r in terms:
            ok = np.isfinite(r) & (r > 0)
            rr = np.where(ok, r, 1.0)
            tot += np.where(ok, sign / rr * _pp_cumulative(z, rr), 0.0)
        return tot

    total = np.zeros(xs[0].shape)
    for sign, r in terms:
        ok = np.isfinite(r) & (r > 0)
        total += np.where(ok, sign / np.where(ok, r, 1.0), 0.0)
    span = np.nanmax(np.abs(np.column_stack(xs) - np.nanmean(np.column_stack(xs), 1)[:, None]),
                     axis=1)
    lo, hi = np.zeros_like(total), 20 * np.maximum(span, 1e-6) + 1.0
    for _ in range(60):  # bisection, vectorised
        mid = 0.5 * (lo + hi)
        below = cum(mid) < 0.5 * total
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# The data container
# ---------------------------------------------------------------------------


@dataclass
class ERTData:
    """Electrodes and the measurements made with them.

    ``electrodes`` is ``(n_electrodes, 2)``: along-line position ``x`` and
    depth ``z`` (zero on a flat surface), metres, sorted by ``x``.

    ``data`` has one row per measurement with integer electrode indices
    ``a``, ``b``, ``m``, ``n`` (``-1`` for a remote electrode), ``rhoa``
    (ohm-m), ``k`` (m) and ``r`` (``rhoa / k``, ohm). ``x`` and ``depth``
    place each measurement on a pseudosection (mid-point and Edwards median
    depth). For dipole-dipole data from a standard-array file, ``dipole``
    and ``sep`` are the RES2DINV ``a`` (in metres) and ``n``.
    """

    electrodes: np.ndarray
    data: pd.DataFrame
    spacing: float
    array: str = "general"
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)

    def __post_init__(self) -> None:
        self.electrodes = np.asarray(self.electrodes, dtype=float).reshape(-1, 2)
        need = {"a", "b", "m", "n", "rhoa"}
        missing = need - set(self.data.columns)
        if missing:
            raise ValueError(f"data is missing column(s) {sorted(missing)}")
        d = self.data.reset_index(drop=True).copy()
        for c in "abmn":
            d[c] = d[c].astype(int)
        idx = d[list("abmn")].to_numpy()
        if idx.max() >= len(self.electrodes) or idx.min() < -1:
            raise ValueError("electrode index out of range")
        pos = self._positions(idx)
        d["k"] = geometric_factor(*pos)
        d["r"] = d["rhoa"] / d["k"]
        xs = np.column_stack([p[:, 0] for p in pos])
        d["x"] = np.nanmean(xs, axis=1)
        d["depth"] = median_depth(*xs.T)
        self.data = d

    def _positions(self, idx: np.ndarray) -> list[np.ndarray]:
        e = np.vstack([self.electrodes, [np.nan, np.nan]])  # -1 -> remote
        return [e[np.where(idx[:, j] < 0, -1, idx[:, j])] for j in range(4)]

    # -- basic properties ----------------------------------------------------

    @property
    def n_electrodes(self) -> int:
        return len(self.electrodes)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return (f"<ERTData: {len(self)} {self.array} measurements on "
                f"{self.n_electrodes} electrodes, spacing {self.spacing:g} m>")

    def copy(self) -> ERTData:
        return ERTData(self.electrodes.copy(), self.data.copy(), self.spacing, self.array,
                       self.title, dict(self.metadata), self.provenance.copy())

    def used_electrodes(self) -> np.ndarray:
        """Indices of the electrodes that at least one measurement uses."""
        idx = self.data[list("abmn")].to_numpy().ravel()
        return np.unique(idx[idx >= 0])

    # -- cleaning ------------------------------------------------------------

    def select(self, keep, *, reason: str = "select") -> ERTData:
        """A copy with only the measurements where ``keep`` is True."""
        keep = np.asarray(keep, dtype=bool)
        out = self.copy()
        out.data = self.data[keep].reset_index(drop=True)
        out.provenance.record(reason, kept=int(keep.sum()), dropped=int((~keep).sum()))
        return out

    def trim(self, n_electrodes: int, *, first: int = 0) -> ERTData:
        """Drop every measurement that uses an electrode outside the cable.

        Parameters
        ----------
        n_electrodes
            Electrodes actually in the ground (42 for a 42-takeout cable).
        first
            Index of the first electrode in the ground, in this object's
            numbering (0 when the file starts at the start of the cable).

        The electrodes are numbered by position along the line from the
        unit spacing in the file, so electrode ``first + n_electrodes - 1``
        is the last real one and anything beyond it -- however reasonable
        its readings look -- was measured with an address that had no
        electrode on it. Electrodes outside the range are removed from
        ``electrodes`` as well, and the rest renumbered from zero.
        """
        if n_electrodes < 2:
            raise ValueError("n_electrodes must be at least 2")
        lo, hi = first, first + n_electrodes - 1
        idx = self.data[list("abmn")].to_numpy()
        inside = (idx < 0) | ((idx >= lo) & (idx <= hi))
        keep = inside.all(axis=1)
        out = self.copy()
        d = self.data[keep].reset_index(drop=True)
        for c in "abmn":
            d[c] = np.where(d[c] >= 0, d[c] - lo, -1)
        out.electrodes = self.electrodes[lo:hi + 1]
        out.data = d
        out.__post_init__()
        out.provenance.record("trim_electrodes", n_electrodes=n_electrodes, first=first,
                              kept=int(keep.sum()), dropped=int((~keep).sum()))
        return out

    def drop_nonpositive(self) -> ERTData:
        """Drop measurements with zero or negative apparent resistivity."""
        return self.select(self.data["rhoa"] > 0, reason="drop_nonpositive")

    # -- coverage ------------------------------------------------------------

    def _configurations(self) -> tuple[np.ndarray, np.ndarray]:
        """Each measurement as (electrode offsets from its left-most electrode, that electrode).

        Offsets of remote electrodes stay ``-1``. Two measurements with the
        same offsets are the same array shape at different places on the line.
        """
        idx = self.data[list("abmn")].to_numpy()
        real = idx >= 0
        start = np.where(real, idx, np.iinfo(idx.dtype).max).min(axis=1)
        return np.where(real, idx - start[:, None], -1), start

    def _configuration_labels(self, offsets: np.ndarray) -> dict[str, np.ndarray]:
        """Dipole length (in spacings) and ``n`` for dipole-dipole offsets, else nothing."""
        if self.array != "dipole-dipole":
            return {}
        dip = np.abs(offsets[:, 0] - offsets[:, 1])
        return {"dipole": dip, "sep": (offsets[:, 2] - offsets[:, 0]) // np.maximum(dip, 1)}

    def coverage(self) -> pd.DataFrame:
        """For each array shape in the data, how many of its possible positions were measured.

        An array shape (a *configuration*) is a measurement's four electrodes
        relative to the left-most of them: for dipole-dipole, a dipole length
        and an ``n``. It fits at every position along the line where all of
        its electrodes land on ``electrodes``; this counts those positions and
        how many the data hold. Run it after :meth:`trim`, so that the
        electrodes are the ones in the ground.

        A survey whose command file measures every shape at every position
        has nothing missing. A gap means a reading the survey's own scheme
        allows was not made or not exported -- a skipped injection, a filtered
        export, an aborted run -- and leaves a hole in the pseudosection that
        :meth:`trim` did not cause. :meth:`missing_readings` lists them.

        Returns one row per configuration: ``offsets`` (``a b m n`` relative
        to the left-most electrode), ``dipole`` and ``sep`` for dipole-dipole
        (in spacings), ``measured``, ``possible`` and ``missing`` (counts of
        positions; repeat readings count once).
        """
        offsets, start = self._configurations()
        shapes, which = np.unique(offsets, axis=0, return_inverse=True)
        which = which.ravel()
        rows = []
        for i, shape in enumerate(shapes):
            possible = max(self.n_electrodes - int(shape.max()), 0)
            measured = np.unique(start[which == i]).size
            rows.append({"offsets": tuple(int(o) for o in shape), "measured": measured,
                         "possible": possible, "missing": possible - measured})
        out = pd.DataFrame(rows)
        for name, values in self._configuration_labels(shapes).items():
            out.insert(1, name, values.astype(int))
        if "dipole" in out:
            out = out[["offsets", "dipole", "sep", "measured", "possible", "missing"]]
            out = out.sort_values(["dipole", "sep"], ignore_index=True)
        return out

    def missing_readings(self) -> pd.DataFrame:
        """Readings the survey's own configurations allow on these electrodes but the data lack.

        One row per missing reading, with its electrodes ``a b m n`` and where
        it would sit on a pseudosection (``x``, ``depth``), plus ``dipole``
        and ``sep`` for dipole-dipole. See :meth:`coverage`.
        """
        offsets, start = self._configurations()
        shapes, which = np.unique(offsets, axis=0, return_inverse=True)
        which = which.ravel()
        rows, row_shapes = [], []
        for i, shape in enumerate(shapes):
            have = set(start[which == i].tolist())
            for s in range(max(self.n_electrodes - int(shape.max()), 0)):
                if s not in have:
                    rows.append(np.where(shape >= 0, shape + s, -1))
                    row_shapes.append(shape)
        cols = ["a", "b", "m", "n", "x", "depth"]
        if not rows:
            return pd.DataFrame(columns=cols)
        ghost = ERTData(self.electrodes,
                        pd.DataFrame(np.array(rows), columns=list("abmn")).assign(rhoa=np.nan),
                        self.spacing, self.array)
        out = ghost.data[cols].copy()
        for name, values in self._configuration_labels(np.array(row_shapes)).items():
            out[name] = values.astype(int)
        return out

    # -- writing -------------------------------------------------------------

    def to_res2dinv(self, path: str | Path, *, general: bool | None = None) -> Path:
        """Write a RES2DINV ``.dat`` file that RES2DINV will load.

        Standard arrays are written in the compact ``x a n rho_a`` form with
        integer ``n`` and the mid-point convention; anything else, or
        ``general=True``, as array type 11 with explicit electrode positions.
        The count of measurements in the header always matches the file.
        For ResIPy pass ``general=True``: its reader rebuilds electrode
        positions from the mid-points of the standard form and ends up with
        duplicate electrodes (106 for the 42 of the example line).
        """
        path = Path(path)
        code = _CODES.get(self.array, 11)
        if general or code == 11 or not self._is_standard(code):
            text = self._general_text()
        else:
            text = self._standard_text(code)
        path.write_text(text)
        return path

    def _standard_params(self, code: int):
        """``(x_mid, a, n)`` arrays for a standard array, or None if any row is not one."""
        idx = self.data[list("abmn")].to_numpy()
        x = np.append(self.electrodes[:, 0], np.nan)
        A, B, M, N = (x[np.where(idx[:, j] < 0, -1, idx[:, j])] for j in range(4))
        # Dipole length and separation from the electrode roles of each layout.
        a = {1: M - A, 2: M - A, 3: A - B, 4: A - B, 5: M - A, 6: N - M, 7: N - M}[code]
        n = (M - A) / a if code in (3, 6, 7) else np.ones_like(a)
        if not (np.all(np.isfinite(a)) and np.all(a > 0)):
            return None
        n_int = np.round(n)
        if np.any(np.abs(n - n_int) > 1e-6) or np.any(n_int < 1):
            return None
        x0 = np.nanmin(np.column_stack([A, B, M, N]), axis=1)
        tol = 1e-6 * max(self.spacing, 1.0)
        mids = np.empty_like(a)
        for i in range(a.size):
            lay = _LAYOUT[code](n_int[i])
            exp = np.array([np.nan if o is None else x0[i] + o * a[i] for o in lay])
            got = np.array([A[i], B[i], M[i], N[i]])
            if not np.array_equal(np.isnan(exp), np.isnan(got)) or \
                    np.nanmax(np.abs(exp - got)) > tol:
                return None
            mids[i] = 0.5 * (np.nanmin(exp) + np.nanmax(exp))
        return list(zip(mids, a, n_int.astype(int)))

    def _is_standard(self, code: int) -> bool:
        return self._standard_params(code) is not None

    def _standard_text(self, code: int) -> str:
        params = self._standard_params(code)
        lines = [self.title or "shallowgeo", f"{self.spacing:10.4f}", str(code),
                 str(len(self)), "1", "0"]
        for (x, a, n), rho in zip(params, self.data["rhoa"]):
            if code in (3, 6, 7):
                lines.append(f"{x:10.4f} {a:10.4f} {n:4d} {rho:14.5f}")
            else:
                lines.append(f"{x:10.4f} {a:10.4f} {rho:14.5f}")
        lines += ["0", "0", "0", "0", ""]
        return "\n".join(lines)

    def _general_text(self) -> str:
        lines = [self.title or "shallowgeo", f"{self.spacing:10.4f}", "11", "0",
                 "Type of measurement (0=app. resistivity,1=resistance)", "0",
                 str(len(self)), "1", "0"]
        e = self.electrodes
        for row in self.data.itertuples():
            ids = [row.a, row.b, row.m, row.n]
            pts = [(e[i, 0], -e[i, 1]) for i in ids if i >= 0]
            coords = " ".join(f"{px:.4f} {pz:.4f}" for px, pz in pts)
            lines.append(f"{len(pts)} {coords} {row.rhoa:.5f}")
        lines += ["0", "0", "0", "0", ""]
        return "\n".join(lines)

    # -- plotting ------------------------------------------------------------

    def plot_pseudosection(self, ax=None, *, values=None, cmap: str = "turbo",
                           vmin=None, vmax=None, log: bool = True, size: float | None = None,
                           colorbar: bool = True, label: str | None = None):
        """Apparent resistivity at each measurement's mid-point and median depth.

        A picture of the *data*, not of the ground: the depth axis is a
        pseudo-depth, and a single buried body shows up as an inverted V
        whose shape depends on the array. Each measurement is one marker,
        uncontoured, so gaps in the measurement scheme stay visible.
        ``values`` replaces ``rhoa`` (for plotting misfits, for example).
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm, Normalize

        ax = ax or plt.gca()
        d = self.data
        v = d["rhoa"].to_numpy(float) if values is None else np.asarray(values, dtype=float)
        x, z = d["x"].to_numpy(float), d["depth"].to_numpy(float)
        if log:
            pos = v[v > 0]
            norm = LogNorm(vmin=vmin or pos.min(), vmax=vmax or pos.max())
        else:
            lim = np.nanmax(np.abs(v))
            norm = Normalize(vmin=-lim if vmin is None else vmin,
                             vmax=lim if vmax is None else vmax)
        order = np.argsort(-z)  # deep first, so shallow markers sit on top
        sc = ax.scatter(x[order], z[order], c=v[order], cmap=cmap, norm=norm,
                        s=size or 30, marker="s", edgecolors="none")
        if colorbar:
            cb = ax.figure.colorbar(sc, ax=ax, shrink=0.8, pad=0.02,
                                    label=label or ("apparent resistivity (Ω·m)"
                                                    if values is None else ""))
            if log:
                _log_colorbar_ticks(cb)
        ex = self.electrodes[self.used_electrodes(), 0]
        ax.plot(ex, np.zeros_like(ex), "v", color="k", ms=3, clip_on=False)
        ax.set_xlim(self.electrodes[:, 0].min() - self.spacing,
                    self.electrodes[:, 0].max() + self.spacing)
        ax.set_ylim(z.max() * 1.1, 0)
        ax.set_xlabel("position along line (m)")
        ax.set_ylabel("pseudo-depth (m)")
        return ax

    def summary(self) -> pd.Series:
        d = self.data
        return pd.Series({
            "array": self.array, "measurements": len(self),
            "electrodes": self.n_electrodes, "electrodes_used": self.used_electrodes().size,
            "spacing_m": self.spacing,
            "line_length_m": float(np.ptp(self.electrodes[:, 0])),
            "rhoa_min": float(d["rhoa"].min()), "rhoa_median": float(d["rhoa"].median()),
            "rhoa_max": float(d["rhoa"].max()),
            "max_pseudo_depth_m": float(d["depth"].max()),
        })


def _log_colorbar_ticks(cb) -> None:
    """Label a logarithmic colorbar 1-2-5 in plain numbers (10, 20, 50, 100 ...).

    Matplotlib's default labels every minor tick as ``6x10^1`` when the range
    is under about two decades, which piles the labels on top of each other.
    """
    from matplotlib.ticker import LogLocator, NullFormatter, StrMethodFormatter

    cb.ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    cb.ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    cb.ax.yaxis.set_minor_formatter(NullFormatter())


# ---------------------------------------------------------------------------
# Reading RES2DINV files
# ---------------------------------------------------------------------------


def _numbers(line: str) -> list[float]:
    out = []
    for tok in line.replace(",", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            return []
    return out


def read_res2dinv(path: str | Path, *, n_electrodes: int | None = None,
                  first_electrode: int = 0) -> ERTData:
    """Read a RES2DINV ``.dat`` file.

    Standard arrays (types 1-7) and the general array (type 11) are read.
    Every electrode position is snapped to the nearest multiple of the unit
    spacing in line 2 -- which repairs the non-integer ``n`` the SuperSting
    writes -- and positions that do not land within 2 % of a spacing of a
    grid point are an error rather than being silently moved.

    Parameters
    ----------
    n_electrodes
        If given, also :meth:`~ERTData.trim` the measurements to this many
        electrodes starting at ``first_electrode``. Leave it out to see
        everything in the file first.
    first_electrode
        Index of the first electrode in the ground, counting from the
        left-most position in the file.
    """
    path = Path(path)
    lines = [ln.strip() for ln in path.read_text(errors="replace").splitlines()]
    lines = [ln for ln in lines if ln != ""]
    title = lines[0]
    spacing = float(lines[1].split()[0])
    code = int(float(lines[2].split()[0]))
    if code not in ARRAYS:
        raise ValueError(f"{path.name}: unknown RES2DINV array type {code}")
    meta: dict[str, Any] = {"source_file": path.name, "array_code": code}
    raw_positions = []
    values = []
    if code == 11:
        # general array: sub-type, a text line, measurement type, count,
        # x-location type, IP flag, then "4 xa za xb zb xm zm xn zn value"
        meta["subtype"] = int(float(lines[3].split()[0]))
        i = 4
        if not _numbers(lines[i]):
            i += 1  # the "Type of measurement ..." text line
        meas_type = int(float(lines[i].split()[0]))
        count = int(float(lines[i + 1].split()[0]))
        meta["x_location"] = int(float(lines[i + 2].split()[0]))
        ip = int(float(lines[i + 3].split()[0]))
        start = i + 4
        for ln in lines[start:start + count]:
            v = _numbers(ln)
            ne = int(v[0])
            pts = np.array(v[1:1 + 2 * ne]).reshape(ne, 2)
            full = [tuple(pts[0]), (np.nan, np.nan), tuple(pts[1] if ne > 1 else (np.nan,) * 2),
                    (np.nan, np.nan)]
            if ne == 4:
                full = [tuple(p) for p in pts]
            elif ne == 3:  # pole-dipole: A M N
                full = [tuple(pts[0]), (np.nan, np.nan), tuple(pts[1]), tuple(pts[2])]
            raw_positions.append(full)
            values.append(v[1 + 2 * ne])
        if meas_type == 1:
            meta["values_were"] = "resistance"
    else:
        count = int(float(lines[3].split()[0]))
        meta["x_location"] = int(float(lines[4].split()[0]))
        ip = int(float(lines[5].split()[0]))
        start = 6 + (4 if ip else 0)  # IP adds four lines of chargeability settings
        meas_type = 0
        for ln in lines[start:start + count]:
            v = _numbers(ln)
            if code in (3, 6, 7):
                x, a, n, rho = v[:4]
            else:
                x, a, rho = v[:3]
                n = 1.0
            n_eff = abs(n)
            lay = _LAYOUT[code](n_eff)
            finite = [o for o in lay if o is not None]
            mid = 0.5 * (min(finite) + max(finite))
            x0 = x - mid * a if meta["x_location"] == 1 else x
            pos = [(np.nan, np.nan) if o is None else (x0 + o * a, 0.0) for o in lay]
            if code == 6 and n < 0:  # reverse pole-dipole: mirror about the pole
                pole = pos[0][0]
                pos = [(np.nan, np.nan) if not np.isfinite(p[0]) else (2 * pole - p[0], 0.0)
                       for p in pos]
            raw_positions.append(pos)
            values.append(rho)
    if ip:
        meta["ip"] = True
    if len(values) != count:
        raise ValueError(f"{path.name}: header says {count} measurements, found {len(values)}")

    P = np.array(raw_positions, dtype=float)  # (count, 4, 2)
    xs = P[:, :, 0]
    origin = np.nanmin(xs)
    grid_idx = (xs - origin) / spacing
    snapped = np.round(grid_idx)
    off = np.abs(grid_idx - snapped)
    if np.nanmax(off) > _SNAP_TOL:
        worst = int(np.nanargmax(np.nanmax(off, axis=1)))
        raise ValueError(
            f"{path.name}: measurement {worst + 1} has an electrode "
            f"{np.nanmax(off[worst]):.2f} spacings off the {spacing:g} m grid; "
            "check the unit spacing on line 2")
    # The first position carries the file's rounding noise too; take the
    # grid origin from all electrodes, to the file's 0.1 mm precision.
    origin = round(float(np.nanmedian(xs - snapped * spacing)), 4)
    meta["snapped_max_m"] = float(np.nanmax(np.abs(xs - origin - snapped * spacing)))
    used = np.unique(snapped[np.isfinite(snapped)]).astype(int)
    n_e = int(used.max()) + 1
    electrodes = np.column_stack([origin + spacing * np.arange(n_e), np.zeros(n_e)])
    zs = P[:, :, 1]
    if code == 11 and np.nanmax(np.abs(zs)) > 0:
        meta["electrode_z_in_file"] = float(np.nanmax(np.abs(zs)))
    idx = np.where(np.isfinite(snapped), snapped, -1).astype(int)
    df = pd.DataFrame({"a": idx[:, 0], "b": idx[:, 1], "m": idx[:, 2], "n": idx[:, 3],
                       "rhoa": np.asarray(values, dtype=float)})
    out = ERTData(electrodes, df, spacing, ARRAYS[code], title, meta)
    if meas_type == 1:  # resistances in the file: convert
        out.data["r"] = out.data["rhoa"]
        out.data["rhoa"] = out.data["r"] * out.data["k"]
    trailing = [_numbers(ln) for ln in lines[start + count:]]
    if trailing and trailing[0] and trailing[0][0] != 0:
        meta["topography"] = True
    out.provenance.record("read_res2dinv", path=str(path), array=ARRAYS[code],
                          measurements=count, snapped_max_m=meta["snapped_max_m"])
    if n_electrodes is not None:
        out = out.trim(n_electrodes, first=first_electrode)
    return out


# ---------------------------------------------------------------------------
# Reading AGI SuperSting .stg files
# ---------------------------------------------------------------------------

_FEET = 0.3048
# Detect the array from the electrode layout, most specific first (an
# all-n=1 Wenner-Schlumberger file is a Wenner file).
_DETECT_ORDER = (3, 1, 7, 4, 5, 6, 2)


def read_stg(path: str | Path, *, n_electrodes: int | None = None,
             first_electrode: int = 0, spacing: float | None = None) -> ERTData:
    """Read the raw measurement file an AGI SuperSting writes (``.stg``).

    The ``.stg`` file is what the instrument records; the RES2DINV ``.dat``
    is converted from it. Reading the ``.stg`` directly keeps what the
    conversion throws away: the measured transfer resistance, the
    instrument's repeat error and the injected current for every reading,
    and the readings with negative apparent resistivity (AGI's conversion
    drops them, which hides how many there were). Electrode positions are
    absolute, so electrode numbering does not shift when readings at the
    end of the line are missing.

    Three header lines (instrument, survey period and record count, unit),
    then one comma-separated line per reading::

        1,USER,20261111,12:06:20, 4.18646E+00, 3,435, 4.81053E+01,111126,
          Ax,Ay,Az, Bx,By,Bz, Mx,My,Mz, Nx,Ny,Nz, Cmd=1,HV=258,...,Ch=1

    record, type, date, time, ``V/I`` (ohm), repeat error (per mille),
    current (mA), apparent resistivity, command file, the four electrode
    positions, then the instrument settings.

    ``rhoa`` is recomputed as ``k V/I`` from the electrode positions, in
    ohm-m whatever the file's unit; for a metre file it agrees with the
    instrument's own value to its six significant figures
    (``metadata["rhoa_file_max_rel_diff"]``). Extra columns: ``record``,
    ``time``, ``command``, ``channel``, ``current_ma``, ``stack_error``
    (relative, 0.01 = 1 %; the spread of the stacked cycles, *not* a full
    data error -- it is often zero) and ``rhoa_file``.

    Parameters
    ----------
    n_electrodes, first_electrode
        As for :func:`read_res2dinv`: trim to the electrodes in the ground.
        Electrode 0 is the position at ``x = 0`` in the file (normally the
        first takeout on the cable), not the left-most position that happens
        to be used.
    spacing
        Unit electrode spacing (m). Inferred from the positions if omitted.

    Assumes a 2-D line: every ``y`` is zero. Positions must sit on a
    regular grid; remote (pole) electrodes are not supported yet.
    """
    path = Path(path)
    text = path.read_text(errors="replace").splitlines()
    header = [ln.strip() for ln in text[:3]]
    if not header or "Sting" not in header[0]:
        raise ValueError(f"{path.name}: not an AGI Sting/SuperSting .stg file")
    unit_line = next((h for h in header if h.lower().startswith("unit")), "Unit: meter")
    unit = unit_line.split(":", 1)[1].strip().lower()
    if unit.startswith("m"):
        to_m = 1.0
    elif unit.startswith("f"):
        to_m = _FEET
    else:
        raise ValueError(f"{path.name}: unknown unit {unit!r}")
    meta: dict[str, Any] = {"source_file": path.name, "instrument": header[0], "unit": unit}
    for h in header[1:]:
        if "Records:" in h:
            meta["records_in_header"] = int(h.split("Records:")[1].split()[0])

    rows = []
    for lineno, ln in enumerate(text[3:], start=4):
        if not ln.strip():
            continue
        f = [t.strip() for t in ln.split(",")]
        if len(f) < 21:
            raise ValueError(f"{path.name} line {lineno}: expected at least 21 fields, "
                             f"found {len(f)}")
        settings = dict(t.split("=", 1) for t in f[21:] if "=" in t)
        rows.append({
            "record": int(f[0]), "time": f"{f[2]} {f[3]}", "r_file": float(f[4]),
            "stack_error": float(f[5]) / 1000.0, "current_ma": float(f[6]),
            "rhoa_file": float(f[7]), "program": f[8],
            "xyz": [float(v) for v in f[9:21]],
            "command": int(settings.get("Cmd", -1)), "channel": int(settings.get("Ch", -1)),
        })
    if not rows:
        raise ValueError(f"{path.name}: no readings")
    if "records_in_header" in meta and meta["records_in_header"] != len(rows):
        raise ValueError(f"{path.name}: header says {meta['records_in_header']} records, "
                         f"found {len(rows)}")
    P = np.array([r["xyz"] for r in rows]).reshape(-1, 4, 3) * to_m  # (N, ABMN, xyz)
    if np.any(P[:, :, 1] != 0):
        raise ValueError(f"{path.name}: electrodes have y coordinates; only a straight "
                         "2-D line along x is supported")
    if np.any(P[:, :, 2] != 0):
        meta["electrode_z_in_file"] = float(np.abs(P[:, :, 2]).max())
    xs = P[:, :, 0]
    if spacing is None:
        gaps = np.diff(np.unique(np.round(xs, 6)))
        spacing = round(float(gaps.min()), 6) if gaps.size else 1.0
    grid = xs / spacing
    snapped = np.round(grid)
    off = np.abs(grid - snapped)
    if off.max() > _SNAP_TOL:
        worst = int(np.argmax(off.max(axis=1)))
        raise ValueError(f"{path.name}: record {rows[worst]['record']} has an electrode "
                         f"{off[worst].max():.2f} spacings off the {spacing:g} m grid")
    if snapped.min() < 0:
        raise ValueError(f"{path.name}: electrodes at negative x are not supported")
    idx = snapped.astype(int)
    n_e = int(idx.max()) + 1
    electrodes = np.column_stack([spacing * np.arange(n_e), np.zeros(n_e)])
    k = geometric_factor(*(electrodes[idx[:, j]] for j in range(4)))
    r = np.array([row["r_file"] for row in rows])
    df = pd.DataFrame({"a": idx[:, 0], "b": idx[:, 1], "m": idx[:, 2], "n": idx[:, 3],
                       "rhoa": k * r})
    for col in ("record", "time", "command", "channel", "current_ma", "stack_error",
                "rhoa_file"):
        df[col] = [row[col] for row in rows]
    meta["program"] = rows[0]["program"]
    if to_m == 1.0:
        rf = df["rhoa_file"].to_numpy()
        ok = rf != 0
        meta["rhoa_file_max_rel_diff"] = float(
            np.max(np.abs(df["rhoa"][ok] - rf[ok]) / np.abs(rf[ok]))) if ok.any() else 0.0
    meta["nonpositive"] = int((df["rhoa"] <= 0).sum())
    out = ERTData(electrodes, df, spacing, "general", meta["program"], meta)
    for code in _DETECT_ORDER:
        if out._is_standard(code):
            out.array = ARRAYS[code]
            break
    out.provenance.record("read_stg", path=str(path), array=out.array,
                          measurements=len(out), nonpositive=meta["nonpositive"])
    if n_electrodes is not None:
        out = out.trim(n_electrodes, first=first_electrode)
    return out


# ---------------------------------------------------------------------------
# Measurement schemes (for synthetic surveys)
# ---------------------------------------------------------------------------


def _scheme(rows, n_electrodes, spacing, array) -> ERTData:
    rows = [r for r in rows if max(r) < n_electrodes]
    df = pd.DataFrame(rows, columns=list("abmn"))
    df["rhoa"] = 1.0
    e = np.column_stack([spacing * np.arange(n_electrodes), np.zeros(n_electrodes)])
    return ERTData(e, df, spacing, array)


def dipole_dipole(n_electrodes: int, spacing: float, *, n_max: int = 8,
                  a_max: int = 1) -> ERTData:
    """A dipole-dipole scheme: every dipole length up to ``a_max`` spacings,
    every separation up to ``n_max``. ``rhoa`` is set to 1."""
    rows = []
    for am in range(1, a_max + 1):
        for nn in range(1, n_max + 1):
            for s in range(n_electrodes):
                rows.append((s + am, s, s + am * (nn + 1), s + am * (nn + 2)))
    return _scheme(rows, n_electrodes, spacing, "dipole-dipole")


def wenner(n_electrodes: int, spacing: float, *, a_max: int | None = None) -> ERTData:
    """A Wenner-alpha scheme with every spacing that fits (or up to ``a_max``)."""
    a_max = a_max or (n_electrodes - 1) // 3
    rows = [(s, s + 3 * am, s + am, s + 2 * am)
            for am in range(1, a_max + 1) for s in range(n_electrodes)]
    return _scheme(rows, n_electrodes, spacing, "wenner")


def schlumberger(n_electrodes: int, spacing: float, *, n_max: int = 8,
                 a_max: int = 1) -> ERTData:
    """A Wenner-Schlumberger scheme (RES2DINV array type 7)."""
    rows = [(s, s + am * (2 * nn + 1), s + am * nn, s + am * (nn + 1))
            for am in range(1, a_max + 1) for nn in range(1, n_max + 1)
            for s in range(n_electrodes)]
    return _scheme(rows, n_electrodes, spacing, "wenner-schlumberger")
