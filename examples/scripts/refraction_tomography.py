"""Refraction tomography, end to end: read, pick, clean against a model, invert.

    python examples/scripts/refraction_tomography.py "path/to/raw/*.sgy" [--max-time 0.075] [--lam 30]

Reads every shot record (a file holding several field records is split into
one gather per shot), picks first breaks with the AIC picker, and then
alternates between inverting and cleaning: picks whose residual is an
outlier are dropped, every trace is re-picked near the clean model's
prediction, and the result is inverted again
(:func:`shallowgeo.refraction.refine_picks`). Writes the picks, the velocity
grid and the figures next to the data.

Needs a shot every one or two geophones. For a handful of shots use
``refraction_line.py`` and the intercept-time method instead.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shallowgeo.refraction import (
    PickingSession,
    fit_layers,
    load_shots,
    reciprocity,
    refine_picks,
    traveltime_tomography,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pattern", help="glob for shot records, e.g. 'raw/*.sgy'")
    ap.add_argument("--min-time", type=float, default=0.001,
                    help="ignore the first N seconds (trigger spike)")
    ap.add_argument("--max-time", type=float, default=None,
                    help="no first break later than this many seconds; see "
                         "refraction_line.py. Default: 1.5 x the longest "
                         "offset over the direct-wave velocity of a first pass.")
    ap.add_argument("--lam", type=float, default=30.0, help="regularisation strength")
    ap.add_argument("--z-weight", type=float, default=0.3,
                    help="vertical relative to horizontal smoothing")
    ap.add_argument("--depth", type=float, default=None, help="grid depth, m")
    ap.add_argument("--exclude", nargs="*", default=(),
                    help="skip files whose name contains any of these")
    ap.add_argument("--out", type=Path, default=None, help="directory for outputs")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"no files match {args.pattern!r}")
    out = args.out or Path(files[0]).parent
    out.mkdir(parents=True, exist_ok=True)

    shots = load_shots(files, exclude=args.exclude)
    print(f"{len(shots)} shots from {len(files)} file(s)")
    session = PickingSession(shots, min_time=args.min_time, max_time=args.max_time)
    if args.max_time is None:
        first = session.table()
        near = first.nsmallest(max(5, len(first) // 3), "offset")
        v0 = float((near["offset"] / near["time"]).median())
        max_time = 1.5 * float(first["offset"].max()) / v0
        print(f"direct wave about {v0:.0f} m/s -> picks bounded at {1e3 * max_time:.0f} ms")
        session.auto_pick(max_time=max_time)

    kwargs = {"lam": args.lam, "z_weight": args.z_weight}
    if args.depth is not None:
        kwargs["depth"] = args.depth
    refine_picks(session, verbose=True, **kwargs)
    table = session.table()
    session.save(out / "picks_tomography.csv")

    pairs = reciprocity(table)
    if len(pairs):
        print(f"\nreciprocal pairs: {len(pairs)}, median |dt| "
              f"{1e3 * pairs['dt'].abs().median():.2f} ms")

    layered = fit_layers(table["offset"], table["time"], 2)
    print("\ntwo-layer intercept-time fit, for comparison:")
    print(layered.summary().round(2).to_string(index=False))

    res = traveltime_tomography(table, **kwargs)
    print("\n", res.summary().to_string())

    g = res.grid
    X, Z = g.mesh()
    pd.DataFrame({"x": X.ravel(), "depth": Z.ravel(), "velocity": res.velocity.ravel(),
                  "coverage": res.coverage.ravel()}).to_csv(out / "velocity_grid.csv",
                                                             index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    res.plot(ax=axes[0], rays=True, contours=[750, 1500, 2500])
    for d in np.atleast_1d(layered.depths):
        axes[0].axhline(d, color="#eb6834", ls="--", lw=1.2,
                        label="intercept-time interface")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_title(f"velocity, lam={args.lam:g} - chi² {res.chi2:.2f}, "
                      f"RMS {1e3 * res.rms:.2f} ms")
    res.plot(ax=axes[1], field="coverage")
    axes[1].set_title("ray coverage")
    fig.tight_layout()
    fig.savefig(out / "tomography.png", dpi=110)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    res.plot_fit(axes)
    fig.tight_layout()
    fig.savefig(out / "tomography_fit.png", dpi=110)
    print(f"\npicks, velocity grid and figures written to {out}")


if __name__ == "__main__":
    main()
