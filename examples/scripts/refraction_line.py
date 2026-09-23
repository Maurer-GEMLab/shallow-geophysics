"""Refraction line, end to end: read every shot, pick, fit, plot.

    python examples/scripts/refraction_line.py "path/to/raw/*.dat" [--layers 2] [--min-quality 10]

Reads each shot record with ``shallowgeo`` (SEG-2 or SEG-Y, geometry and
units from the headers), picks first breaks with the AIC picker, drops
low-quality picks, merges all shots into one travel-time table, and fits a
horizontal layered model by the intercept-time method. Writes the picks as
CSV next to the data so they can be corrected by hand and re-fitted.

Assumes the shots share one spread and the ground is close to horizontally
layered. The residual plot is where that assumption gets checked.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt

import shallowgeo as sg
from shallowgeo.refraction import (
    fit_layers,
    pick_first_breaks,
    plot_picks,
    traveltime_table,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pattern", help="glob for shot records, e.g. 'raw/*.dat'")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--min-quality", type=float, default=10.0,
                    help="drop picks with energy ratio below this")
    ap.add_argument("--min-time", type=float, default=0.001,
                    help="ignore the first N seconds (trigger spike)")
    ap.add_argument("--max-time", type=float, default=None,
                    help="no first break later than this many seconds. The "
                         "longest offset over the slowest plausible velocity "
                         "is the bound; picks past it are mispicks on a later "
                         "arrival, and a few are enough to bend the fit.")
    ap.add_argument("--exclude", nargs="*", default=(),
                    help="skip files whose name contains any of these, for "
                         "dropping a record the plots show to be bad")
    ap.add_argument("--out", type=Path, default=None, help="directory for picks and figures")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if args.exclude:
        kept = [f for f in files if not any(e in Path(f).name for e in args.exclude)]
        print(f"excluded {len(files) - len(kept)} of {len(files)} files")
        files = kept
    if not files:
        raise SystemExit(f"no files match {args.pattern!r}")
    out = args.out or Path(files[0]).parent
    out.mkdir(parents=True, exist_ok=True)

    shots = []
    fig, axes = plt.subplots(1, len(files), figsize=(4.5 * len(files), 6), squeeze=False)
    for ax, f in zip(axes[0], files, strict=True):
        survey = sg.read(f)
        picks = pick_first_breaks(survey, min_time=args.min_time,
                                  max_time=args.max_time)
        picks.to_csv(out / (Path(f).stem + "_picks.csv"), index=False)
        plot_picks(survey, picks, ax=ax, tmax=min(0.1, survey.duration))
        prov = survey.provenance[0].parameters
        sources = survey.geometry.sources
        where = ", ".join(f"{x:.1f}" for x in sources["x"])
        print(f"{Path(f).name}: {survey.n_traces} traces, "
              f"{len(sources)} shot(s) at {where} m, header units "
              f"{prov['header_units']}, delay {survey.delay * 1e3:.1f} ms")
        shots.append((survey, picks))
    fig.tight_layout()
    fig.savefig(out / "shot_records.png", dpi=110)

    table = traveltime_table(*shots)
    good = table[(table["quality"] >= args.min_quality) & (table["offset"] > 0)]
    print(f"\n{len(good)} of {len(table)} picks kept (quality >= {args.min_quality})")

    model = fit_layers(good["offset"], good["time"], n_layers=args.layers)
    print("\n", model)
    print(model.summary().round(2).to_string(index=False))

    fig, (ax_t, ax_r) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    model.plot(ax=ax_t)
    ax_r.axhline(0, color="k", lw=0.8)
    ax_r.plot(model.offsets, 1e3 * model.residuals(), "o", ms=4)
    ax_r.set_ylabel("residual (ms)")
    ax_r.set_xlabel("offset (m)")
    fig.savefig(out / f"layers_{args.layers}.png", dpi=110)
    print(f"\nfigures and picks written to {out}")


if __name__ == "__main__":
    main()
