"""MASW on one shot record: dispersion image, pick, Vs inversion.

    python examples/scripts/masw_shot.py raw/3001.dat [--fmax 60] [--layers 6]

Best on an off-end shot with the longest record you have. Prints the
wavelength range the spread can resolve and clips the picked curve to it
before inverting, so the Vs profile is not extrapolated beyond the data.
Needs the ``masw`` extra (``disba``) for the inversion; the image and pick
work without it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import shallowgeo as sg
from shallowgeo.surfacewave import dispersion_image


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path")
    ap.add_argument("--fmin", type=float, default=5.0)
    ap.add_argument("--fmax", type=float, default=60.0)
    ap.add_argument("--vmin", type=float, default=80.0)
    ap.add_argument("--vmax", type=float, default=1200.0)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--smoothing", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    survey = sg.read(args.path)
    out = args.out or Path(args.path).parent
    print(f"{Path(args.path).name}: {survey.n_traces} traces, "
          f"{survey.duration * 1e3:.0f} ms record, offsets "
          f"{survey.offsets().min():.1f}-{survey.offsets().max():.1f} m")
    if survey.duration < 0.5:
        print("  note: records shorter than ~0.5 s clip the low-frequency "
              "surface waves; expect a usable band only above ~15 Hz")

    img = dispersion_image(survey, fmin=args.fmin, fmax=args.fmax,
                           vmin=args.vmin, vmax=args.vmax)
    lam_min, lam_max = img.metadata["wavelength_limits"]
    f_floor = img.metadata["frequency_floor"]
    print(f"  resolvable wavelengths: {lam_min:.1f}-{lam_max:.1f} m "
          f"(2 x spacing to spread length); frequency floor from record "
          f"length: {f_floor:.1f} Hz")

    curve = (img.pick(fmin=f_floor, min_quality=2.0)
             .clip_wavelength(lam_min, lam_max))
    print(f"  picked {curve}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    img.plot(ax=axes[0], curve=curve)
    axes[0].set_title("dispersion image")
    curve.plot(ax=axes[1], x="wavelength")
    axes[1].set_title("picked curve")

    try:
        from shallowgeo.surfacewave import invert_dispersion

        result = invert_dispersion(curve, args.layers, smoothing=args.smoothing)
        print(f"  inverted: {result.model}  rms {result.rms:.1f} m/s")
        result.model.plot(ax=axes[2], color="firebrick", label="inverted")
        result.initial.plot(ax=axes[2], ls="--", color="0.6", label="initial")
        axes[2].legend()
        axes[1].plot(result.predicted.wavelength, result.predicted.velocity, "-",
                     color="firebrick", label="model")
        axes[1].legend()
    except ImportError as exc:
        print(f"  skipping inversion: {exc}")
        axes[2].set_axis_off()

    fig.tight_layout()
    fig.savefig(out / (Path(args.path).stem + "_masw.png"), dpi=110)
    print(f"  figure written to {out}")


if __name__ == "__main__":
    main()
