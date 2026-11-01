# Rerunning the pyGIMLi comparison

This directory reruns the comparison between `shallowgeo.ert` and pyGIMLi
reported in [`docs/ert-validation.md`](../../ert-validation.md). pyGIMLi is
used only here; `shallowgeo.ert` does not depend on it.

| File | What it is |
|---|---|
| `pygimli_comparison.py` | The whole comparison: a Jacobian check, forward models, synthetic and field inversions, and the follow-up checks on the field fit |
| `results/` | The tables, figures and `results.json` from the run on 2026-10-31 (pyGIMLi 1.5.4) |

Output goes to `output/` (ignored by git) unless you pass `--out`.

## Setup (once)

pyGIMLi installs from conda-forge. Keep it in its own environment:

```bash
micromamba create -y -p ./pygimli-env -c conda-forge python=3.12 pygimli matplotlib pandas scipy pyproj
```

The script imports `shallowgeo` from this checkout, so it doesn't need to be
installed in that environment.

## Run

```bash
./pygimli-env/bin/python docs/validation/ert-pygimli/pygimli_comparison.py
```

This runs every part and takes about 10 minutes on an M1. To run only some,
name them: `jacobian`, `forward`, `synthetic`, `field`. Each part merges its
numbers into `results.json` in the output directory.

`synthetic_inversion.csv` from this run also feeds the pyGIMLi columns of
the R2 comparison: `r2_comparison.py compare --pygimli-csv
docs/validation/ert-pygimli/results/synthetic_inversion.csv`.

## What's in the output

| File | Contents |
|---|---|
| `forward_comparison.csv`, `forward_pct_diff.png` | shallowgeo against pyGIMLi for a two-layer and a block model on both schemes; both against the image series for the two layers; pyGIMLi over uniform ground; pyGIMLi against itself on a coarse P1 mesh; and the effect of the nominal rather than the effective geometry |
| `synthetic_inversion.csv`, `synthetic_inversion_<scheme>.png` | Both codes inverting the same 3 % noisy block-model data; the images, the true model and the log10 difference on covered cells |
| `field_inversion.csv`, `field_inversion.png` | Both codes on the field line at several λ, and pyGIMLi with a robust (L1) data norm |
| `field_resistance_by_dipole_sep.csv` | The field line's transfer resistance by dipole length and n, the diagnostic in the dataset's `dataset.md` |
| `results.json` | All of the above, plus the Jacobian check and the field follow-up: shallowgeo's forward response of pyGIMLi's field model, both solvers on the identical grid with that model, and shallowgeo started from it |

## What to expect

`results/` holds the numbers in `docs/ert-validation.md`. The synthetic
noise has a fixed seed, so a rerun with the same versions reproduces them:
shallowgeo's to every digit, pyGIMLi's inversions to about four significant
figures (its threaded solver is not bit-for-bit repeatable).

- **Jacobian:** within 4 × 10⁻⁵ of finite differences, relative to each
  column's largest entry.
- **Forward:** shallowgeo within about 0.1-0.2 % of pyGIMLi (median) and
  0.6 % (max); each within 0.3 % of the image series.
- **Synthetic inversions:** both codes fit to chi² about 0.8-0.9, and the
  images differ by a median of 0.035-0.043 in log10.
- **Field line:** shallowgeo stops at chi² 172 and pyGIMLi at 164 (λ = 20).
  Started from pyGIMLi's model, shallowgeo reaches 173, and on that model
  the two forward solvers agree to a median 0.55 %. The misfit is in the
  data, not either code.

## Details that matter

- **Same ground in both codes.** The models are evaluated at cell centres,
  so on shallowgeo's grid the interface and block edges fall on cell edges
  (interface at 1.396 m, block x 17.37-22.40 m, z 1.40-4.24 m). pyGIMLi's
  mesh has the interface and block outlines at exactly those positions.
  The forward table also gives the difference against pyGIMLi on the nominal
  geometry (interface 1.5 m), to show what that snapping is worth.
- **pyGIMLi meshes.** Forward modelling uses a quadratic (P2) triangular
  mesh, cells of at most 0.05 m² near the line, a node 0.09 m below each
  electrode, and a world four line-lengths past each end and five deep.
  Geometric factors are analytic.
- **Inversion settings.** Log data, 3 % error, λ = 20, `zWeight` 0.5, cells
  of at most 0.25 m² in the parameter domain, which goes as deep as
  shallowgeo's mesh. pyGIMLi's chi² and RMS in the tables are recomputed
  from its response the same way as shallowgeo's.
