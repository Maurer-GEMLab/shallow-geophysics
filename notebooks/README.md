# Notebooks

Colab-friendly teaching notebooks. Each one runs top to bottom with
`Runtime > Run all`, installs anything it needs in its first cell, and works
both in Colab and on a local kernel from `pixi run -e dev jupyter lab`.

| Notebook | Topic | Data |
|---|---|---|
| `seismic_refraction_2layer.ipynb` | Interactive one-layer-over-half-space refraction: travel-time curves, ray paths, slope-intercept inversion of synthetic picks | synthetic |
| `refraction_field_data.ipynb` | Refraction end to end on your own shot records: read every shot, pick first breaks and correct them by hand, all shots on one travel-time graph, 2- or 3-layer intercept-time fit, model-versus-data residuals, dip check | Geode SEG-2 or SEG-Y field data, uploaded by the student |
| `passive_shear_wave_colab.ipynb` | Passive surface wave end to end: read and merge ATOM-1C `.atm` nodes, QC, array resolution limits, SPAC and passive-MASW dispersion, pick, invert for Vs | ATOM-1C field data, uploaded by the student |

## Planned

Each follows the same arc: forward model with sliders to build intuition,
then the same workflow on an example dataset from `../examples/data/`.

- **Dipping interface** — forward and reverse shots, apparent velocities,
  true velocity and dip.
- **MASW** — dispersion image (`surfacewave.dispersion_image`), what the
  spread can and cannot resolve, picking the fundamental mode, Vs inversion
  (`surfacewave.invert_dispersion`) and its non-uniqueness.
- **Gravity reduction** — CG-5 drift and tide, the antenna-height trap in
  GNSS positions, free-air and Bouguer.

## Conventions

- First cell: `%pip install -q shallow-geophysics[masw]` guarded so it is a
  no-op when the package is already importable.
- Units in metres and seconds throughout; say so in the first markdown cell.
- Keep outputs cleared in git except for the final figure of each section,
  so the notebook reads as a document on GitHub without running.
- Derivations in markdown with LaTeX; code cells short, one idea each.

## A note on `refraction_field_data.ipynb`

The picking step is the notebook. `PickingSession.widget()` gives one shot at
a time with its picks on it and the selected trace enlarged beside it; picks
move with a slider, or by clicking when `%matplotlib widget` is available.
Hand corrections survive a re-run of the automatic picker and are saved to a
CSV, so a student can stop and come back.

Everything after it is deliberately fast to re-run, because the intended use
is a loop: fit, look at the residuals, go back and fix the three picks the
residual plot points at, fit again. The section on choosing two layers or
three exists because the misfit always falls when a layer is added, and a
class will otherwise take that as evidence.

## A note on `passive_shear_wave_colab.ipynb`

This one is built to be able to say no. Passive surface-wave processing will
return a plausible dispersion curve from data that contain no propagating
wavefield at all, so the notebook runs two tests — does the coherency decay
with station separation, and does the wavelength move when the frequency does
— and stops at the inversion if either fails. Students should expect some
datasets to be rejected, and the last section explains what array geometry
would not have been.
