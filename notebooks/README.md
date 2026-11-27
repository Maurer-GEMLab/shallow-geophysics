# Notebooks

Colab-friendly teaching notebooks. Each one runs top to bottom with
`Runtime > Run all`, installs anything it needs in its first cell, and works
both in Colab and on a local kernel from `pixi run -e dev jupyter lab`.

| Notebook | Topic | Data |
|---|---|---|
| `seismic_refraction_2layer.ipynb` | Interactive one-layer-over-half-space refraction: travel-time curves, ray paths, slope-intercept inversion of synthetic picks | synthetic |
| `refraction_field_data.ipynb` | Refraction end to end on your own shot records: read every shot, pick first breaks and correct them by hand, all shots on one travel-time graph, 2- or 3-layer intercept-time fit, model-versus-data residuals, dip check | Geode SEG-2 or SEG-Y field data, uploaded by the student |
| `refraction_tomography.ipynb` | Refraction tomography: two-pass picking, reciprocal-time check, shortest-path rays, Gauss-Newton inversion that finds mispicks and guides re-picking, velocity image with coverage, choosing λ, synthetic resolution test, optional pyGIMLi cross-check | `examples/data/2026-09-25-srt-line/` (16 shots, one multi-record SEG-Y), or the student's own dense line |
| `magnetics_field_data.ipynb` | Ground magnetics from a per-station table: diurnal correction from base-station re-occupations (or dropping an unverifiable correction column), repeat-reading QC, profile per line, all lines in plan view, contour map of the anomaly with and without noisy stations | A synthetic survey with base-station readings (`magnetics.synthetic_table`), or the student's own table, uploaded |
| `ert_supersting.ipynb` | Resistivity (ERT) end to end: read a SuperSting RES2DINV file, repair non-integer *n* and drop readings past the end of the cable, geometric factors and pseudosections, 2.5-D forward model, inversion of synthetic data with the line's own scheme, choosing λ, depth of investigation, checks that show the field data cannot be a ground response, and the failed field inversion | `examples/data/2610234-ert-line/` (42 electrodes, dipole-dipole), or the student's own `.dat` file |
| `ert_synthetic_inversion.ipynb` | Resistivity forward and inverse problem on synthetic ground: an air-filled cave in layered soil, clay and limestone, and soil over pinnacled, karsted limestone with a sinkhole throat and a small void; noisy synthetic data from a finer mesh (no inverse crime); a λ slider comparing true against inverted resistivity and observed against predicted data; χ², L-curve and the truth's own model error; what a smooth inversion cannot recover; a live inversion with λ and vertical smoothing | synthetic (`ert`) |
| `em_gem2.ipynb` | EM induction with a GEM-2: the platform (hand, sled, drone) and why its height matters, acquisition QC, signal against noise per frequency, transients and spikes, what frequency does and does not do (induction number, cumulative response), apparent conductivity checked against the instrument's own, zero-level offsets, drift, heading check, maps and profiles, judging the vendor's layered inversion | `examples/data/2026-11-11-em-handheld/` (five hand-carried lines over a sand trap, six frequencies), or the student's own GEM-2 export |
| `gravity_magnetics_tunnel.ipynb` | Gravity and magnetics forward and inverse problem for a 2-D tunnel (4 × 3 m, roof 4 m deep under 1 m of soil): why flat layers are invisible, prism kernels, what one station sees, noise, non-uniqueness, depth weighting, choosing β, bounds, compact (minimum-support) inversion, what the data do and do not determine | synthetic (`potentialfields`) |
| `gravity_simple_bodies.ipynb` | Gravity of simple bodies: the mathematics of the 2-D line mass and rectangular prism and the 3-D point mass, sphere and vertical cylinder, chosen so that all five anomalies match; noisy synthetic data; non-linear least-squares fits with the right shape (uncertainties, correlations, a parameter that cannot be resolved, repeat-survey scatter) and the wrong one; smooth gridded inversion in 2-D and 3-D, reading its residuals, and what gravity does and does not determine | synthetic (`potentialfields`) |
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

## A note on `refraction_tomography.ipynb`

It runs top to bottom in about ten seconds on the example line, so it can be
re-run freely while changing λ or the synthetic model. The inversion itself
needs only NumPy and SciPy; pyGIMLi is an optional last section.

Two things it is built to show. First, that tomography is also a picking
tool: the misfit of a smooth model points straight at mispicks, and the
model then guides a re-pick of every trace. Second, that the image is not the
ground — it is faded wherever no ray went, and the synthetic test at the end
lets students find out for themselves what a line of this size can and
cannot resolve.


## A note on `ert_supersting.ipynb`

Like the passive notebook, this one says no. The example line's data fail
three checks that need no inversion — apparent resistivity rising faster
with n than any ground allows, a transfer resistance that hardly falls, and
readings from electrodes that were not in the ground looking like the rest —
and the inversion then fails to fit them. The working inversion is shown on
synthetic data recorded with the same 214 measurements. It runs in about a
minute and a half.


## A note on `em_gem2.ipynb`

The reader will not run without `platform=`: a GEM-2 carried at 1 m reads
64 % of what it reads on a sled at 0.25 m over the same ground, and in HCP
the in-phase from magnetic soil changes sign in between. The notebook also
shows how to check a platform from the data alone: the GEM-2 software's own
conductivities are exactly the HCP, 1 m half-space values, which is how the
example was carried. Like the ERT and passive notebooks, it is built to say no where the
data do: the two lowest frequencies are below the noise, the frequencies
disagree about the conductivity level because of zero-level offsets (all six
are at a low induction number, so it cannot be depth), and the vendor's
layered inversion misfits its own data by over 100 %. It runs in a few
seconds.


## A note on `ert_synthetic_inversion.ipynb`

The companion to `ert_supersting.ipynb`, with the answer known. Both models
are written as `rho(X, Z)` functions whose numbers sit in two dictionaries,
so students can move the cave, fill it with water or write their own ground.
The data are computed on a mesh four times finer than the inversion grid.
Every λ is inverted to convergence rather than stopped at χ² = 1, so the
slider shows what λ alone does: at 0.1 the cave grows spurious resistive
neighbours while χ² falls below 1, and at 1000 it is a faint smear. Because
the truth is known, the notebook plots the image's own error against λ: it
is U-shaped, and χ² = 1 lands slightly on the smooth side of its minimum.
The λ sweep (nine inversions per model) takes about three minutes; after
that the slider only looks results up. Each run of the live inversion in
the last section takes a few seconds.


## A note on `gravity_simple_bodies.ipynb`

The five bodies are matched on purpose: within each dimension the anomalies
differ by about a microgal at most, against 5 µGal of noise, so every
section ends up making the same point from a different side. The fits recover
what the data determine — position, depth to the centre of mass, excess
mass — and wander along a curve for what they do not (prism width against
thickness, cylinder radius against length, sphere radius against density
contrast, which is exactly singular). Fitting the wrong shape works just as
well, and one profile across a 3-D body read as 2-D puts it a fifth too
shallow. The gridded inversion reaches χ²/N = 1 while under-fitting the
stations over the peak, which the notebook uses to make students read the
residual map rather than the misfit number. The noise seed (111) was chosen
so that the true model's own χ²/N is close to 1 for every data set; with an
unlucky draw the discrepancy principle under-fits badly, which is itself
worth showing a class by changing `SEED`. It runs in about forty seconds,
six of them per 3-D inversion set-up.
