# Validating `shallowgeo.ert`

`shallowgeo.ert` is a native NumPy/SciPy implementation: 2.5-D bilinear
finite elements, a numerical geometric factor, sensitivities by reciprocity,
and smoothness-constrained Gauss-Newton inversion in log-resistivity. This
note records how it was checked against analytic solutions and against
independent codes. Neither pyGIMLi nor ResIPy is a dependency; both were run
in throwaway environments only.

## Against analytic solutions (in the test suite)

`tests/test_ert.py` checks, on every run:

| Check | Result |
|---|---|
| Wavenumber quadrature reproduces 1/r over the survey's distances | < 0.1 % |
| Uniform ground returns ρa = ρ | exact (numerical geometric factor) |
| Mesh geometric factor against the analytic k | < 5 %, typically 0.3 % |
| Two layers (50/500 and 500/50 Ω·m) against the image-series solution, dipole-dipole a = 1-3, n = 1-8 | < 1.5 % max; 0.1-0.5 % median |
| Jacobian against finite differences | 1e-5 absolute |
| Row sums of the Jacobian over uniform ground (must be 1) | 1 |
| Edwards (1977) median depths, Wenner and dipole-dipole n = 1-4 | to 0.002 a |
| Conductive block recovered from 3 % noisy synthetic data | chi² < 1.5 |

## Against pyGIMLi 1.5.4

pyGIMLi (open source) ran on its own quadratic (P2) triangular meshes,
refined at each electrode, with analytic geometric factors. The model
geometry was taken from `shallowgeo`'s cell centres, so both codes saw the
same ground: interface at 1.396 m, block at x 17.37-22.40 m, z 1.40-4.24 m.
The script and instructions for rerunning it are in
[`validation/ert-pygimli/`](validation/ert-pygimli/README.md).

**Forward modelling.** Schemes: full dipole-dipole on 42 electrodes
(a = 1-4, n = 1-8, 824 readings), and the field line's own 214 readings.

| Model | shallowgeo vs pyGIMLi: median / p95 / max |
|---|---|
| 80 over 400 Ω·m, interface 1.4 m | 0.08-0.13 % / 0.43-0.45 % / 0.46 % |
| the same with a 20 Ω·m block | 0.21-0.23 % / 0.45 % / 0.57 % |
| shallowgeo vs image series (two layers) | 0.11-0.12 % / 0.21 % / 0.23 % |
| pyGIMLi vs image series (two layers) | 0.04-0.05 % / 0.27-0.29 % / 0.29 % |

On the most extreme model either code produced (pyGIMLi's best fit to the
field data, up to 5.5 × 10⁵ Ω·m), the two forward solvers still agree to a
median 0.55 %, maximum 4.5 %.

**Inversion of synthetic data** (block model, 3 % Gaussian noise,
λ = 20, vertical weight 0.5, 3 % error):

| Scheme | shallowgeo chi² / RMS / iterations | pyGIMLi chi² / RMS / iterations | median \|Δ log10 ρ\| over covered cells | block median ρ (true 20) |
|---|---|---|---|---|
| full dipole-dipole, 824 | 0.91 / 2.9 % / 3 | 0.88 / 2.8 % / 3 | 0.043 (980 cells) | 36 / 35 Ω·m |
| field scheme, 214 | 0.91 / 2.9 % / 2 | 0.76 / 2.6 % / 3 | 0.035 (797 cells) | 37 / 33 Ω·m |

A median difference of 0.04 in log10 is 10 % in resistivity, the size of
the differences expected between two regularised inversions on different
meshes. Both recover the block in the same place, at the same smoothed
contrast.

**Field data** (`examples/data/2610234-ert-line`, 214 readings after
trimming to 42 electrodes, 3 % error):

| Run | chi² | RMS |
|---|---|---|
| shallowgeo, λ = 20 (converges in 14 iterations) | 172 | 38 % |
| pyGIMLi, λ = 20 | 164 | 36 % |
| pyGIMLi, λ = 1 | 110 | 34 % |
| pyGIMLi, λ = 20, robust (L1) data norm | 305 | 35 % |

Neither code comes near chi² = 1, and started from pyGIMLi's model
`shallowgeo` reaches the same chi² (173). The problem is the data, not the
inverter: see "Known problems" in the dataset's
[`dataset.md`](../examples/data/2610234-ert-line/dataset.md). The transfer
resistance follows the receiver channel, not the geometry, and readings with
no electrode attached look like the rest.

## Against ResIPy 3.6.6 / R2 v4.10

ResIPy ships R2 only as a Windows executable (`R2.exe`). It ran under
wine 8.0 in a `linux/amd64` Debian container (Docker Desktop, emulated on
Apple Silicon). The scripts, the Dockerfile and instructions for rerunning
are in [`validation/ert-r2/`](validation/ert-r2/README.md). ResIPy built the
meshes and input files and read the results back; its wine lookup was
pointed at a two-line shim that runs
`docker run ... wine R2.exe` in the job directory. R2 used ResIPy's
quadrilateral meshes with nodes added on the interface and block edges, so
both codes saw the same ground as in the pyGIMLi comparison.

**Forward modelling.** R2 computes ρa with the analytic geometric factor,
so it carries its own discretisation error even over uniform ground.
shallowgeo uses a numerical one and is exact there. Errors are given
against shallowgeo as median / p95 / max; the ranges cover both schemes.
R2 ran on ResIPy's default mesh density (4 elements per electrode spacing,
about 9 000 elements).

| Model | R2 as computed | R2 divided by its own uniform-ground result |
|---|---|---|
| uniform 100 Ω·m (R2 against exact) | 0.38-0.44 % / 2.0-2.7 % / 2.8 % | -- |
| 80 over 400 Ω·m, interface 1.4 m | 0.49-0.54 % / 2.4-2.7 % / 3.3 % | 0.13-0.21 % / 0.32-0.36 % / 0.46 % |
| the same with a 20 Ω·m block | 0.54-0.71 % / 2.7 % / 3.9 % | 0.21-0.23 % / 0.72-0.87 % / 1.1 % |

Once R2's geometric-factor error is divided out, the two solvers agree as
closely as shallowgeo and pyGIMLi do. On the coarser mesh used for the
inversions below (2 elements per spacing, 4 800 elements), R2's
uniform-ground error is 0.7 % median, 4.3 % max.

**Inversion of synthetic data.** This is the same block model and the same
3 % noisy data as the pyGIMLi run. R2 was set to log data, a 3 % error
(`a_wgt = 0`, `b_wgt = 0.03`), no error reweighting (`error_mod = 0`) and
target RMS 1. It used `alpha_aniso = 4`, the squared equivalent of a
vertical roughness weight of 0.5. R2 picks its own smoothing at each
iteration, while shallowgeo uses a fixed λ = 20.

| Scheme | shallowgeo chi² / RMS | R2 chi² / RMS / iterations | median \|Δ log10 ρ\| shallowgeo vs R2 (p90) | block median ρ, shallowgeo / R2 / pyGIMLi (true 20) |
|---|---|---|---|---|
| full dipole-dipole, 824 | 0.91 / 2.9 % | 1.00 / 3.0 % / 3 | 0.023 (0.043), 930 cells | 36 / 37 / 35 Ω·m |
| field scheme, 214 | 0.91 / 2.9 % | 1.00 / 3.0 % / 3 | 0.014 (0.030), 755 cells | 37 / 39 / 33 Ω·m |

The two images differ by a median of 3-5 % in resistivity, which is closer
than shallowgeo and pyGIMLi agree (about 10 %). R2 took 8-17 minutes
under emulation; shallowgeo takes seconds.

**Field data.** These are the same 214 readings; the chi² and RMS below are
computed the same way for every code, at 3 % error.

| Run | chi² | RMS |
|---|---|---|
| R2, 3 % error, no reweighting (stopped after 10 iterations: "solution not improved") | 271 | 61 % |
| R2, ResIPy defaults (0.001 Ω + 2 %, error reweighting, 10 iterations, not converged) | 206 | 36 % |

R2 does no better than shallowgeo (172) or pyGIMLi (164), and its image
has the same thin resistive layer at 0.5-1.5 m. Three independent codes
cannot fit these readings with any resistivity model.

Notes for running ResIPy on SuperSting data:

- ResIPy's RES2DINV reader (`ftype='ResInv'`) misreads the original file:
  221 electrodes, all at x ≈ 0. It also reads a standard-form file from
  `ERTData.to_res2dinv` with the right ρa but duplicate electrodes (106 for
  42). Write `to_res2dinv(path, general=True)` (array type 11), which loads
  cleanly. Alternatively, load a, b, m, n (`ERTData.data`, 0-based) as
  `ProtocolDC`.
- With pandas 3, `createSurvey` fails inside ResIPy's reciprocal-error
  code; pin `pandas<3` in the ResIPy environment.
- On a quad mesh with `elemx` below 4, raise `pad` so that `elemx * pad`
  stays at 8. Otherwise the side boundary sits about 7 m past the outer
  electrodes and long-offset readings come out up to 40 % low.
