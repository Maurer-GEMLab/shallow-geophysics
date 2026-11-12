# Rerunning the ResIPy / R2 comparison

This directory reruns the comparison between `shallowgeo.ert` and ResIPy's
R2 reported in [`docs/ert-validation.md`](../../ert-validation.md). ResIPy
and R2 are used only here; `shallowgeo` does not depend on either.

| File | What it is |
|---|---|
| `r2_comparison.py` | The R2 jobs (forward models, synthetic and field inversions) and the `compare` step that turns their output into tables and figures |
| `run_all.sh` | Runs every job in parallel, then `compare` |
| `Dockerfile` | Debian bookworm, `linux/amd64`, wine 8.0, to run `R2.exe` |
| `bin/dockerwine` | Stand-in for `wine`: runs a Windows exe in the current directory inside that container |
| `results/` | The summary tables and figures from the run on 2026-10-31 (ResIPy 3.6.6, R2 v4.10) |

Output goes to `output/` (ignored by git) unless you pass another directory.

## Why a container

ResIPy ships R2 only as a Windows executable and runs it through wine on
macOS and Linux. Apple Silicon Macs without Rosetta 2 have no usable wine,
so R2 runs under wine in an amd64 container, which Docker Desktop emulates.
The script points ResIPy's wine lookup at `bin/dockerwine`. ResIPy still
builds the meshes and input files and reads the results back.

## Setup (once)

1. **A ResIPy environment.** Keep it separate from the shallowgeo
   environment. It needs pandas below 3, because ResIPy's `createSurvey`
   fails with pandas 3.

   ```bash
   micromamba create -y -p ./resipy-env -c conda-forge python=3.12 numpy scipy pandas matplotlib
   ./resipy-env/bin/python -m pip install resipy "pandas<3" pyproj
   ```

   The script imports `shallowgeo` from this checkout, so it doesn't need to
   be installed in that environment.

2. **The wine image.** Start Docker Desktop, then:

   ```bash
   docker build --platform linux/amd64 -t shallowgeo-r2wine docs/validation/ert-r2
   ```

   `bin/dockerwine` uses that tag unless `R2_IMAGE` names another one.

## Run

```bash
PYTHON=./resipy-env/bin/python docs/validation/ert-r2/run_all.sh
```

This starts five jobs at once, one container each, and then runs `compare`.
On an M1 under emulation it takes about half an hour. The field-line
inversions are the slowest, at 2-5 minutes per R2 iteration. Each job's log
is in `output/log_<job>.txt`.

To run a single job, put `bin/` on `PATH` first:

```bash
PATH="$PWD/docs/validation/ert-r2/bin:$PATH" ./resipy-env/bin/python docs/validation/ert-r2/r2_comparison.py syn_field214
```

`compare` needs no container. It uses whichever jobs have finished.
`--pygimli-csv` adds pyGIMLi's columns from the pyGIMLi comparison's
`synthetic_inversion.csv`, for example
`docs/validation/ert-pygimli/results/synthetic_inversion.csv` (see
[`../ert-pygimli/`](../ert-pygimli/README.md)).

## What to expect

`results/` holds the numbers in `docs/ert-validation.md`. A rerun with the
same ResIPy, R2 and shallowgeo versions should reproduce them: the synthetic
noise has a fixed seed, and a rerun of one forward model through these files
matched to every printed digit.

- **Forward:** R2 within about 0.5 % of shallowgeo (median), and within
  0.13-0.23 % once R2's own uniform-ground error is divided out.
- **Synthetic inversions:** both codes fit to chi² about 1, and the images
  differ by a median of 0.014-0.023 in log10.
- **Field line:** R2 stops at chi² 271 at a 3 % error, or 206 with ResIPy's
  default error model. It no more fits these data than shallowgeo or
  pyGIMLi does.

## Details that matter

- **Mesh padding.** ResIPy's quadrilateral mesh pads `elemx * pad` columns
  beyond the outer electrodes. The script keeps that at 8 for any `elemx`.
  At `elemx=2` with the default `pad=2`, the side boundary sits about 7 m
  past the line and long-offset readings come out up to 40 % low.
- **Shared model edges.** The models are evaluated at cell centres, so on
  shallowgeo's grid the interface and block edges fall on cell edges
  (interface at 1.396 m, block x 17.37-22.40 m, z 1.40-4.24 m). The R2 mesh
  gets nodes at exactly those depths, so both codes model the same ground.
- **Reading the data.** ResIPy's RES2DINV reader mishandles mid-point
  dipole-dipole files, so the script writes a, b, m, n as a `ProtocolDC`
  file. Use `ERTData.to_res2dinv(path, general=True)` if you want ResIPy to
  read a `.dat` file.
- **R2 settings.** Log data, 3 % error (`a_wgt = 0`, `b_wgt = 0.03`), no
  error reweighting (`error_mod = 0`), target RMS 1, `alpha_aniso = 4`. R2
  chooses its own smoothing at each iteration.
