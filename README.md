# shallow-geophysics

Open-source, cross-platform tooling for near-surface geophysics: read field
instruments into one common data model, interpret each method with
transparent, teachable code, then invert several methods onto a shared
discretized model of the subsurface.

Built for a field-methods course, so the install has to work on a student
laptop — Windows, macOS, or Linux, no administrator rights, no compiler.

**Status: early development.** The Layer 0 data model, five instrument
drivers, first-break picking with 2- and 3-layer refraction interpretation,
and MASW dispersion imaging with 1D Vs inversion are implemented and tested.
Gravity and magnetic corrections, meshing, tomography, and joint inversion are
not yet built. See [docs/roadmap.md](docs/roadmap.md).

## Scope of this phase

| Method | Instrument | Driver | Interpretation |
|---|---|---|---|
| Seismic refraction | Geometrics Geode, SEG-2 export | `geode-seg2` | `shallowgeo.refraction`: picking, 2/3-layer fit |
| Seismic refraction | Geometrics Geode, SEG-Y export | `geode-segy` | same |
| Active MASW | Geometrics Geode | `geode-seg2`, `geode-segy` | `shallowgeo.surfacewave`: dispersion image, Vs inversion |
| Passive surface wave | Geometrics ATOM-1C, native `.atm` | `atom-atm` | `shallowgeo.passive`: merge, QC, SPAC, passive MASW, Vs inversion |
| Passive surface wave | Geometrics ATOM-1C, SEG-2 export | `atom-seg2` | same |
| Ground gravity | Scintrex CG-5 | `cg5` | reads; corrections not yet built |
| Ground magnetics | Geometrics G-857 | `g857` | reads; corrections not yet built |

Station positioning is handled by `shallowgeo.positions`, which consumes
*corrected* GNSS exports (Emlid Reach, and any CSV whose headers match) — it
does not process RINEX. See [ADR-003](docs/architecture.md).

ERT, GPR, EM, and MT come later; see the
[originating concept note](docs/concept-summary.md).

## Install

Students use [pixi](https://pixi.sh) — it installs into the user's home
directory, needs no admin rights, and pins an identical environment on all
four platforms from a lockfile.

```bash
# 1. install pixi (once)
curl -fsSL https://pixi.sh/install.sh | bash            # macOS / Linux
# iwr -useb https://pixi.sh/install.ps1 | iex           # Windows PowerShell

# 2. get the code and build the environment
git clone https://github.com/Maurer-GEMLab/shallow-geophysics.git
cd shallow-geophysics
pixi run verify
```

`pixi run <cmd>` activates the environment automatically, so there is no
`conda init`, no shell restart, and no "which environment am I in" failure mode.

For the readers and the refraction tools alone, plain pip works. Add the
`masw` extra for surface-wave inversion (it brings in `disba` and `numba`):

```bash
pip install -e ".[seismic]"          # readers, refraction, dispersion imaging
pip install -e ".[seismic,masw]"     # + Vs inversion
```

> **Why Python 3.12 is pinned.** `pgcore`, pyGIMLi's compiled core, is the only
> hard binary dependency in the stack, and 3.12 is the sole version with conda
> builds for all of linux-64, osx-64, osx-arm64, and win-64. The PyPI wheels are
> not a substitute: the macOS ones are tagged `macosx_26_0`, so they refuse to
> install on macOS 15 or earlier and fall back to a source build that needs a
> C++ toolchain. Re-check [anaconda.org/gimli/pgcore](https://anaconda.org/gimli/pgcore)
> before relaxing the pin in `pixi.toml`.

## Use

### Read

```python
import shallowgeo as sg

# Geode shot record, SEG-2 or SEG-Y. Geometry and length unit come from the
# headers when the operator entered them; a Geode run in feet is converted
# to metres and the conversion is recorded in provenance.
shot = sg.read("3001.dat")
print(shot.offsets())             # metres, whatever the file said
print(shot.delay)                 # trigger delay, already on shot.times()

# When the headers carry no geometry, say what the spread was (metres):
shot = sg.read("LINE1.DAT", spacing=2.0, source_offset=-1.0)

stream = shot.to_obspy()          # hand off to ObsPy for filtering

# Passive array. The ATOM-1C writes one file per node per minute into
# hourly folders; read_atm_deployment merges them onto one absolute-time
# axis, with NaN (never zero) wherever a node was not recording.
from shallowgeo.drivers.atom_atm import read_atm_deployment
noise = read_atm_deployment("ATOM-1C_PassiveShearData/")

# Gravity. The CG-5 records station numbers, never positions.
grav = sg.read("GRAV.TXT", coordinates=positions, spatial_ref=utm15)
print(grav.station_means())       # repeat-occupation scatter = your error bar

# GNSS. heights_are is required -- see below.
from shallowgeo.positions import BaseStation, apply_base_shift, attach_positions

pos = sg.read_positions("stations.csv", heights_are="antenna_phase_center")
pos = pos.require_status("FIX")                    # gate on status, not RMS
pos = apply_base_shift(pos, BaseStation.from_csv("ppp_base.csv",
                                                 heights_are="ground_mark"))
grav = attach_positions(grav, pos)                 # matched on absolute UTC
```

### Refraction

```python
from shallowgeo.refraction import (pick_first_breaks, plot_picks,
                                   traveltime_table, fit_layers)

picks = pick_first_breaks(shot)                    # AIC picker; a DataFrame
plot_picks(shot, picks, tmax=0.1)                  # look before you trust
good = picks[picks["quality"] > 10]                # energy ratio across the pick

model = fit_layers(good["offset"], good["time"], n_layers=2)
print(model)                      # velocities, thicknesses, rms misfit
print(model.summary())
model.plot()

# Several shots into one spread: merge, then fit once.
table = traveltime_table((shot_a, picks_a), (shot_b, picks_b))
model3 = fit_layers(table["offset"], table["time"], n_layers=3)

# A whole line, with the picks correctable by hand and kept together.
from pathlib import Path
from shallowgeo.refraction import PickingSession, load_shots, plot_traveltimes

session = PickingSession(load_shots(Path("raw").glob("*.sgy")), max_time=0.05)
session.widget()                  # in a notebook: correct the picks
session.use_shot("4011", False)   # reject a bad record
table = session.table()           # every shot, ready to fit
plot_traveltimes(table, model=fit_layers(table["offset"], table["time"]))
session.save("picks.csv")         # reload later with session.load(...)
```

`fit_layers` is the intercept-time method with the by-eye step automated:
every partition of the picks into branches is tried, partitions whose
velocities do not increase with depth are rejected, and the smallest misfit
wins. Pass `crossovers=[...]` to place the branch boundaries yourself
instead, which is the method as it is taught and the only way to overrule a
search that has found the wrong bend. Horizontal layers are assumed; a dipping interface needs forward and
reverse shots and is on the roadmap. Travel-time tomography
(`refraction.tomography`) is a documented placeholder for the pyGIMLi wrapper.

### MASW

```python
from shallowgeo.surfacewave import dispersion_image, invert_dispersion

img = dispersion_image(shot, fmin=5, fmax=60, vmin=80, vmax=1200)
lam_min, lam_max = img.metadata["wavelength_limits"]    # 2*dx to spread length
curve = img.pick(min_quality=2).clip_wavelength(lam_min, lam_max)
img.plot(curve=curve)

result = invert_dispersion(curve, 6, smoothing=0.3)     # needs the masw extra
print(result.model, result.rms)
result.plot()
```

The image is the phase-shift transform (Park et al., 1998). Forward
dispersion uses `disba`; the inversion is a plain least-squares fit of
log-Vs in fixed layers, with an option to free the thicknesses. A record
cut for refraction (100–200 ms) limits MASW to roughly 15 Hz and above;
record a second or more when surface waves are the goal.

### Passive surface wave

No source, so the array itself has to be interrogated before it is trusted.

```python
from shallowgeo.drivers.atom_atm import scan_atm, group_deployments, read_atm_deployment
from shallowgeo import passive as P

scan = scan_atm(Path("ATOM-1C_PassiveShearData").rglob("*.atm"))
print(group_deployments(scan))          # hourly folders -> field days

survey = read_atm_deployment("ATOM-1C_PassiveShearData/26091816")
print(P.node_coverage(survey))          # who recorded, when, how loud

layout = P.array_layout(survey)
limits = P.resolution_limits(layout)    # the wavelengths this geometry can see
print(limits.frequency_band(250))       # ...as a frequency band

# Ambient noise: SPAC, but only where its assumptions hold.
spac = P.spac_coherency(survey, window=30.0)
print(P.separation_test(spac).verdict)
curve = P.spac_dispersion(spac)         # raises if the assumptions fail

# Quiet site, busy road: use the traffic as the source instead.
events = P.detect_events(survey, threshold=3.0)
image = P.event_dispersion_image(survey, events)
```

`separation_test` and `wavelength_test` exist because passive processing
fails silently. A coherency that does not decay with station separation, or
a curve whose wavelength stays put while the frequency sweeps past it,
inverts through `J0` into a smooth rising curve that describes the array
rather than the ground. Both checks run automatically inside
`spac_dispersion`, which refuses rather than returning such a curve.

See [notebooks/passive_shear_wave_colab.ipynb](notebooks/passive_shear_wave_colab.ipynb)
for the whole workflow with the diagnostics plotted.

### Command line

```bash
shallowgeo info                   # environment + registered drivers
shallowgeo identify 4001.sgy      # which drivers claim this file
shallowgeo read 3001.dat          # summary, metadata, provenance
shallowgeo read LINE1.DAT --spacing 2.0 --source-offset -1.0
```

End-to-end scripts live in [examples/scripts/](examples/scripts/).

## How it fits together

Four layers, following the plan in the concept note:

- **Layer 0 — `shallowgeo.core`.** One method-agnostic data model.
  `Geometry` (georeferenced sources/receivers/stations), `SeismicSurvey`
  (waveforms, with `delay` and `times()`), `PointSurvey` (scalar readings
  through time), `SpatialRef` (mandatory CRS *and* vertical datum),
  `Provenance` (append-only lineage).
- **Layer 1 — `shallowgeo.drivers`.** One reader per format, registered through
  the `shallowgeo.drivers` entry-point group like GDAL drivers. External
  packages can add instruments without touching this repository.
- **Layer 2 — method modules.** `shallowgeo.refraction` and
  `shallowgeo.surfacewave` hold the classroom interpretations and the thin
  wrappers onto external solvers (`disba` today; pyGIMLi and SimPEG to come).
- **Layer 3 — joint modeling.** The shared subsurface model that makes this
  more than a format converter. Not built yet.

Three design decisions are worth knowing before reading the code:

**Provenance is not decoration.** A gravity reading is uninterpretable without
knowing which corrections are already in it, and the CG-5's own options block
changes the meaning of the numbers it exports. The `cg5` driver parses that
block and records instrument-applied tide, tilt, and terrain corrections as
provenance steps, so a later correction can refuse to double-apply. The
seismic drivers record where the geometry came from (headers or arguments)
and the unit conversion factor, for the same reason.

**Antenna height is never assumed.** `read_positions` requires
`heights_are=` — whether the file's heights sit at the ground mark or the
antenna phase centre. Exports rarely record which, and the difference is a
constant offset the size of the pole: for a 2.13 m antenna that is 0.66 mGal of
free-air error, which on a low-relief line can exceed the entire signal.
A `PositionTable` is guaranteed to hold ground-mark heights and absolute UTC.

**Every dataset declares a CRS and a vertical datum.** `Geometry` cannot be
constructed without one, and combining ellipsoidal with orthometric heights
raises rather than silently converting. `local_grid()` covers the tape-measure
survey with no GPS. Along-line distances from a Geode are placed on that grid
in metres, whatever unit the field software was set to.

The choice of discretization for joint modeling is written up in
[docs/architecture.md](docs/architecture.md).

## Examples, notebooks, data

- [notebooks/](notebooks/) — Colab-friendly teaching notebooks: an
  interactive two-layer refraction demo on synthetic data,
  [`refraction_field_data.ipynb`](notebooks/refraction_field_data.ipynb) for
  running a real refraction line end to end (pick, correct by hand, fit,
  check the residuals), and a passive surface-wave workflow.
- [examples/](examples/) — end-to-end scripts and, in `examples/data/`,
  documented example datasets. The first is
  `2026-09-18-refraction-line/`: a 24-channel, 92 ft Geode spread shot from
  eight positions, with a `dataset.md` sidecar giving the acquisition
  parameters, the known problems, and the result to expect.
- [tests/data/](tests/data/) — the format-sample corpus: small real files
  that prove a driver still parses a vendor layout.

Bulk field data under the repository's root `data/` directory is
git-ignored; example datasets belong in `examples/data/` with the metadata
sidecar described there.

## Contributing a driver

The most useful contribution, and the most tractable. A driver is a
`can_open` sniffer plus a `read` function returning a Layer 0 survey:

```python
# mypkg/reader.py
from shallowgeo.drivers import Driver

driver = Driver(
    name="my-instrument",
    description="...",
    can_open=lambda path: path.read_bytes()[:4] == b"MAGC",
    read=read_my_instrument,
    extensions=(".mid",),
    methods=("magnetics",),
)
```

```toml
# your pyproject.toml
[project.entry-points."shallowgeo.drivers"]
my-instrument = "mypkg.reader:driver"
```

Format samples for the test corpus are especially welcome — see
[tests/data/README.md](tests/data/README.md).

## Development

```bash
pixi run -e dev test
# or
pip install -e ".[seismic,masw,test]" && pytest
```

## License

MIT. See [LICENSE](LICENSE).
