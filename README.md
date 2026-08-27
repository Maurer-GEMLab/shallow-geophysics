# shallow-geophysics

Open-source, cross-platform tooling for near-surface geophysics: read field
instruments into one common data model, then invert several methods onto a
shared discretized model of the subsurface.

Built for a field-methods course, so the install has to work on a student
laptop — Windows, macOS, or Linux, no administrator rights, no compiler.

**Status: early development.** The Layer 0 data model and the four instrument
drivers below are implemented and tested. Corrections, meshing, and inversion
are not yet built. See [docs/roadmap.md](docs/roadmap.md).

## Scope of this phase

| Method | Instrument | Driver | Status |
|---|---|---|---|
| Seismic refraction | Geometrics Geode | `geode-seg2` | reads |
| Active MASW | Geometrics Geode | `geode-seg2` | reads |
| Passive surface wave | Geometrics ATOM-1C | `atom-seg2` | reads |
| Ground gravity | Scintrex CG-5 | `cg5` | reads |
| Ground magnetics | Geometrics G-857 | `g857` | reads |

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

For the readers alone, without the inversion stack, plain pip works:

```bash
pip install -e ".[seismic]"
```

> **Why Python 3.12 is pinned.** `pgcore`, pyGIMLi's compiled core, is the only
> hard binary dependency in the stack, and 3.12 is the sole version with conda
> builds for all of linux-64, osx-64, osx-arm64, and win-64. The PyPI wheels are
> not a substitute: the macOS ones are tagged `macosx_26_0`, so they refuse to
> install on macOS 15 or earlier and fall back to a source build that needs a
> C++ toolchain. Re-check [anaconda.org/gimli/pgcore](https://anaconda.org/gimli/pgcore)
> before relaxing the pin in `pixi.toml`.

## Use

```python
import shallowgeo as sg

# Refraction. Pass geometry when the Geode headers don't carry it,
# which is most of the time in a teaching lab.
shot = sg.read("LINE1.DAT", spacing=2.0, source_offset=-1.0)
print(shot.n_traces, shot.offsets())

stream = shot.to_obspy()          # hand off to ObsPy for filtering and picking

# Passive array: many node files, aligned on GPS time.
from shallowgeo.drivers.atom_seg2 import read_atom_array
noise = read_atom_array(sorted(Path("deployment").glob("*.DAT")))

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

Command line:

```bash
shallowgeo info                   # environment + registered drivers
shallowgeo identify LINE1.DAT     # which drivers claim this file
shallowgeo read LINE1.DAT --spacing 2.0 --source-offset -1.0
```

## How it fits together

Four layers, following the plan in the concept note:

- **Layer 0 — `shallowgeo.core`.** One method-agnostic data model.
  `Geometry` (georeferenced sources/receivers/stations), `SeismicSurvey`
  (waveforms), `PointSurvey` (scalar readings through time), `SpatialRef`
  (mandatory CRS *and* vertical datum), `Provenance` (append-only lineage).
- **Layer 1 — `shallowgeo.drivers`.** One reader per format, registered through
  the `shallowgeo.drivers` entry-point group like GDAL drivers. External
  packages can add instruments without touching this repository.
- **Layer 2 — method wrappers.** Thin adapters onto pyGIMLi (refraction),
  SimPEG (gravity, magnetics), and disba/evodcinv (surface wave). Not built yet.
- **Layer 3 — joint modeling.** The shared subsurface model that makes this
  more than a format converter. Not built yet.

Two design decisions are worth knowing before reading the code:

**Provenance is not decoration.** A gravity reading is uninterpretable without
knowing which corrections are already in it, and the CG-5's own options block
changes the meaning of the numbers it exports. The `cg5` driver parses that
block and records instrument-applied tide, tilt, and terrain corrections as
provenance steps, so a later correction can refuse to double-apply.

**Antenna height is never assumed.** `read_positions` requires
`heights_are=` — whether the file's heights sit at the ground mark or the
antenna phase centre. Exports rarely record which, and the difference is a
constant offset the size of the pole: for a 2.13 m antenna that is 0.66 mGal of
free-air error, which on a low-relief line can exceed the entire signal.
A `PositionTable` is guaranteed to hold ground-mark heights and absolute UTC.

**Every dataset declares a CRS and a vertical datum.** `Geometry` cannot be
constructed without one, and combining ellipsoidal with orthometric heights
raises rather than silently converting. `local_grid()` covers the tape-measure
survey with no GPS.

The choice of discretization for joint modeling is written up in
[docs/architecture.md](docs/architecture.md).

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
pip install -e ".[seismic,test]" && pytest
```

## License

MIT. See [LICENSE](LICENSE).
