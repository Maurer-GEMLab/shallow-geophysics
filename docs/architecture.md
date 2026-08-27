# Architecture decisions

## ADR-001: Discretization for multi-method integration

**Status:** accepted, 2026-08-26
**Scope:** seismic refraction, MASW/passive surface wave, ground gravity, ground magnetics

### Context

Integration happens at the modeling level: different datasets inverted onto a
common discretized model of the subsurface, supporting 2D profiles and 3D
volumes. The question is what that discretization should be.

The framing of "finite volume vs. 3D finite difference" turns out not to apply
yet. **None of the four initial forward problems is a PDE.**

| Method | Forward problem | What it wants |
|---|---|---|
| Gravity | Integral equation, dense sensitivity matrix | Rectangular prisms |
| Magnetics | Integral equation, dense sensitivity matrix | Rectangular prisms |
| Refraction | Graph shortest-path (Dijkstra) or eikonal | Unstructured triangles/tets, topography-conforming |
| MASW / passive | 1D layered propagator matrix (disba) | No mesh at all |

So the methods split into "wants prisms" (potential fields, via SimPEG's
analytic prism formulas, which have no tetrahedral support) and "wants tets"
(refraction, via pyGIMLi, whose accuracy depends on mesh conformance to
topography). Surface wave is indifferent — it produces 1D profiles that are
*projected* onto whatever model grid exists.

Note also that pyGIMLi is not optional: **SimPEG has no seismic refraction
tomography**, so refraction forces pyGIMLi into the stack, and pyGIMLi's
unstructured preference comes with it.

### Options considered

**A. One mesh; every method conforms (octree or tensor).**
Requires replacing pyGIMLi's refraction with an eikonal solver on the
rectilinear grid and writing the tomography Jacobian ourselves. Staircases
topography, which refraction is unusually sensitive to in the near surface.
`pykonal` ships no binary wheels, so it would also reintroduce a compile step
into a stack we are deliberately keeping compiler-free. Real cost, no payoff
this year.

**B. Method-native meshes plus a canonical model grid with explicit projections.**
Each method inverts on the mesh its solver actually wants. A canonical grid
holds the physical property volumes. Each method carries a projection operator
`P: canonical → native` and its adjoint for gradients. Joint constraints
(cross-gradient, petrophysical coupling) are formulated in canonical space.
Works today against unmodified pyGIMLi and SimPEG.

**C. Everything unstructured tetrahedral.**
Polyhedral gravity is well posed analytically (Okabe 1979), but SimPEG does not
implement it, so gravity and magnetics forwards would be written from scratch.

### Decision

**Option B**, with the canonical grid as a `discretize` `TensorMesh` or
`TreeMesh`.

Reasons:

1. **Two of four methods need no projection at all.** SimPEG's gravity and
   magnetics operate natively on exactly these meshes.
2. **It is the right choice for the methods not yet in scope.** ERT, IP, FDEM,
   and MT *are* genuine PDE problems, and those are where the finite-volume
   question becomes real. `discretize` is SimPEG's FV mesh for all of them.
   Choosing the canonical grid now for what the initial four methods need would
   optimize for the case that constrains it least.
3. **It serializes cleanly.** A `TensorMesh` maps one-to-one onto
   netCDF/xarray, which the concept note already wants for model output.
4. **It keeps Option A reachable.** If refraction later moves to an eikonal
   solver on the canonical grid, nothing above the projection layer changes.

Unstructured meshes remain an interop and export target, not the canonical
store.

### Consequences

- A `shallowgeo.model` layer owns the canonical grid and the projection
  operators. It is the next thing to build after corrections.
- pyGIMLi meshes are **generated from** the canonical grid; results project
  back onto it. The projection and its adjoint must be tested as an
  adjoint pair, not just for forward accuracy.
- **"2D profile" is a workflow concept, not a uniform physics mode.** Refraction
  2D is genuinely 2D. There is *no 2D gravity or magnetics in SimPEG* — a 2D
  potential-field profile is really 2.5D and must be modelled as a thin,
  strike-elongated 3D mesh. Design the profile abstraction around this from the
  start rather than discovering it mid-semester.
- Joint inversion is an *enabled capability*, not a solved problem. The
  projection architecture is what makes cross-gradient coupling expressible;
  it does not make it work well on real data.

### Revisit when

ERT or GPR enters scope (both change the mesh calculus), or if refraction
accuracy on staircased topography turns out to be acceptable in practice — which
would make Option A's unification worth its cost.

---

## ADR-002: Python 3.12 and conda-first distribution

**Status:** accepted, 2026-08-26

### Context

The install must work on a student laptop on Windows, macOS, or Linux, with no
administrator rights and no compiler toolchain.

`pgcore` — pyGIMLi's compiled core, and the stack's only hard binary
dependency — has this build coverage at version 1.6.0:

| Channel | linux-64 | osx-64 | osx-arm64 | win-64 |
|---|---|---|---|---|
| conda (`gimli`) | py312/313/314 | py312/313/314 | py312/313/314 | **py312 only** |
| PyPI wheels | cp310–314 | cp313 | cp312 | cp312–314 |

### Decision

Pin the conda environment to **Python 3.12** — the only version with all four
platforms — and distribute with **pixi**.

The PyPI path is rejected for the class: the macOS wheels are tagged
`macosx_26_0`, so on macOS 15 or earlier pip silently falls through to a source
build requiring a C++ toolchain. That is a guaranteed week-one support burden.

pixi over bare conda/micromamba because it installs per-user, resolves a
per-platform lockfile committed to the repo, and `pixi run` needs no shell
activation — removing "which environment am I in" as a failure mode.

The **library** itself declares only `requires-python = ">=3.12"`. The `<3.13`
cap lives in `pixi.toml`, so anyone using just the readers is not held back by
an inversion dependency they did not install.

### Revisit when

`anaconda.org/gimli/pgcore` gains win-64 builds beyond py312, or pyGIMLi
publishes broadly-tagged macOS wheels.

---

## ADR-003: Consume corrected GNSS solutions; do not process RINEX

**Status:** accepted, 2026-08-27

### Context

Station positions come from Emlid Reach RS2/RS3 receivers in RTK, base plus
rover. The base is post-processed through CSRS-PPP; rovers are computed in
Emlid Studio from base and rover RINEX plus navigation files. The question is
whether this package should process RINEX itself.

### Decision

**No.** `shallowgeo.positions` consumes corrected position exports.

RTKLIB is the de facto engine — Emlid Studio wraps it, as do RTKPOST and
rtklib-py — and CSRS-PPP is a government service with no open equivalent of
comparable quality. Reimplementing either is a specialty project, contradicts
the concept note's own principle of wrapping mature solvers, and would require
compiled code, which ADR-002's compiler-free student install rules out.

RINEX belongs in **provenance, not input**: record which files and which
software produced a solution; do not reprocess them.

### The actual work is downstream of the solution

Auditing one real Emlid export surfaced six failure modes, none of which
involve GNSS processing and all of which silently corrupt a gravity survey:

1. **Antenna height.** 2.134 m, and the export does not say whether the
   heights are at the phase centre or the ground mark. At 0.3086 mGal/m that is
   0.659 mGal — against 2.914 m of total relief, i.e. 0.899 mGal of actual
   signal. The ambiguity is 73% of the measurement. Hence `heights_are=` is
   required with no default, and a `PositionTable` is ground-mark by construction.
2. **UTC offsets that change mid-occupation.** 11 of 21 rows started at
   `UTC-05:00` and ended at `UTC-04:00`. Correctly parsed: 35 s. Naively: 3635 s.
   This breaks occupation matching specifically.
3. **A station named like the base that is not the base.** `999 - Base` was a
   rover occupation 15.56 m away; the authoritative coordinate is in the
   export's `Base *` columns. Confusing them costs 1.167 m vertically —
   0.360 mGal. `apply_base_shift` refuses shifts beyond `max_shift_m`.
4. **A name reused at two marks 39 m apart**, one FIX and one FLOAT. Any
   dictionary-keyed join takes one silently; `on_duplicate="error"` is default.
5. **FLOAT solutions with deceptively good RMS.** One FLOAT row reported the
   same vertical RMS as the FIX rows. Gate on status, never on RMS.
6. **Empty projected columns.** `Easting`/`Northing`/`Elevation` present but
   entirely unpopulated; only the geographic columns held data.

### Consequences

- Formats are **profiles**, not readers. Trimble is expected to differ and
  should be a `PROFILES` entry plus header aliases.
- The base station accepts plain coordinates, a one-row CSV, or a row from a
  table — all three go through the same antenna-height declaration.
- Matching is on absolute UTC, never on station name, which makes it an
  independent check: `name_agrees` flags a station mislabelled in the meter.
- Ellipsoidal heights are kept as such. A geoid model is needed for absolute
  gravity; over a ~200 m line the geoid gradient is negligible for relative work.

### Revisit when

A receiver in use emits no post-processed export, or the course needs PPP
without internet access.
