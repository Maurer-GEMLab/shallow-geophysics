# Roadmap — phase 1

Scope: seismic refraction, MASW and passive surface wave, ground gravity,
ground magnetics. Target: usable in a fall field-methods course.

Instruments: Geometrics Geode, Geometrics ATOM-1C, Scintrex CG-5,
Geometrics G-857.

## Milestone 1 — Layer 0 and Layer 1 ✅ done

Core data model and the four instrument drivers.

- [x] `SpatialRef` — mandatory CRS + vertical datum, refuses silent mixing
- [x] `Geometry` — sources/receivers/stations, reprojection, line layout
- [x] `Provenance` — append-only lineage with `applied()` guard
- [x] `SeismicSurvey` / `PointSurvey`
- [x] Entry-point driver registry with content sniffing
- [x] SEG-2 parser (shared by Geode and ATOM)
- [x] `geode-seg2`, `atom-seg2`, `cg5`, `g857` drivers
- [x] CLI: `info`, `identify`, `read`
- [x] 57 tests, synthetic SEG-2 fixture writer

## Milestone 2 — validate against real instruments 🔜 next

**This is the gate. Everything downstream is built on assumptions that only
real files can confirm.** Nothing here needs new architecture; it needs data.

- [x] **G-857 column layout resolved.** Both layouts in use are supported and
      detected from the header line: the 5-column export
      (`Line Station Time Total_Field Quality/Signal`) and the 3-column manual
      entry (`station time value_nT`). Samples in `tests/data/`.
- [ ] Confirm the G-857 layouts against a dump from the actual instrument —
      the samples were written from a specified layout, not exported from
      hardware, so firmware-specific header wording is still unconfirmed.
- [ ] Confirm the Geode's `RECEIVER_LOCATION` / `SOURCE_LOCATION` convention
      against files from *your* Geode and firmware. Check whether geometry
      entered in the field software actually lands in the headers.
- [ ] Confirm ATOM-1C SEG-2 export: which GPS header key it writes, and
      whether `UNIT_NUMBER` carries the node serial.
- [ ] Confirm CG-5 column header text on your firmware. The `Tilt x` / `Tilt y`
      embedded-space handling is already in, but other labels may vary.
- [ ] Check the ATOM `.ATM` native format. If SEG-2 export is lossy or
      awkward at scale, `.ATM` may need reverse-engineering.
- [ ] Build `tests/data/` into a real regression corpus.
- [ ] **Verify the pixi environment actually solves on all four platforms.**
      CI currently tests only the pip path (`pip install -e ".[seismic,test]"`),
      so ADR-002's install claim rests on reading the `pgcore` build matrix, not
      on a real solve. Add a `pixi install` CI job and commit `pixi.lock` before
      any student runs the setup.

## Milestone 2b — GNSS positioning ✅ done

`shallowgeo.positions`. Consumes corrected exports; does not process RINEX
(ADR-003).

- [x] `PositionTable` — ground-mark heights and absolute UTC as enforced invariants
- [x] Profile-driven reader; a new receiver is a `PROFILES` entry, not a new reader
- [x] Required `heights_are=` declaration and antenna-height reduction
- [x] `require_status()` — gate on FIX/FLOAT, not on reported RMS
- [x] Duplicate-name and coincident-mark detection
- [x] `BaseStation` from coordinates, a one-row CSV, or a table; `apply_base_shift()`
- [x] `match_occupations()` / `attach_positions()` — UTC time-window matching
- [ ] Add a **Trimble** profile once a sample export is available
- [ ] Geoid model (GEOID18) for ellipsoidal → orthometric. Not urgent: over a
      ~200 m line the geoid gradient is negligible, so *relative* gravity is
      fine on ellipsoidal heights. Absolute values are not.

## Milestone 3 — corrections

The per-method processing that makes readings interpretable. Each correction
checks `provenance.applied()` and refuses to double-apply.

- [ ] Gravity: drift (from repeat base occupations), Longman tide, latitude,
      free-air, Bouguer slab, terrain
- [ ] Magnetics: diurnal from base station, IGRF regional removal
- [ ] Seismic: first-break picking assistance, geometry QC, trace editing
- [ ] Surface wave: dispersion via `swprocess`-style wavefield transforms

## Milestone 4 — model layer (ADR-001)

- [ ] Canonical grid on `discretize` `TensorMesh` / `TreeMesh`
- [ ] Projection operators canonical ↔ method-native, tested as adjoint pairs
- [ ] 2D profile abstraction — **note the 2.5D asymmetry for potential fields**
- [ ] netCDF/xarray model serialization

## Milestone 5 — method wrappers (Layer 2)

- [ ] Refraction → pyGIMLi `TravelTimeManager`
- [ ] Gravity, magnetics → SimPEG `potential_fields`
- [ ] Surface wave → `disba` / `evodcinv` 1D, projected onto the canonical grid

## Milestone 6 — joint modeling (Layer 3)

The layer that justifies the project. Cross-gradient structural coupling
between velocity, density, and susceptibility on the canonical grid.
An enabled capability, not a solved problem.

## Deferred

GUI (Layer 4), ERT, GPR, EM, MT, live instrument control. See
[concept-summary.md](concept-summary.md).

## Open questions

- Does the Geode write anything useful into `NOTE` that we should parse?
- Is there a Geometrics-documented SEG-2 key list, or is the community
  reverse-engineering effort the only source?
- For passive MASW, do we standardize on SPAC, ReMi, or beamforming first?
- Does the CG-5's internal `SD.` correlate with repeat-occupation scatter well
  enough to use as an inversion weight, or is repeat scatter always better?
