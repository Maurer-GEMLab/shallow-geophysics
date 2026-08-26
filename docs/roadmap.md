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

- [ ] **Verify the G-857 column layout.** `DEFAULT_COLUMNS` is a documented
      guess — Geometrics publishes no spec. Run `inspect_g857()` on a real
      dump, fix the default, add the file to `tests/data/`.
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
