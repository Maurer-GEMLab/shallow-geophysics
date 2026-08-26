# Open-Source Multi-Method Near-Surface Geophysics Stack — Project Concept Summary

## Goal
Build an open-source software stack for processing near-surface geophysical data across multiple methods: ERT, seismic refraction, MASW/SASW/passive shear waves, seismic reflection, GPR, gravity, EM, magnetotellurics, and magnetics.

Key aims:
- Ingest (and ideally directly access) data from widely used commercial hardware, which currently ships with proprietary, siloed software.
- Make **multi-sensor integration** and **flexible/joint modeling** first-class capabilities — something no individual commercial tool offers.
- Be more usable for real field data than existing open-source tools (e.g., pyGIMLi, SimPEG), which are powerful but not designed for easy hardware-data workflows the way commercial software is.
- Overall bar: better than any single commercial package specifically for multi-sensor integration and modeling flexibility, while matching (not necessarily exceeding) commercial usability per-method.

## 1) Key Challenges

**Hardware/format access**
- Vendor formats (Syscal, ABEM, GSSI .dzt, Geometrics SEG-2, EM31/34, Phoenix/Zonge MT, Scintrex CG-5, GEM magnetometers, etc.) are often undocumented, proprietary binaries that shift across firmware versions.
- Live device control requires vendor SDKs/NDAs, and vendors have little incentive to cooperate with a competing open tool — realistic scope is **ingestion of exported files**, not live acquisition control.
- Reverse-engineering formats needs a GDAL-style crowdsourced process (sample files + community-documented parsers) and a public test-file corpus for regression testing.

**Physics/modeling unification**
- ERT, seismic (travel-time/waveform), potential fields, and EM/MT are fundamentally different forward problems — no single inversion engine will handle all of them. The right approach is wrapping mature, method-specific solvers (pyGIMLi, SimPEG, ObsPy/Pyrocko-adjacent tools) rather than reimplementing them.
- Joint/multi-physics inversion (cross-gradient constraints, petrophysical coupling) is itself an open research area — should be an enabled capability, not something you promise to fully solve.

**Metadata/geometry harmonization**
- All methods reduce to "sources/receivers at positions in space and time," but each has different survey geometry conventions. CRS, vertical datum, and topography handling are common sources of real-world error and need to be consistent across methods.

**Sustainability**
- Most open academic geophysics tools lose to commercial software on polish, support, and guided workflows, not raw capability — UX and QA are chronically underfunded in grant-driven projects.
- Governance/funding model (institutional home, foundation, NumFOCUS-style sponsorship) matters as much as architecture — cite ObsPy, SimPEG, pyGIMLi as survivors due to institutional support and multiple maintainers.

## 2) Project Design

**Recommended approach: layered, multi-package architecture** (not a monolith), analogous to GDAL/OGR + QGIS Processing, or ObsPy's Stream/Trace + format plugins.

- **Layer 0 — Core data model.** Method-agnostic schema for a "geophysical survey": georeferenced source/receiver geometry, channels, units, sign conventions, processing provenance. This is the shared contract everything else reads/writes.
- **Layer 1 — I/O plugins**, one per instrument/format (e.g., `geocore-io-syscal`, `geocore-io-gssi`), registered via entry points like GDAL drivers. Most tractable, most crowdsourceable part of the project.
- **Layer 2 — Method wrappers.** Thin adapters normalizing Layer 0 data into/out of pyGIMLi (ERT/MT), SimPEG (EM/gravity/magnetics), ObsPy/Pyrocko-style tools (seismic), and existing GPR/MASW packages. Value-add is consistency and sane defaults, not new solvers.
- **Layer 3 — Multi-sensor integration.** Spatial co-registration, joint/constrained inversion utilities, shared visualization — the layer where this project actually beats individual commercial tools.
- **Layer 4 — GUI/workflow layer** (see UI discussion below).

Strategy: seek early buy-in/collaboration from pyGIMLi/SimPEG/ObsPy maintainer communities rather than competing with them on numerics — the differentiator is hardware ingestion + multi-sensor integration + UX, not new inversion algorithms.

## 3) Data Standards & Metadata to Investigate

- **SEG-Y / SEG-2** — established seismic trace-data standards, already partially supported by ObsPy/Pyrocko.
- **netCDF-CF conventions** — mature, self-describing gridded format with strong xarray ecosystem support; good for output models (resistivity/velocity/susceptibility grids). CF discrete-sampling-geometry extensions cover profile/point data too.
- **SensorML / OGC Observations & Measurements** — designed for describing arbitrary instrument metadata (make, model, calibration, geometry).
- **StationXML/FDSN conventions** (seismology) — mature template for the instrument-metadata problem.
- **PROJ/EPSG** — mandate explicit, machine-readable CRS/vertical datum on every dataset (common real-world error source).
- **STAC (SpatioTemporal Asset Catalog)** — core spec + domain extensions as a model for "core item spec + per-method extension" metadata schema; plugs into existing geospatial cataloging tooling.
- Also standardize: **source/receiver geometry tables** (usable across ERT, seismic, EM, GPR), **units/sign conventions** (vary maddeningly by vendor), and **processing provenance/lineage** (raw vs. corrected data commonly conflated — ObsPy's processing-history-in-trace-stats pattern is a good model).
- Analogy: this is similar to what ISCE/ARIA and NISAR product standards did for SAR/InSAR — a shared product/metadata spec enabling interoperability across a fragmented processor/hardware ecosystem. Worth exploring similar coordination via bodies like SEG or EarthScope-adjacent groups.

## UI/Architecture Discussion

**Clarified framing:** The GUI should be an **independent, standalone desktop application** — not embedded in QGIS. QGIS was raised only as an analogy for a *Processing-toolbox-style architecture*: many modular, chainable algorithms (filtering, cleaning, processing, inversion) operating over a shared typed data model, discoverable and composable into workflows.

**Recommendation: standalone installable desktop app**, built as a thin GUI shell (e.g., Qt/PySide, for good interop with the scientific Python stack) over the Python library/CLI core (Layers 0–3), using a QGIS-Processing-style toolbox/pipeline paradigm for chaining steps.

Reasoning:
- **Data gravity** — field files (seismic/GPR especially) are large, and fieldwork often happens with poor/no connectivity; upload-centric web apps are a poor fit.
- **Compute ownership** — inversion (3D ERT/MT/joint inversion) is heavy; standalone apps use the user's own hardware, whereas web apps require server-side compute (ongoing hosting cost/liability — a common cause of death for grant-funded tools post-funding) or WASM (still behind native solver performance).
- **Processing-toolbox UX is a solved desktop pattern** — precedents: QGIS Processing, ParaView's pipeline browser, Fiji/ImageJ's plugin ecosystem. Doesn't require a browser.
- **Distribution cost is comparable either way** — desktop packaging (conda-forge/pixi, PyInstaller/briefcase) is bounded, mostly one-time-per-release work, versus a web app's *ongoing* hosting/auth/security burden for the life of the project.

**Suggested overall build order:**
1. **Python library + CLI first** — cheapest to build, most reproducible, matches how pyGIMLi/SimPEG/ObsPy are already used (scriptable, Jupyter/HPC-compatible), captures the research/power-user/early-contributor audience.
2. **Standalone desktop GUI second** — Processing-toolbox-style front end over the library/CLI, for field practitioners and less code-comfortable users.
3. **Browser component, narrowly scoped, optional/later** — a lightweight results viewer/QC tool and public dataset gallery for outreach, teaching, and collaboration — not the primary heavy-processing environment, to avoid taking on production compute-hosting costs.

This mirrors how ObsPy, SimPEG, and pyGIMLi themselves evolved: Python library core first, with GUI/notebook layers added opportunistically.
