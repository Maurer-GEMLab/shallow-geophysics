# Format sample corpus

Regression fixtures: small, real files from real instruments, used to prove the
drivers keep working as vendor firmware changes.

Currently holds the two G-857 layouts (`g857_exported.asc`,
`g857_manual.txt`), written from layouts specified by the instrument owner
rather than exported from hardware — so header wording on a given firmware is
still unconfirmed — and the Emlid Reach GNSS export. Everything else is
validated against the synthetic SEG-2 and SEG-Y fixtures written by
`tests/seg2_writer.py` and `tests/segy_writer.py`.

## Validated against real files but not yet committed

The Geode SEG-2 and SeisModule SEG-Y drivers were run against twelve SEG-2
shot records and eight SEG-Y records exported by a Geometrics SeisModule
Controller between September 2025 and September 2026. The synthetic
fixtures `geode_file_feet` and `segy_file` in `tests/conftest.py` reproduce
what those files actually contain:

- SEG-2: `INSTRUMENT GEOMETRICS SEISMODULES CONTROLLER 0000`, file-header
  `UNITS FEET`, per-trace `RECEIVER_LOCATION` / `SOURCE_LOCATION` as scalar
  distances along the line, `DELAY` (0 or 0.001), `CHANNEL_NUMBER`,
  `SAMPLE_INTERVAL 0.00025`, `SHOT_SEQUENCE_NUMBER`, `STACK`, `FIXED_GAIN`,
  `DESCALING_FACTOR`, and a `NOTE` block with `BASE_INTERVAL`,
  `SHOT_INCREMENT`, `PHONE_INCREMENT`.
- SEG-Y: rev 0, big-endian, IBM float, textual header naming
  `GEOMETRICS SEISMODULE CONTROLLER`, binary-header measurement system 2
  (feet), `source_x` / `group_x` as along-line distances with scalar 1,
  `field_record` = shot number, two-digit year and day-of-year in the trace
  headers.

Those records were not added here because they are not a coherent survey
and have known quality problems. A single clean shot record of each format
(65 KB SEG-2, 108 KB SEG-Y) is still wanted.

## Contributing a sample

Good samples are small (trim to a few traces or readings), from a stated
instrument and firmware version, and carry no survey data you would mind
publishing — this repository is public.

1. Add the file here, with a sibling `<name>.md` recording instrument model,
   firmware version, acquisition software, and anything unusual.
2. Add a test in `tests/test_drivers.py` asserting the values you know to be
   correct — not just that parsing succeeds.
3. Commit normally — this directory is explicitly un-ignored. `.gitignore`
   excludes `/data/` at the repo root for bulk field data, but not here. If a
   sample seems to vanish on commit, check `git check-ignore -v <path>`.

## What is most useful right now

| Priority | File | Why |
|---|---|---|
| **High** | One clean Geode SEG-2 shot record | Real-file regression for the header layout above |
| **High** | One clean SeisModule SEG-Y shot record | Same, for the SEG-Y path |
| Medium | Geode SEG-2 with geometry *not* entered | Confirms the fallback path on a real file |
| Medium | G-857 dump from the real instrument | Confirms firmware header wording |
| Medium | ATOM-1C SEG-2 export | Confirms the GPS header key |
| Medium | Trimble GNSS point export | Adds a second positions profile |
| Medium | CG-5 `.TXT` dump | Confirms column labels per firmware |
| Low | ATOM-1C native `.ATM` | Would need reverse-engineering |
