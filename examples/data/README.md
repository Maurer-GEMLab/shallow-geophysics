# Example datasets

One directory per dataset, laid out as below.

| Dataset | Method | What it is |
|---|---|---|
| [`2026-09-18-refraction-line/`](2026-09-18-refraction-line/) | Refraction | 24-channel, 4 ft spacing, 92 ft spread, 13 shots from 8 positions, Geode SEG-Y in feet |

```
examples/data/
  <slug>/                      e.g. 2026-09-campus-line-1
    dataset.md                 required: the sidecar described below
    raw/                       the instrument files exactly as exported
      3001.dat 3002.dat ...    SEG-2 (Geode) and/or
      4001.sgy 4002.sgy ...    SEG-Y (SeisModule export)
    geometry.csv               optional: surveyed receiver/shot positions
    elevations.csv             optional: per-receiver elevation, metres
    picks/                     optional: hand-checked first breaks, one CSV
                               per shot in the format pick_first_breaks() writes
    fetch.py                   only if raw/ is too large to commit
```

Root-level `/data/` in this repository is git-ignored for bulk field data;
`examples/data/` is not, so files placed here are tracked as usual.

## `dataset.md`

Written for the student who opens the folder without the instructor. All of
these, even when the answer is "not recorded":

| Field | Why it matters |
|---|---|
| Site, date, crew | Provenance; which class collected it |
| Purpose | Refraction, MASW, or both. A record cut for refraction (100-200 ms) is short for MASW. |
| Instrument, firmware | Header layout changes between firmware versions |
| Export format | SEG-2 or SEG-Y, and which software exported it |
| Length unit in the headers | `UNITS FEET` versus metres. The drivers convert, but say so. |
| Geophone type and natural frequency | 4.5 Hz for MASW; 14 or 40 Hz refraction phones lose the low frequencies |
| Spread: number of channels, spacing, first geophone position | Cross-check against `RECEIVER_LOCATION` |
| Shot positions and which file is which | The headers carry a distance; the sidecar says which end and why |
| Source | Sledgehammer on plate, weight drop, ... and the stack count |
| Sample interval and record length | Both in the headers; repeat them for the reader |
| Trigger and delay | Hammer switch, geophone trigger; `DELAY` value |
| Line orientation and topography | Azimuth, whether elevations were levelled, sloping or flat |
| Known problems | Clipped channels, dead geophones, noisy traces, weather |
| Expected result | A sentence on what the interpretation should show, if known |

## Quality bar

A dataset is worth including only if it is one coherent survey: one site,
one spread configuration (or a documented change), shots that belong
together. Records from different days with different spacings belong in
separate datasets or not at all. Data with unresolved quality problems can
still be valuable as a *teaching* example, but the problems must be listed
in `dataset.md` so a student is not left guessing whether the software or
the field crew is at fault.
