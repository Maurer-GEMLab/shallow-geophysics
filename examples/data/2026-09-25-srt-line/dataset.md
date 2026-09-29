# 2026-09-25 refraction tomography line

Sixteen hammer shots into one 69 ft (21 m) spread, a shot at every other
geophone plus off-end shots at both ends, all saved by the SeisModule
Controller into a single SEG-Y file. It is the worked example for
`shallowgeo.refraction.tomography` and
[`notebooks/refraction_tomography.ipynb`](../../../notebooks/refraction_tomography.ipynb):
dense enough for a velocity image, small enough to invert in a second.

> **Fields marked _not recorded_ below were not written to the headers and
> are not known to the software.** If you ran this line, fill them in — a
> student opening this folder cannot get them anywhere else.

## The survey

| Field | Value |
|---|---|
| Site | *Not recorded* |
| Date | 2026-09-25, 12:54 to 13:12 local (trace headers, day 268) |
| Crew | *Not recorded* |
| Purpose | Seismic refraction tomography. The 128 ms record is cut for refraction and is short for MASW. |
| Instrument | Geometrics Geode, SeisModule Controller (`INSTRUMENT GEOMETRICS SEISMODULE CONTROLLER` in the textual header); firmware *not recorded* |
| Export format | SEG-Y rev 0, big-endian, IBM floating point (format code 1), 240-byte trace headers, no extended textual headers. Line number 11. |
| Length unit in the headers | **Feet.** Binary-header measurement system = 2. `shallowgeo` converts to metres on read. |
| Geophone type and natural frequency | *Not recorded* |
| Spread | 24 channels, 3 ft (0.9144 m) spacing, geophone 1 at 0 ft, geophone 24 at 69 ft (21.03 m). Along-line distances in `group_x`, `coordinate_scalar` = 1. |
| Source | *Not recorded* (hammer assumed); stack count *not recorded* |
| Sample interval and record length | 125 µs, 1024 samples = 128 ms |
| Trigger and delay | `delay recording time` = 0 on every trace. Trigger type *not recorded*. |
| Filters | None applied: `LOW CUT 0 HZ  HIGH CUT 0 HZ  NOTCH 0 HZ` |
| Line orientation and topography | *Not recorded.* No elevations were surveyed; the inversion assumes a flat line. |

## Shots

**One file, sixteen field records.** `3000.sgy` holds records 3000 to 3015,
split by the `field_record` trace-header word. `load_shots` returns one
gather per record, labelled with the record number.

Source positions are along-line distances from geophone 1. The shots walk
from the far end back to the near end, one at every other geophone
(geophones 24, 22, 20 … 2), with off-end shots beyond each end.

| Record | Source (ft) | Source (m) | Time | Note |
|---|---|---|---|---|
| 3000 | 78 | 23.77 | 12:54:10 | off-end, far side |
| 3001 | 69 | 21.03 | 12:56:39 | at geophone 24 |
| 3002 | 63 | 19.20 | 12:57:46 | at geophone 22 |
| 3003 | 57 | 17.37 | 12:58:49 | at geophone 20 |
| 3004 | 51 | 15.54 | 12:59:48 | at geophone 18 |
| 3005 | 45 | 13.72 | 13:00:42 | at geophone 16 |
| 3006 | 39 | 11.89 | 13:01:43 | at geophone 14 |
| 3007 | 33 | 10.06 | 13:02:27 | at geophone 12 |
| 3008 | 27 | 8.23 | 13:03:08 | at geophone 10 |
| 3009 | 21 | 6.40 | 13:03:54 | at geophone 8 |
| 3010 | 15 | 4.57 | 13:04:31 | at geophone 6 |
| 3011 | 9 | 2.74 | 13:05:17 | at geophone 4 |
| 3012 | 3 | 0.91 | 13:06:19 | at geophone 2 |
| 3013 | −8 | −2.44 | 13:08:57 | off-end, near side — weak, see below |
| 3014 | −8 | −2.44 | 13:11:32 | repeat of 3013 |
| 3015 | −8 | −2.44 | 13:12:22 | repeat of 3013 |

Because every shot sits on a geophone, most source–receiver pairs were
recorded in both directions, and the reciprocal times are a model-free
measure of the picking error (`reciprocity()`).

## Known problems

- **The three near-side off-end shots (3013–3015) are weak.** Their peak
  amplitude is 15–40 % of the other records', and beyond
  about 8 m the first break is lost in noise: the automatic picker lands on
  later energy and returns times of 15–40 ms where the rest of the line
  says 11–14 ms. The outlier-and-guided-repick loop keeps 10–12 picks from
  each and drops the rest. Checked and ruled out: a trigger delay and a
  mis-entered source position both fit worse than the recorded geometry.
  The three repeats do not correlate well enough with each other to stack.
- **Traces at the source are saturated.** The trace nearest each shot
  clips and picks early or late with near-zero quality. The quality filter
  removes them.
- **Reciprocal times differ by 0.5 ms (median) with a slight bias.** Picks
  from east to west come out about 0.4 ms earlier than the same pairs west
  to east. That is three samples, inside the error model used for the
  inversion, but it is the first thing to look at if a shot's residuals sit
  off zero.
- **Short for MASW.** 128 ms gives two cycles only above roughly 15 Hz.

No geophone is dead.

## Not included from the same day

The same export (`seismic_Sept252026.zip`, not in the repository) holds
two more acquisitions over what looks like the same spread. Neither is
committed; this is why.

- **Files `2.sgy`–`15.sgy`** (records 2–15, 10:11–10:34, 250 µs, 128 ms):
  shots at 0, 6, 12 … 72 ft, the geophones *between* this dataset's
  shots. Headers are consistent. `2a.sgy` is byte-identical to `2.sgy`.
  Combining it with this line would give a shot at every geophone, but
  reciprocal pairs between the two sets differ by 0.9 ms on median — a
  timing offset between the acquisitions that would need a static
  correction first. It is a good second exercise.
- **Files `21.sgy`–`34a.sgy`** (records 21–35, 12:01–12:19): **the source
  positions in the headers are mirrored.** Every record's loudest,
  earliest trace is at the *opposite* end of the spread from the header
  source position (header 0 ft → strongest at geophone 24, header 66 ft →
  geophone 2). Either the shots walked from geophone 24 while positions were
  entered from 0, or the cable was laid reversed; the first breaks cannot
  tell the two apart because the ground is close to laterally uniform.
  As read, every offset in these files is wrong. `21a.sgy` repeats
  record 21 and adds 23; `34a.sgy` is identical to `34.sgy`.

## Expected result

```bash
python examples/scripts/refraction_tomography.py \
    "examples/data/2026-09-25-srt-line/raw/*.sgy"
```

The first pass bounds the picks at 75 ms from a direct wave of about
475 m/s. 318 of 384 picks pass the quality filter; three rounds of outlier
rejection and one guided re-pick leave **292 picks from 16 shots**, and the
default inversion (`lam=30`, `z_weight=0.3`, error 0.5 ms + 3 %) gives:

| | |
|---|---|
| chi² | 0.88 |
| RMS misfit | 0.74 ms |
| Reciprocal pairs | 59, median \|dt\| 0.50 ms |
| Near-surface velocity | 300–450 m/s in the top half metre |
| 750 m/s contour | about 1.4 m deep along the whole line |
| 1500 m/s contour | 1.8 m under the eastern two-thirds, deepening to about 3 m at the west end |
| Below 5 m | 3500–5000 m/s, constrained to about 7.5 m under the middle of the line |

The two-layer intercept-time fit to the same picks gives **450 m/s over
3750 m/s with the interface at 1.7 m**, which sits between the tomogram's
750 and 1500 m/s contours: tomography smooths a sharp interface into a
gradient several metres thick, and the residuals show it — positive at
short offsets and negative at long ones, a pattern that lowering `lam`
reduces. The pyGIMLi backend on the same picks agrees with the native one
to a median of 6 % over the covered part of the grid.
