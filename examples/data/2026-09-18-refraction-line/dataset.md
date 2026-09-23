# 2026-09-18 refraction line

A 92 ft (28 m) refraction spread shot from eight source positions in one
hour, exported as SEG-Y from the SeisModule Controller. It is the worked
example for `shallowgeo.refraction`: forward and reverse off-end shots,
mid-spread shots, one repeat pair, and one bad record left in on purpose.

> **Fields marked _not recorded_ below were not written to the headers and
> are not known to the software.** If you ran this line, fill them in — a
> student opening this folder cannot get them anywhere else.

## The survey

| Field | Value |
|---|---|
| Site | *Not recorded* |
| Date | 2026-09-18, 12:05 to 13:09 local (trace headers, day 261) |
| Crew | *Not recorded* |
| Purpose | Seismic refraction. The 128 ms record is cut for refraction and is short for MASW — see "Known problems". |
| Instrument | Geometrics Geode, SeisModule Controller (`INSTRUMENT GEOMETRICS SEISMODULE CONTROLLER` in the textual header); firmware *not recorded* |
| Export format | SEG-Y rev 0, big-endian, IBM floating point (format code 1), 240-byte trace headers, no extended textual headers |
| Length unit in the headers | **Feet.** Binary-header measurement system = 2. `shallowgeo` converts to metres on read and records the factor in provenance. |
| Geophone type and natural frequency | *Not recorded* |
| Spread | 24 channels, 4 ft (1.2192 m) spacing, geophone 1 at 0 ft, geophone 24 at 92 ft (28.042 m). Positions are in `group_x` with `coordinate_scalar` = 1; they are along-line distances, not coordinates. |
| Source | *Not recorded* (hammer assumed); stack count *not recorded* |
| Sample interval and record length | 125 µs, 1024 samples = 128 ms |
| Trigger and delay | `delay recording time` = 0 on every trace. Trigger type *not recorded*. |
| Filters | None applied: `LOW CUT 0 HZ  HIGH CUT 0 HZ  NOTCH 0 HZ` |
| Line orientation and topography | *Not recorded.* No elevations were surveyed, so the interpretation below assumes a flat line. |

## Shots

Source positions are along-line distances from geophone 1, positive toward
geophone 24. Negative and >92 ft are off-end shots beyond the spread.

| File | Field record | Source (ft) | Source (m) | Time | Note |
|---|---|---|---|---|---|
| `4009.sgy` | 4009 | 108 | 32.92 | 12:05:53 | off-end, far side |
| `4010.sgy` | 4010 | 92 | 28.04 | 12:08:12 | at geophone 24 |
| `4011.sgy` | 4011 | 46 | 14.02 | 12:11:02 | mid-spread — **bad record**, see below |
| `4012.sgy` | 4012 | 0 | 0.00 | 12:14:44 | at geophone 1 |
| `4012.sgy` | 4013 | 0 | 0.00 | 12:14:44 | repeat of 4012, same file |
| `4014.sgy` | 4014 | −15 | −4.57 | 12:19:01 | off-end, near side |
| `4017.sgy` | 4017 | 112 | 34.14 | 12:41:19 | off-end, far side |
| `4017.sgy` | 4018 | 92 | 28.04 | 12:41:19 | at geophone 24, same file |
| `4019.sgy` | 4019 | 72 | 21.95 | 12:55:08 | mid-spread |
| `4020.sgy` | 4020 | 46 | 14.02 | 12:58:56 | mid-spread, good repeat of 4011 |
| `4021.sgy` | 4021 | 18 | 5.49 | 13:03:53 | mid-spread |
| `4022.sgy` | 4022 | 0 | 0.00 | 13:06:09 | at geophone 1 |
| `4023.sgy` | 4023 | −15 | −4.57 | 13:08:33 | off-end, near side |

**Two files hold two field records each.** `4012.sgy` and `4017.sgy` each
contain two shots saved together by the field software. `shallowgeo` reads
them as one `SeismicSurvey` with 24 geophones and two sources, `S1` and
`S2`; `survey.gather("S2")` pulls the second record out. Reading such a
file as a flat 48-trace record — which is what a reader that ignores
`field_record` produces — attaches the second record's traces to the first
record's source and puts every one of its offsets out by up to 6 m.

## Known problems

- **`4011.sgy` is a poor record.** Low signal-to-noise across the spread;
  the AIC picker lands on later arrivals and returns times up to 50 ms
  where its good repeat, `4020.sgy`, tops out at 16 ms from the same source
  position. It is kept deliberately: a student should see what a bad record
  looks like and learn to reject it. Drop it before fitting.
- **Record 4018 (in `4017.sgy`), channels 1 and 2, mispick at about
  100 ms** with quality 13 and 17 — above the usual quality > 10 cut, so
  the filter does not catch them. Two picks are enough to bend the fitted
  refractor branch badly. Look at the record.
- **The zero-offset trace picks late** on several shots (for example
  channel 24 of record 4018, 11.75 ms at zero offset). The hammer blow
  saturates the geophone at the source point. Its quality is near zero, so
  the standard filter does remove it.
- **The spread is too short to constrain a third layer.** A three-layer fit
  returns a bottom velocity in the tens of thousands of m/s: the far branch
  is nearly flat over the 28 m of spread available, so its slope carries
  almost no information. Two layers is what this geometry supports.
- **Short for MASW.** 128 ms gives two cycles only above roughly 15 Hz, so
  the dispersion image is unusable below that. Record a second or more when
  surface waves are the goal.
- The original export also contained `4014a.sgy` and `4014b.sgy`, whose
  waveforms are byte-identical to `4014.sgy` and differ only in the
  field-record number (4015, 4016). They are an export artefact and are not
  included here.

No traces are clipped and no geophone is dead.

## Expected result

Drop `4011.sgy`, bound the picker at 50 ms — no first break can be later
than that here, since the longest offset is 34 m and the slowest branch
about 960 m/s — take AIC picks with quality > 10, and merge the remaining
twelve shots into one travel-time table:

```bash
python examples/scripts/refraction_line.py \
    "examples/data/2026-09-18-refraction-line/raw/*.sgy" \
    --exclude 4011 --max-time 0.05
```

265 of 288 picks survive, and the two-layer intercept-time fit gives:

| | |
|---|---|
| V1 | 958 m/s |
| V2 | 3380 m/s |
| Depth to the refractor | 6.2 m |
| RMS misfit | 1.64 ms |

**V2 is the weak number.** The refractor branch spans only the far half of
a 28 m spread, so its slope is poorly determined: plausible variations in
the pick filter move V2 between about 3200 and 4100 m/s while leaving V1
within about 20 m/s and the depth within about 1 m. Say 3000–4000 m/s
and a refractor at 6 to 7 m. Getting V2 down to one significant figure
needs a longer spread, not better picking.

Without `--max-time`, the two mispicks in record 4018 (see "Known
problems") pull V2 to 18700 m/s and the RMS to 6.8 ms. That failure is
worth showing a class before the flag is explained.

[`notebooks/refraction_field_data.ipynb`](../../../notebooks/refraction_field_data.ipynb)
reaches the same answer interactively, and derives the 50 ms bound from the
data instead of being told it: a first pass with no bound gives a direct-wave
velocity, and no first break can arrive later than the direct wave. Point its
`DATA_DIR` at `raw/` and put `"4011"` in `EXCLUDE`.

Horizontal layers are assumed. The forward and reverse off-end shots
(`4014`/`4023` against `4009`/`4017`) are there so that assumption can be
checked; a dipping-interface interpretation is not yet implemented in
`shallowgeo.refraction`.
