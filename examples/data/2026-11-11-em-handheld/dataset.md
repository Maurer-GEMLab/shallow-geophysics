# Hand-carried GEM-2 over a sand trap, 2026-11-11

One Geophex GEM-2 survey (file 038), about 2.5 minutes long: five roughly
parallel lines, 15-23 m long and 2-4 m apart, walked in alternating
directions over a sand trap on the S&T old golf course, with six
frequencies logging at once. The ski was hand-carried about 1 m above the
ground, lying flat (HCP). It is the hand-carried EM example for
[`notebooks/em_gem2.ipynb`](../../../notebooks/em_gem2.ipynb).

The instrument works well at 5-63 kHz. The data are also a good teaching
case for three GEM-2 problems: frequencies below the noise, zero-level
offsets, and a vendor inversion that does not fit its own data.


## The survey

| Field | Value |
|---|---|
| Site | S&T old golf course, Rolla, MO, over a sand trap; near 37.9479 N, 91.7862 W. The same site as the ERT example [`2026-11-11-ert-line`](../2026-11-11-ert-line/dataset.md), whose resistive body is a sand trap. The files do not say whether the two surveys crossed the same trap or where its edges are |
| Date, crew | **2026-11-11** (Wednesday; field notes; the export itself has times but no date), 18:56:36-18:59:02 UTC = 12:56-12:59 CST, about 20 minutes after the ERT line finished. Crew *not recorded* |
| Purpose | Ground-conductivity mapping (EM induction) over a sand trap |
| Instrument, firmware | Geophex GEM-2, serial G2-000-7007, firmware `ARMEN.230101.31` (from the header line) |
| Export format | `038_7007_gem+inv.csv`: Geophex exporter 1.45, latitude/longitude, with the GEM-2 software's apparent conductivity, susceptibility and three-layer inversion appended. `038_7007_gps.csv`: the raw `$GPGGA` sentences, one per second, with the instrument's time stamp |
| Coils | 1.66 m transmitter-receiver separation (GEM-2 ski), **horizontal coplanar (HCP)**: ski flat (field notes) |
| Platform, height | **Hand-carried, about 1 m above the ground** (field notes). Confirmed by the GEM-2 software's own numbers; see below |
| Frequencies | 450, 1530, 5310, 8850, 18 030, 63 030 Hz |
| Sampling | 166.7 ms (6 Hz: five 30 Hz base periods per output sample); 492 samples in the file, on lines 1-5 (108, 102, 102, 90, 90) |
| Lines | Roughly parallel, azimuth about 55-60 degrees, walked alternately north-east and south-west; line 5 is shorter and at the west end. Lines are numbered in the order walked |
| Walking speed | 0.9-1.4 m/s, samples 0.18-0.22 m apart |
| Positioning | External GPS, DGPS fix (GGA quality 2) on every fix, 8-9 satellites, HDOP 0.9-1.0: about 1 m. Positions interpolated by the GEM-2 between 1 Hz fixes. `GPSalt(m)` is above the geoid (GGA altitude; geoid separation -30.5 m) |
| GPS antenna position relative to the ski | *Not recorded* (treated as zero offset) |
| Free-air or calibration reading | *None recorded* |
| Instrument state | Battery 10.4-12.2 V; internal temperature 31.1-36.0 °C during the survey (header line) |

## Files

`raw/` holds both files exactly as exported.

```python
from shallowgeo.datasets import example_data
from shallowgeo.drivers.gem2 import read_gem2, read_gem2_gps

raw = example_data("2026-11-11-em-handheld")
survey = read_gem2(raw / "038_7007_gem+inv.csv", platform="handheld",
                   sensor_height=1.0, orientation="HCP", date="2026-11-11")
gps = read_gem2_gps(raw / "038_7007_gps.csv", date="2026-11-11")   # 86 fixes
```

## The platform, and a check on it

The files do not say how the ski was carried; the field notes do
(hand-carried, HCP, about 1 m). Before those notes were to hand, the
platform was inferred from the data, and the inference agrees with them —
which is the check worth teaching for a survey whose notes are lost:

- the walking speed (0.9-1.4 m/s) and a GPS altitude that stays within
  2 m of 340 m (not a drone);
- the GEM-2 software's own apparent conductivity (`Econd_*`) is reproduced
  to within 1 % at 5-18 kHz by `shallowgeo.em.apparent_conductivity` for
  **HCP coils 1.0 m above uniform ground**. At 0.25 m (a sled) it would be a
  third lower;
- the GEM-2 software's susceptibility (`MSusc_*`) is −0.0115 × 10⁻³ SI per
  ppm of in-phase, which matches the static HCP response at about 1 m
  (−0.010 × 10⁻³ SI per ppm at 1.0 m). Below 0.59 m the sign would flip.


## Known problems

- **450 Hz and 1530 Hz are below the noise.** Uniform 20 mS/m ground would
  read about 30 ppm and 100 ppm of quadrature at these frequencies; the
  sample-to-sample noise is 230 ppm and 75 ppm. Their quadrature is
  negative as often as positive, and the GEM-2 software's `Econd` at 450 Hz
  (0-1060 mS/m) is meaningless. Use 5310 Hz and above.
- **Zero-level offsets.** Every frequency is at a low induction number
  (B < 0.15), so all should give the same apparent conductivity. They give
  7.6, 13, 23 and 30 mS/m (medians, 5-63 kHz). The in-phase sits at −340 to
  −1900 ppm where the ground would give +20 to +570. Both are instrument
  offsets (bucking-coil residual), not depth or susceptibility. The
  variations along the lines are real: 18 kHz tracks 63 kHz with
  correlation 0.92 and close to the predicted ratio. Absolute levels are not
  known without a free-air reading, and the GEM-2 software's `MSusc` at
  63 kHz (19 × 10⁻³ SI) is the in-phase offset read as susceptibility.
- **Drift or geology: cannot tell.** Line medians at 63 kHz fall steadily
  from 31.8 mS/m (line 1) to 28.8 mS/m (line 5), about 1 mS/m per minute,
  while the internal temperature rose 5 °C. The lines were walked in order
  across the site, so a trend in time is also a trend in space. No line was
  repeated. The sand trap may account for some of it (sand is resistive),
  but where its edges are relative to the lines was not recorded.
- **Transient at the start of line 3.** Samples 1-7 read about −38 000 to
  −8000 ppm in-phase on every frequency, recovering over 1.3 s, with the
  quadrature low too. Line 2 ended at the same spot six seconds earlier with
  normal readings, so it is the instrument settling or the ski being swung,
  not a buried object. Drop samples 3-1 to 3-7; `em.despike` then removes
  3-8 to 3-11 and the one-sample dips at 1-30 and 3-30.
- **Fewer samples than the header says.** The header line says 526 samples;
  the file has 492. The other 34 were presumably logged between lines.
- **GPS scatter is about 1 m** (DGPS), comparable to the line spacing.
  Path lengths summed sample to sample are 1-4 m too long; the kinks at the
  ends of lines 3 and 5 are GPS, not the operator.
- **The GEM-2 software's three-layer inversion does not fit.** `FitError`
  is 49-220 % (median 135 %), the second-layer thickness is exactly 10 m on
  89 % of samples, and the resistivities come from about 20 repeated values.
  It has fitted the zero-level offsets as layers. Do not interpret it.
- **Power-line column units.** Labelled nT in the export; the GEM-2 manual
  describes the power-line column in milligauss. Values 0.8-4.2, steady:
  no power-line problem either way.

## Expected result

At 63 kHz, with the transient and spikes removed, apparent conductivity is
26-34 mS/m (about 30-40 Ω·m), higher at the north-east ends of lines 1-3
and lower towards the north-west (lines 4-5). The lower values could be the
sand trap, but that trend is also the order in time (see Known problems). At 18 kHz the pattern is the same at
18-27 mS/m; the difference in level is the offsets.
