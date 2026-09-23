# ATOM-1C passive dataset, June–September 2026

What is in `data/ATOM-1C_PassiveShearData.zip`, what the format is, and what
the data can and cannot support. Written up because the format had to be
reverse-engineered and because the processing conclusion is negative — both
are the kind of thing that gets rediscovered expensively.

Source: 1391 `.atm` files, 34 MB, five nodes (`101525`, `101527`, `101528`,
`101529`, `101530`), three field days.

## The `.atm` format

No published specification; the layout below was worked out from the corpus
and is pinned by `tests/test_atom_atm.py`. The reader is
`shallowgeo.drivers.atom_atm`.

**Layout.** 512-byte header of CRLF-delimited ASCII, then samples. Files are
named `<6-digit unit id><2-digit minute>.atm` and live in `YYMMDDHH` folders
named for the **UTC** hour.

**Header fields**, in order: `Atom` magic; unit id; format version (`2.00`);
header size (`0512`); ADC bits at index 9 (`24`); date `YYYY/MM/DD` and time
`HH:MM:SS` at indices 16–17. Past the ASCII block, a GPS fix written
`N3755.5010` / `W09146.5853` / `+0333.29` — **degrees and decimal minutes**,
not decimal degrees. Fields 4–5, 7–8 and 10–15 are constant across all 1391
files, so their meaning is unknown; they are preserved verbatim rather than
guessed at.

**Samples.** Four bytes each, three of which carry data:

| byte | meaning |
|---|---|
| b0 | `0x00` or `0xff`, sign extension |
| b1 | least significant byte |
| b2 | middle byte |
| b3 | most significant byte |

so the value is a signed 24-bit **little-endian** integer in b1–b3, with b0
duplicating its sign. Verified on 1.77 M samples across 120 files: zero
mismatches between b0 and the sign of the decoded value.

Reading the group as a plain little-endian `int32` is the obvious mistake and
a quiet one — it puts the sign byte in the least significant position and
returns every count 256 times too large, which leaves waveforms and spectra
looking perfectly normal.

**Sample rate is 250 Hz** and is *not* in the header. Established from the
data: a full file is 15000 samples, consecutive files within a node-hour are
stamped exactly 60 s apart, and on that basis every partial file in the corpus
ends exactly on the minute boundary where the next one begins. `read_atm`
takes `sample_rate` as a parameter so a firmware change cannot silently
rescale every derived velocity.

## The three deployments

Folders are hourly, deployments are not: one afternoon spans three folders.
`group_deployments` splits on time gaps instead of folder names.

| Date | Site | Nodes | Aperture | Layout |
|---|---|---|---|---|
| 2026-06-18 | A (37.9250 N, 91.7767 W) | 5 | 5.4 m | 2-D cluster |
| 2026-07-18 | B (37.9640 N, 91.7808 W) | 5, one dead | 6.1 m | 2-D cluster |
| 2026-09-18 | A (37.9250 N, 91.7764 W) | 5, one dies early | 77.4 m | line, ~20 m spacing, azimuth 174° |

June and July were small-aperture arrays; September was a line. June and
September are the same site, roughly 27 m apart, so in principle their
wavelength bands are complementary.

### Known problems

- **101529 stops at 16:43 on 2026-09-18**, 22 minutes into a 110-minute
  deployment. Keeping it cuts the five-node common window from 110 min to 23.
- **101530 saturates 16:16–16:19 on 2026-09-18**, hitting the 24-bit rail
  (`abs_max` = 8388608) while being planted. Its GPS is also still moving
  through that interval. Exclude those minutes.
- **101525 is dead on 2026-07-18** — coherency with every other node is
  0.00–0.01 at all frequencies, while the other four are 0.7–0.99.
- **101527 is near-incoherent on 2026-09-18**, 0.08–0.16 against the others.
- **GPS outliers**: a handful of fixes per deployment land ~1 km away. Take
  medians, not means; inlier scatter is 0.4–3.5 m.

## Why no Vs profile comes out of this

The ambient coherency is *real* — it survives every control. Time-shifting one
node by 4, 20 or 120 s, time-reversing it, or pairing nodes from different
field days all give ρ ≈ 0.00–0.02, while the synchronous pairs give 0.5–0.9.
It is genuinely simultaneous ground motion.

It is nevertheless not usable, for two separate reasons.

**The coherency does not vary with frequency.** On 2026-09-18 it sits near
0.53 at 28.6 m from 1 Hz to 16 Hz. Inverting a constant ρ through
`J0(2πfr/c)` forces `c ∝ f`, which pins the wavelength — here at ~87–146 m
while the frequency spans a factor of 24. The resulting curve rises smoothly
from 226 to 6176 m/s and looks entirely respectable. It is the array, not the
ground. This is what `wavelength_test` detects.

**On the small-aperture days it does not vary with separation either.** On
2026-06-18 the per-pair inverted velocities at 4 Hz are 14.9, 17.8, 29.5,
41.9, 53.5, 56.0 and 81.5 m/s for separations of 0.9 to 5.2 m — exactly
proportional to `r`, which is what you get when ρ is the same at every
separation. This is what `separation_test` detects.

Physically this is a partial common-mode correlation: a coherent fraction
α ≈ 0.3–0.5 shared by all nodes with near-zero phase lag (4–9° across 14.6 m),
plus node-local noise. Either broadside-incident energy or shared
non-propagating motion; a line array cannot distinguish the two, which is one
more reason not to use a line.

**What does survive.** Transient events — vehicles, footfalls — do propagate.
One at 16:37:45 moves out monotonically across the whole 77 m line at 236 m/s,
and the 101525–101528 pair (28.6 m) gives a stable 150–255 m/s from 3 to
30 Hz. Apparent phase velocities of 200–350 m/s are well supported. But the
20 m station spacing aliases everything above ~7 Hz, only one station pair has
usable event coherence, and there is no corroboration, so this is a velocity
estimate and not a profile.

## The nodes also recorded the refraction shots

The 2026-09-18 ATOM deployment (16:21–18:11 UTC) overlaps the refraction line
in `examples/data/2026-09-18-refraction-line/` (12:05–13:09 **local** =
17:05–18:09 UTC). All 11 hammer shots appear in the node records, located to
within a second of the SEG-Y header times by `passive.shot_windows`, at 12–110×
background and 27–45 dB SNR from 1 to 60 Hz.

This settles the question of whether the instruments or the site are at fault:
neither. The nodes record a hammer at 40 m offset beautifully, and the ground
carries a clean 200–300 m/s surface wave — several shots give monotonic moveout
across the full 77 m line.

It does not rescue the dataset. Four receivers give a phase-shift beam roughly
±300 m/s wide over a 620 m/s search range, which is no resolution at all, and
the 14.6 m minimum spacing aliases everything above about 7 Hz. Shot energy and
resolvable band barely overlap. **The binding constraint is the number of
receivers and their spacing, not signal quality.**

The 24-channel Geode records of the same shots do give a usable curve —
265–276 m/s at 39–60 Hz to ±23–43 m/s, reproducible between shots — but only
above about 25 Hz, because the 128 ms record cannot hold a longer wavetrain.
That is wavelengths of 4–11 m, so depths of 1.5–4 m. Below 25 Hz both
instruments fail, for opposite reasons, and that gap is exactly where the
interesting part of the profile lives.

`passive.shot_windows` exists for this case: it places windows at known shot
times so free-running nodes can serve as an active-source receiver array at
offsets the cable cannot reach. The source *position* is not needed — shifting
every receiver offset by the same amount multiplies the phase-shift transform
by a unit-modulus factor and leaves the image unchanged (verified in
`tests/test_passive.py`). What the source position decides is which receivers
are usable, since a shot inside the array means discarding one side.

## How many nodes does passive actually need?

Measured, not guessed. A synthetic wavefield with a known dispersion curve was
run through the same `shallowgeo.passive` code, varying the array and the
character of the noise. Incoherent station noise was set at a coherent
fraction of 0.4, matching what the September records show. Median error in the
recovered phase velocity, over three random seeds:

| Array | N | distinct separations | noise from all azimuths | noise from one 40° sector |
|---|---|---|---|---|
| line, uniform 20 m | 5 | 4 | 4.8 % | 245 % |
| line, log 2–50 m | 5 | 9 | 1.8 % | 262 % |
| 2-D, one ring R=15 m | 5 | 3 | 2.0 % | 55 % |
| 2-D, log 3–45 m | 5 | 7 | 1.8 % | 35 % |
| 2-D, log 2–50 m | 7 | 16 | 1.1 % | 12 % |
| 2-D, log 2–60 m | 9 | 23 | 1.1 % | **3.5 %** |
| 2-D, log 2–60 m | 13 | 37 | 0.7 % | 2.9 % |

Three things fall out of it.

**Five nodes are enough when the noise is well distributed.** Every design
except the evenly spaced line recovers the curve to about 2 %. Passive work
with five nodes is not futile; the September array simply was not one of these.

**Separations must be spaced geometrically.** An evenly spaced line puts four
of its ten pairs at nearly the same separation and spans a factor of four;
a logarithmic layout spans a factor of twenty-five with the same five nodes.
This costs nothing and is most of the difference between the first two rows.

**Directional noise is what forces the node count up.** When the energy
arrives from a single 40° sector — one road, a quiet rural site, exactly the
September case — no five-node array survives, and the improvement from five to
nine is a factor of ten in error. Nine is where it becomes reliable; seven is
usable but its worst-case error is still poor.

`passive.suggest_array(n_nodes, depth_min=…, depth_max=…)` generates a layout
on these lines: radii geometrically spaced across the wavelengths the target
depths need, azimuths spread by the golden angle so no two stations line up.

## What would have worked

- **A 2-D layout, not a line.** A line cannot resolve the arrival azimuth, and
  a wavefield arriving broadside gives near-zero phase difference across every
  pair — high coherency, no velocity information, and it looks like a strong
  measurement.
- **Geometrically spaced separations**, from about a metre to a third of the
  target depth's wavelength.
- **Nine nodes** if the site's noise is directional, which is worth measuring
  before deciding: deploy the five, compute the coherency, and see whether it
  fits a Bessel function.
- **Check coherency before pulling the nodes.** Ten minutes of computation in
  the field, against a repeat visit.
- **For active shear wave, use the Geode, not the nodes** — 24 channels beats
  5, and the only change needed is a longer record. A 1 s record on a 24-channel
  spread at 4 m spacing (92 m long) resolves wavelengths of 8–90 m, so depths of
  3–30 m, from one deployment. The nodes then become worthwhile only as extra
  long-offset channels beyond the end of the cable.

## Status in the repo

The zip stays in the git-ignored `data/`. It has **not** been promoted to
`examples/data/`: the bar there is a coherent survey with every known problem
written down, and while the problems are now written down, the September line
cannot deliver the result the example would exist to demonstrate. Worth
revisiting if a circular-array deployment is collected at the same site — June
plus a new circle at site A would make a genuinely good teaching dataset.
