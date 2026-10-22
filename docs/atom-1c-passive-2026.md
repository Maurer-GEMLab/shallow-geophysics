# ATOM-1C passive datasets, June–October 2026

What is in the two ATOM-1C zips, what the format is, and what the data can
and cannot support. Written up because the format had to be
reverse-engineered and because the processing conclusion is mostly
negative — both are the kind of thing that gets rediscovered expensively.

| Zip | Files | Field days |
|---|---|---|
| `data/ATOM-1C_PassiveShearData.zip` | 1391 `.atm`, 34 MB | 2026-06-18, 07-18, 09-18 |
| `data/ATOM_updated.zip` | 1807 `.atm`, 109 MB | 2026-09-25, 10-01 |

Five nodes throughout (`101525`, `101527`, `101528`, `101529`, `101530`).

**Short version.** The October analysis of the 2-D arrays
([below](#october-2026-the-2-d-arrays)) changes the interpretation of
everything before it. The strong, coherent ambient field at both sites is
not a common-mode artefact: it is a fast wave, at least 1.5–2 km/s, carried
by the shallow bedrock. At site B (about 1.7 m of soil over rock) a surface
wave slow enough to describe the soil exists only above roughly 30 Hz, where
the ambient field no longer correlates across even 5 m. At site A (about
6 m over rock) the soil's band reaches down to 15 Hz, inside the coherent
band, but the fast rock-guided field still dominates it there. The arrays
were not too small or too few for the soil; at these sites the ambient
field the nodes record barely carries the soil's velocity.

*Barely* is not *not at all*. On the 2026-10-01 compact circle at site A the
coherency at 25–30 Hz is negative on the 6–9 m pairs and positive again on
the 13 m pairs: a Bessel function past its first zero, which no fast or
common-mode field can produce. A joint fit across the rings gives about
325 m/s at 28 Hz, a soil Vs of about 290 m/s that agrees with the Geode
MASW ([below](#the-compact-circle-does-see-the-soil-at-2729-hz)). It is one
phase velocity in a narrow band, not a profile.

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

## The three deployments in the first zip

Folders are hourly, deployments are not: one afternoon spans three folders.
`group_deployments` splits on time gaps instead of folder names, and
`split_layouts` splits further wherever the nodes' GPS positions step, which
is how the two surprises noted under the table were found.

| Date | Site | Nodes | Aperture | Layout |
|---|---|---|---|---|
| 2026-06-18 | A (37.9250 N, 91.7767 W) | 5 | 5.4 m | 2-D cluster |
| 2026-07-18 | B (37.9640 N, 91.7808 W) | 5, one dead | 6.1 m | 2-D cluster |
| 2026-09-18 | A (37.9250 N, 91.7764 W) | 5, one dies early | 77.4 m | line, ~20 m spacing, azimuth 174° |

June and July were small-aperture arrays; September was a line. June and
September are the same site, roughly 27 m apart, so in principle their
wavelength bands are complementary.

Two things in this zip are not survey data, and the original analysis
included both:

- **2026-07-18 after 17:44 UTC was recorded at base, not in the field.**
  All five nodes kept recording for 88 minutes at 37.9554 N, 91.7757 W, about
  1 km from site B. 101530 also logged 85 minutes at the same spot before the
  June 18 deployment started. The July field array is 15:38–17:00 only. The
  original July coherency, including the "101525 is dead" result below, was
  computed over both stretches with field positions, and should be redone
  on 15:38–17:00 before it is relied on.
- **2026-06-18 before 22:45 UTC** is 101530 alone at that base location.

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

> **Superseded in part.** This section was written from the June–September
> data alone. Its measurements stand; its explanation — a common-mode
> correlation between nodes — does not. The 2-D arrays of September 25 and
> October 1 show the same near-zero phase across *every* pair orientation
> and resolve a consistent arrival direction, which a common-mode signal
> cannot have, and the Sept 25 hammer shots travel at the same speeds. See
> [October 2026: the 2-D arrays](#october-2026-the-2-d-arrays).

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

At the time this was read as a partial common-mode correlation: a coherent
fraction α ≈ 0.3–0.5 shared by all nodes with near-zero phase lag (4–9°
across 14.6 m), plus node-local noise — either broadside-incident energy or
shared non-propagating motion, which a line array cannot distinguish. The
2-D arrays settle it in favour of a third option: a propagating wave so fast
that its wavelength dwarfs the array.
Either way the conclusion here holds — no soil velocity comes out of it.

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

## October 2026: the 2-D arrays

`data/ATOM_updated.zip` holds two deployments, both 2-D, at both sites, with
all five nodes recording throughout.

| Layout | Site | Common window, UTC | Separations | Coherent fraction, 3–15 Hz |
|---|---|---|---|---|
| 2026-09-25 | B (37.9641 N, 91.7775 W) | 16:10–17:52 | 6–34 m | 0.87–0.96 |
| 2026-10-01 compact | A (37.9250 N, 91.7766 W) | 15:30–17:35 | 5.6–14 m | 0.75–0.90 |
| 2026-10-01 wide | A | 17:48–19:12 | 12–39 m | 0.61–0.82 |

On October 1 the array was rearranged at about 17:37 UTC: 101525, 101528,
101529 and 101530 moved, 101527 stayed. `group_deployments` returns the day
as one deployment, so take each layout's window from `split_layouts` and
pass it to `read_atm_deployment`, which now takes positions only from files
inside the window and warns when a node's position steps within it.

### The data are clean

- Every one of the 1807 files reads; the sign-byte check passes throughout.
- Ambient levels are 35–50 dB above the digitiser floor (visible as the
  anti-alias cut above ~110 Hz), and between 3 and 10 Hz the spectra of the
  five nodes agree to about 0.5 dB.
- Saturation is limited to a handful of minutes during placement, the
  October 1 move, and hammer blows a metre from a node.
- Timing is sound. The 2026-09-25 refraction shots (12:54–13:12 local)
  appear on every node at 30× to 37,000× background, and their arrivals
  differ between nodes by a few milliseconds to a few tens of milliseconds,
  in a pattern consistent with the source moving along the line.

**Positions are the weak point.** Only the nodes' own GPS is available:
1–3 m of scatter per fix, and stationary nodes that appear to wander 5–10 m
over two hours. On the 5.6 m pairs that is a 20–50 % uncertainty in
separation. A taped layout would remove this entirely; `read_atm_deployment`
accepts `positions=` for it, and `plane_wave_fit` carries `position_error=`
into its brackets.

### What the arrays see

`plane_wave_fit` fits one plane wave per frequency to all ten pairs and
reports velocity and direction with a 68 % bracket. Over the whole
3–15 Hz band, on every layout:

- **One plane wave explains essentially all the coherency**
  (`single_wave` 0.998–1.000). The ambient field is directional, so SPAC's
  azimuthal-average assumption does not hold, whatever `separation_test`
  says.
- **The phase across the array is a few degrees in every pair orientation.**
  Residuals after the fit are mostly under 4°. The apparent velocity is at least
  1.5–2 km/s from every direction, so this is not a slow wave arriving
  broadside to a line, which was the September worry.
- **The direction is stable.** Sept 25: from 108–153° (ESE) at every
  frequency and through six 15-minute blocks. October 1, compact layout:
  from roughly north at 3–10 Hz and from 290–320° at 14–20 Hz; wide layout:
  from 230–270° above 10 Hz.

The only frequencies whose bracket is both finite and within ten apertures
of wavelength:

| Layout | Frequency | Apparent velocity | Arrives from |
|---|---|---|---|
| 2026-09-25 | 9.6–14.7 Hz | 2170–2680 m/s, ±6 % | 126–130° |
| 2026-10-01 wide | 14.7–17 Hz | 3600–4900 m/s, ±20 % | 252–272° |
| 2026-10-01 compact | none | lower bounds of 1.8–9 km/s only | |

Below these bands the wavelength is 10–100 apertures and the fit returns
lower bounds only. Above about 20 Hz the coherency falls below 0.5 at
every separation and the fit becomes unresolved or aliased.

**These are apparent velocities, and an upper bound on the Rayleigh
velocity.** A plane wave arriving steeply from below crosses the array at
V/sin(i), faster than it travels. The 3.6–4.9 km/s at site A exceeds the
rock's own P-wave velocity (3.38 km/s from the 2026-09-18 refraction line),
so it cannot be a horizontally travelling surface wave; part of the field
there arrives from below. The 2.2–2.7 km/s at site B is below that site's
rock P-wave velocity (3.75 km/s) and could be a Rayleigh wave in the rock,
but the data cannot exclude a steep body wave.

### The compact circle does see the soil, at 27–29 Hz

Above 20 Hz the coherency is weak, and the default diagnostics used to stop
there. But on the 2026-10-01 compact layout (centre node 101529, four ring
nodes at GPS radii of 5.9–8.4 m, pairs of 5.6–14.2 m) it does something
specific at 25–30 Hz. It is about −0.24 on the 6–8 m pairs and +0.23 on the
13 m pairs, and the pattern holds in both halves of the record. Coherency
that is negative at one separation and positive again at a larger one is
J0 past its first zero. It needs a wavelength comparable to the
separations, about 12 m here. A wave at 1.5 km/s or more, a common-mode
signal and incoherent noise are all positive at every separation.

`spac_dispersion` now accepts these frequencies by default (see
[below](#why-the-diagnostics-needed-fixing)):

| Frequency | c (m/s) | Coherent fraction A | Fit residual |
|---|---|---|---|
| 27.2–27.9 Hz | 322–328 | 0.67–0.74 | 0.045–0.06 |
| 28.0–28.5 Hz | 326–332 | 0.70–0.74 | 0.06–0.08 |
| 28.6–29.0 Hz | 330–334 | 0.66–0.72 | 0.08–0.10 |

That is about **325 m/s at 28 Hz**, λ ≈ 12 m, and the same within 2 m/s from
10 s and 30 s windows. Fitting 25–30 Hz by hand (`only_usable=False`) gives
366, 340, 324, 326, 334 and 338 m/s. The 25–27 Hz bins are left out by
default. Their imaginary coherency is 0.6 to more than 1 times the real
part, and an isotropic field would have averaged it away.

Interpretation, with the 2026-09-18 refraction layering (6.2 m at Vp 958 m/s
over 3380 m/s) fixed: a soil Vs of about **290 m/s** reproduces 330, 319 and
311 m/s at 27, 28 and 29 Hz. The rock Vs hardly matters here: 1.5 or
2.5 km/s changes those by 2 m/s. That is close to the Geode MASW's 265–276 m/s
at 39–60 Hz. The positions are GPS-only, and velocity scales directly with
the separations, so a ±20 % error in the ring radius moves the soil Vs to
about **250–310 m/s**. Taping the layout would remove most of that.

The same test finds nothing on the other layouts. On 2026-09-25 the outer
pairs go slightly negative at 19–21 Hz, but that is the directional 1.6 km/s
field: one plane wave makes aligned pairs negative through cos(k·Δx), and
its imaginary coherency is 0.9 of the real part. The 2026-10-01 wide
layout fits about 390 m/s at 24–26 Hz, but its closest ring is already
0.8 λ out. Rings that far out see only the Bessel tail, which is nearly
periodic in r, so the lobe is not pinned. That layout is rejected, and
the 2026-09-18 line, June and July with it.

### The hammer shots tell the same story

For the shots nearest 101527 on 2026-09-25, the 15–40 Hz envelope reaches
101525, 37 m away, only 8–24 ms later: well over 1.5 km/s. A surface wave in
the 450 m/s layer would take 80 ms or more. At that range what reaches the
far nodes is rock-guided.

### Why the diagnostics needed fixing

On these data the original `separation_test` said "SPAC assumptions hold
over most of the band" for all three layouts, and `spac_dispersion` then
returned a few stray points (500 m/s at 33 Hz; 760–920 m/s at 20–23 Hz).
The tolerance on the per-separation velocity spread (0.35) is looser than
the spread a 5.6–14 m array produces when the velocity simply scales with
separation (about 0.3), so the test could not see the failure on a compact
array. `separation_test` now also requires the slope of log velocity
against log separation to stay below 0.5 (it is 1 for that failure,
whatever the geometry). With that, both October 1 layouts fail outright,
and September 25 is "valid only in part of the band", where
`spac_dispersion` still refuses to produce a curve.

Those three checks all assume the first lobe of J0, where coherency
falls with separation. That made them blind to the compact circle's
27–29 Hz band, where coherency correctly rises again between the 8 m and
13 m rings. `separation_test` now has a second, **multi-lobe** route. A
frequency is accepted when the joint fit `A·J0(2πfr/c)` over at least
three rings meets all of these:
- residual ≤ 0.10 and A ≥ 0.3;
- some ring at −0.15 or below;
- the closest ring no further out than the first trough of J0
  (r/λ ≤ 0.61);
- no velocity 20 % away fits within 1.5× the residual;
- the rms imaginary coherency is at most 0.6 of the real.

The first-lobe checks are unchanged. `SeparationTest` reports the two
routes as `first_lobe` and `multi_lobe`. `spac_dispersion` inverts
multi-lobe frequencies with the fit only, never point-wise.

A band that short (1.06× in frequency) is too narrow for `wavelength_test`.
`spac_dispersion` reports such a curve as `plausible: None`, never `True`,
and its `wavelength_verdict` says why. If every point came in on the
multi-lobe route, the verdict says the curve cannot be a flat coherency;
otherwise it says NOT VERIFIED.

## Why a 40 m array cannot reach Vs30 here

The question is fair: Vs30 needs wavelengths up to roughly 40–90 m (depth
sensitivity is about a third to a half of the wavelength), and a 5-node
array with 6–34 m separations resolves wavelengths of about 12–100 m. On
paper the array is the right size.

What the array records, though, is frequency, and f = c/λ. At an ordinary
soil site with c ≈ 300 m/s, λ = 40 m is 7 Hz — the heart of the ambient
band — and the same array works. Here the forward model for site B (1.7 m
of soil at Vs ≈ 220 m/s over rock at Vs ≈ 2 km/s, from the 2026-09-25
refraction line) gives the fundamental Rayleigh mode as:

| f (Hz) | c (m/s) | λ (m) |
|---|---|---|
| 10 | 1840 | 184 |
| 20 | 1815 | 91 |
| 30 | 1708 | 57 |
| 40 | 646 | 16 |
| 60 | 374 | 6 |

At site B, every wavelength that carries the Vs30 information, about
6–60 m, sits between 30 and 60 Hz. Two things go wrong there:

1. **The ambient field does not correlate at those frequencies.** Coherency
   is below 0.5 at every separation above about 20 Hz on all three layouts.
   The energy there comes from nearby sources and dies within metres.
2. **The curve is nearly vertical.** Velocity triples between 30 and 40 Hz.
   A frequency-by-frequency method cannot place points on it, and the
   energy tends to show up in higher modes.

Where the ambient field *is* coherent, 3–20 Hz, the rock pushes the
wavelength to 90–600 m, which is what a 34 m array can bound from below
but not measure. That is the sense in which "rock velocity needs a
hundreds-of-metres array" — it applies to the frequencies the noise
actually supplies, not to Vs30 in general.

And Vs30 itself is decided by the rock here: the model gives
30 / (1.7/220 + 28.3/2000) ≈ 1370 m/s, Vs30 class B. The refraction
P-wave velocity with an assumed Vp/Vs ratio pins that about as well as
any passive measurement could.

**Site A is the borderline case.** The 2026-09-18 refraction line gives
6.2 m at Vp 958 m/s over 3380 m/s, and the Geode gives 265–276 m/s at
39–60 Hz, so take a soil Vs of about 280 m/s. The fundamental mode then
falls from about 1.2 km/s at 12 Hz to 560 m/s at 20 Hz and 290 m/s at
30 Hz (λ 100 → 28 → 10 m), and Vs30 comes out near 850 m/s. Here the
soil's branch does overlap the coherent band and the array's wavelength
range. But at 14–20 Hz both October 1 layouts measure 1.7–2.6 km/s with a
single plane wave explaining all the coherency: the fast arrivals swamp the
fundamental mode, and five nodes cannot separate two waves at once. The
plane-wave fit at 22–25 Hz only hints at 400–800 m/s, flagged aliased.

This paragraph used to end by saying that only a tighter array (1–5 m
separations) could reach the soil at site A. **That is superseded.** The
compact circle already did, at 27–29 Hz. It used the spatial pattern across
its rings past the first zero of J0, not a single plane wave:
about 325 m/s, a soil Vs of about 290 m/s
([above](#the-compact-circle-does-see-the-soil-at-2729-hz)). What that
array lacks is bandwidth. One 1.06× band gives one phase velocity, so the
soil thickness still has to come from refraction. A smaller circle would
widen the band upward, but only if the ambient field stays coherent above
30 Hz. That is worth testing, but it is no longer the only route to the
soil. At site B, with the soil band above 30 Hz, there is still none.

## What would work

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
- **Know what the ground is before choosing passive.** Thin soil over hard
  rock — site B here — puts the soil band above 30 Hz, out of reach of
  ambient noise at any array size; a few metres over rock — site A — leaves
  most of the soil band under a much stronger rock-guided field, with only
  a narrow window just above it (27–29 Hz on a 6–14 m circle). A refraction
  line first, which is cheap and was done here anyway, says whether passive
  can work at all.
- **Tape the layout.** GPS-only positions cost 20–50 % on short separations.
- **For active shear wave, use the Geode, not the nodes** — 24 channels beats
  5, and the only change needed is a longer record. A 1 s record on a 24-channel
  spread at 4 m spacing (92 m long) resolves wavelengths of 8–90 m, so depths of
  3–30 m, from one deployment. The nodes then become worthwhile only as extra
  long-offset channels beyond the end of the cable.

## Status in the repo

Both zips stay in the git-ignored `data/`. It has **not** been promoted to
`examples/data/`: the bar there is a coherent survey with every known problem
written down, and while the problems are now written down, the September line
cannot deliver the result the example would exist to demonstrate. The
October 2-D data were the deployment that was to be revisited for; they are
clean and well-designed, and they show that the limitation is the site
rather than the array. They are a good teaching dataset for *why* passive
fails over shallow rock — the plane-wave fit makes the point directly — and,
on the compact circle, for one soil phase velocity past the first zero of
J0; but not for a passive Vs profile.

Tools added for this analysis: `drivers.atom_atm.split_layouts`,
`passive.plane_wave_fit` (also takes `events=` for transients on a 2-D
array), the log-log velocity slope in `passive.separation_test`, and its
multi-lobe criterion past the first zero of J0 (`SeparationTest.multi_lobe`).
`notebooks/passive_shear_wave_colab.ipynb` now works per layout rather than
per field day, adds the plane-wave fit as step 5b, and on a 2-D array takes
its dispersion points from that fit — showing them, but inverting them only
if `INVERT_APPARENT` is set, since over rock they are the rock's apparent
velocity rather than the soil's.
