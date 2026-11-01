# SuperSting dipole-dipole line, program 2610234

One AGI SuperSting resistivity line, exported for RES2DINV. It is the worked
example for `shallowgeo.ert` and
[`notebooks/ert_supersting.ipynb`](../../../notebooks/ert_supersting.ipynb),
and it is here **because of its problems**: it shows both repairs a
SuperSting export can need, and it is a lesson in checking that the data
are physically possible before believing an inversion of them.

> **Fields marked _not recorded_ below are not in the file and are not
> known to the software.** If you ran this line, fill them in.

## The survey

| Field | Value |
|---|---|
| Site | *Not recorded* |
| Date, crew | *Not recorded.* The title line reads `Sting/Swift prg: 2610234`, the SuperSting command-file name. |
| Purpose | 2-D electrical resistivity tomography |
| Instrument, firmware | AGI SuperSting; firmware *not recorded* |
| Export format | RES2DINV `.dat`, standard-array form: array type 3 (dipole-dipole), x-location at the array mid-point (line 5 = `1`), no IP (line 6 = `0`), no topography (trailing zeros) |
| Length unit | Metres in the file. The spacing, 0.9144 m, is **3 ft**. |
| Electrodes | **42** in the ground at 3 ft (0.9144 m), electrode 1 at x = 0, electrode 42 at x = 37.49 m |
| Measurements | 323 in the file; **214** use only electrodes 1-42, out of 466 that the same configurations allow on those electrodes |
| Array | Dipole-dipole, laid out C2 C1 P1 P2. Dipole lengths 1-4 spacings among the 214; n from 1 to 9 |
| Electrode contact, current, stacking | *Not recorded* in a `.dat` export (they are in the SuperSting `.stg` file) |
| Line orientation and topography | *Not recorded*; treated as flat |

## Files

`raw/2610234.dat` is the file exactly as exported. Read it with

```python
from shallowgeo.ert import read_res2dinv
data = read_res2dinv("raw/2610234.dat", n_electrodes=42)
```

or write a RES2DINV-ready copy with
`shallowgeo ert-clean raw/2610234.dat -e 42`.

## Known problems

- **Non-integer n.** `n` is written as `8.00039`, `2.99999` and the like,
  and the dipole length as `6.40079`, because the instrument computes them
  from single-precision electrode positions. RES2DINV refuses the file.
  `read_res2dinv` snaps every electrode to the 0.9144 m grid (the largest
  move is 0.8 mm) and writes integer `n`.
- **109 of the 323 measurements use electrodes that do not exist.** The
  command file addressed electrodes up to position 136 (x = 124 m) but the
  cable had 42, and the SuperSting recorded a value for every one of them.
  `ERTData.trim(42)` drops them.
- **Most of the readings the scheme makes on the cable are not in the file.**
  The 214 kept readings use 19 (dipole length, n) configurations, which fit
  at 466 positions on 42 electrodes; the file has 214 of them. At dipole
  length 1 it holds current dipoles starting at electrodes 1-10, then only
  12, 15, 20, 31 and 38 (the n = 1 mid-points run 1.37, 2.29, ... 9.60 m,
  then jump to 11.43, 14.17, 18.75, 28.80 and 35.20 m); dipole lengths 2
  and 3 have the same pattern. The isolated current dipoles show as slanting
  rows in the pseudosection with gaps between them. These gaps are in the
  export, not made by trimming (`trim` only removes readings that use an
  electrode past 42). Whether the SuperSting skipped those injections, the
  run was cut short, or the export was filtered, the `.dat` cannot say; the
  `.stg` file and the command file can. `ERTData.coverage()` and
  `missing_readings()` list them, and `ert-clean` reports them.
- **The remaining 214 do not look like a response of the ground either.**
  This is the one to understand before using the line for anything:
  - At dipole length 1, apparent resistivity rises from 91 Ω·m at n = 1
    to 3200 Ω·m at n = 8, a factor of 35, laterally uniform along the
    whole line. A thin conductive layer over a perfect insulator -- the
    most extreme layered ground there is -- raises dipole-dipole apparent
    resistivity by 3-5× over the same n.
  - The transfer resistance V/I hardly changes: 5.3 Ω at n = 1, 1.6 Ω at
    n = 8. Over any ground it falls steeply with n (by about 100× here over
    a uniform half-space; 90 Ω·m would give 0.04 Ω at n = 8).
  - The 109 readings whose electrodes are not in the ground have the
    **same** transfer resistances as the 214 that are (median 2.84 Ω
    against 2.74 Ω, same range). 88 of them had both current electrodes
    on the cable and both potential electrodes on nothing at all.
  - The transfer resistance follows the reading's receiver channel, not its
    geometry: (a, n) = (1, 3), (2, 5), (3, 6) and (4, 7) all read 3.07 Ω
    although their geometric factors differ tenfold, and the same series
    3.07, 2.81, 2.59, 2.37, 2.10 Ω repeats for every dipole length.

  Taken together, the recorded voltages look dominated by something other
  than the ground beneath the potential electrodes: leakage between
  conductors in a wet or damaged cable or connector, a cable/switch-box
  address mismatch, or a wrong cable type in the instrument set-up. None of
  `shallowgeo`, pyGIMLi and ResIPy's R2 can fit the 214 readings with any
  resistivity model. At a 3 % error level they stop at chi² 164-271 and an
  RMS misfit of 36-61 %. The `.stg` file, with the current and repeat error of each
  reading, and a contact-resistance test would say which.

## Expected result

```bash
shallowgeo ert-clean examples/data/2610234-ert-line/raw/2610234.dat -e 42
```

reports 323 readings on 136 electrode positions, keeps 214, reports 252
readings missing on the cable (by dipole length and n), and writes a file
RES2DINV loads. The inversion of those 214 (`lam=20`, 3 % error) converges
in 14 iterations to chi² 172 and an RMS misfit of 38 %, and a smaller `lam`
does not do better. On the same data pyGIMLi reaches chi² 164 and 36 %, and
ResIPy's R2 reaches chi² 271 and 61 % (206 and 36 % with ResIPy's default
error reweighting). The
notebook shows why, then demonstrates the same inversion on synthetic data
recorded with this line's exact measurement scheme, which it fits to the
noise level.
