# SuperSting dipole-dipole line 111126, 2026-11-11

One short AGI SuperSting resistivity line: 21 electrodes at 2 ft, dipole-dipole.
It is the clean ERT example: once the readings past the end of the cable are
removed, the data have no negative values and low stacking errors, and they fit
to within their errors (chi² ≈ 1). It is the counterpart of
[`2610234-ert-line/`](../2610234-ert-line/), which fails the same checks.

> **Fields marked _not recorded_ below are not in the files and are not
> known to the software.** If you ran this line, fill them in.

## The survey

| Field | Value |
|---|---|
| Site | *Not recorded* |
| Date, crew | 2026-11-11, 12:06-12:36 local instrument time. Crew *not recorded* |
| Purpose | 2-D electrical resistivity tomography |
| Instrument, firmware | AGI SuperSting R8-IP, firmware 01.33.78E (from the `.stg` header) |
| Export format | `111126.stg`: the SuperSting's raw measurement file. `111126.dat`: RES2DINV standard-array file converted from it (array type 3, mid-point x-location, no IP, no topography) |
| Length unit | Metres in both files. The spacing, 0.6096 m, is **2 ft** |
| Electrodes | **21** in the ground, electrode 1 at x = 0, electrode 21 at x = 12.19 m. *Inferred from the data, not from field notes*; see Known problems |
| Command file | `111126`: 633 readings addressing electrodes 1-87. 184 of them use only electrodes 1-21 |
| Array | Dipole-dipole, laid out B A M N (current dipole left of the potential dipole). On the cable: dipole length 1 with n = 1-16 (168 readings) and dipole length 2 with n = 5-8 (16). Every position those 20 configurations allow on 21 electrodes is measured |
| Transmitter | 2 cycles, 0.8 s measurement time per reading; voltage set per reading by the instrument (10-400 V). Current on the cable 5-673 mA, median 227 mA |
| Stacking error | 0-2.7 % on the cable (median 0.2 %); 0 on 19 % of readings, so do not use it alone as the data error |
| Line orientation and topography | *Not recorded*; treated as flat |
| Electrode type, ground conditions, weather | *Not recorded* |

## Files

`raw/` holds both files exactly as received. Use the `.stg`: it has every
reading, including the negative ones, and the current and stacking error of
each.

```python
from shallowgeo.datasets import example_data
from shallowgeo.ert import read_stg, read_res2dinv

raw = example_data("2026-11-11-ert-line")
data = read_stg(raw / "111126.stg", n_electrodes=21)      # 184 readings
# or, from the RES2DINV export (the same 184 on the cable):
data = read_res2dinv(raw / "111126.dat", n_electrodes=21)
```

`shallowgeo ert-clean raw/111126.stg -e 21` writes a RES2DINV-ready copy of
the 184 readings.

## Known problems

- **The command file runs past the cable.** It addresses electrodes up to 87,
  and the SuperSting recorded a value for all 633 readings. The 449 readings
  that use an electrode past 21 are not measurements of the ground:
  apparent resistivities of 10³-10⁴ Ω·m (median 1900-10 000 Ω·m against 30-150
  on the cable) and 201 of them zero or negative. On electrodes 1-21 there is
  not one negative reading, and the first reading to use electrode 22 is
  negative. That jump is the evidence for 21 electrodes; the field notes
  should confirm it. `read_stg(..., n_electrodes=21)` or `ERTData.trim(21)`
  drops them.
- **The `.dat` export is filtered.** It has 431 of the 633 readings: the
  converter dropped all 201 readings with apparent resistivity ≤ 0 and one
  positive reading (record 513), the only one in the file with a stacking
  error above 10 % (16 %). All of them use electrodes past 21, so the `.dat`
  and the `.stg` give the same 184 readings on the cable, to 1e-5.
- **Two readings were injected at 5 mA** (records 171 and 239), the lowest
  current the SuperSting uses, which points to a poor contact at one of
  their current electrodes. Drop readings below 10 mA.
- **A few outliers.** Seven of the remaining 182 readings misfit a first
  inversion by more than 3 robust standard deviations (records 47, 126, 168,
  176, 203, 236 and 261; misfits 16-81 %).
- **Short line, shallow image.** 12.2 m of line gives a maximum
  pseudo-depth of 2.7 m; model cells below about 3 m have less than 1 %
  of the surface coverage.

## Expected result

With electrodes 1-21, readings below 10 mA dropped, a data error of 5 %
combined with the stacking error, and the default `resistivity_tomography`
settings (`lam=20`, `z_weight=0.5`), the first inversion of the 182 readings
reaches chi² 2.5 and an RMS misfit of 8.8 % in 5 iterations. Dropping the
seven outliers and inverting the 175 left converges in 4 iterations to
**chi² 0.96 and an RMS misfit of 4.9 %**:

```python
import numpy as np
from shallowgeo.ert import resistivity_tomography

d = data.select(data.data["current_ma"] >= 10, reason="drop current < 10 mA")
d.data["error"] = np.hypot(0.05, d.data["stack_error"])
first = resistivity_tomography(d)
d = d.select(~d.data["record"].isin(first.outliers()["record"]),
             reason="drop misfit outliers")
result = resistivity_tomography(d)
```

The image shows a resistive body (several hundred to about 1500 Ω·m) in the
top 0.8 m between x ≈ 1.8 and 5 m, over a conductive layer (15-30 Ω·m) from
about 1 m down along the whole line, more conductive with depth under the
right half. Under the rest of the line the top metre is 30-100 Ω·m. What the
resistive body is (a dry fill, gravel, a buried object) needs the site
description.
[`notebooks/ert_supersting.ipynb`](../../../notebooks/ert_supersting.ipynb)
runs on this line: set `DATA` to `raw/111126.dat` and `N_ELECTRODES` to 21.
