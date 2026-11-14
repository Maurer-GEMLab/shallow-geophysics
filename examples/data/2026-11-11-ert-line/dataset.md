# SuperSting dipole-dipole line 111126, 2026-11-11

One short AGI SuperSting resistivity line: 21 electrodes at 2 ft, dipole-dipole.
It is the clean ERT example: the files hold only the readings on the 21
electrodes in the ground, none of them negative, with low stacking errors, and
they fit to within their errors (chi² ≈ 1).


## The survey

| Field | Value |
|---|---|
| Site | S&T old golf course |
| Date, crew | 2026-11-11, 12:06-12:36 local instrument time. Crew *not recorded* |
| Purpose | 2-D electrical resistivity tomography |
| Instrument, firmware | AGI SuperSting R8-IP, firmware 01.33.78E (from the `.stg` header) |
| Export format | `111126.stg`: the SuperSting's raw measurement file. `111126.dat`: RES2DINV standard-array file converted from it (array type 3, mid-point x-location, no IP, no topography). Both trimmed to electrodes 1-21; see Files |
| Length unit | Metres in both files. The spacing, 0.6096 m, is **2 ft** |
| Electrodes | **21** in the ground (confirmed from the field notes), electrode 1 at x = 0, electrode 21 at x = 12.19 m |
| Measurements | **184**, in both files: dipole-dipole laid out B A M N (current dipole left of the potential dipole), dipole length 1 with n = 1-16 (168 readings) and dipole length 2 with n = 5-8 (16). Every position those 20 configurations allow on 21 electrodes is measured |
| Command file | `111126`, written for a longer cable: 633 readings addressing electrodes 1-87. Only the 184 on electrodes 1-21 are kept |
| Transmitter | 2 cycles, 0.8 s measurement time per reading; voltage set per reading by the instrument (10-400 V). Current 5-673 mA, median 227 mA |
| Stacking error | 0-2.7 % (median 0.2 %); 0 on 19 % of readings, so do not use it alone as the data error |
| Line orientation and topography | generally flat topography, straight line |
| Electrode type, ground conditions, weather | steel, damp to dry, sunny and ~65 deg F |

## Files

`raw/` holds the two files with every reading that used an electrode past 21
removed (see Known problems). Nothing else was changed: the kept lines are
byte-for-byte as the instrument and converter wrote them, the `.stg` keeps
the instrument's record numbers (so they run 1-532 with gaps), and only the
record count in each header was changed (`Records: 184` in the `.stg`, `184`
on line 4 of the `.dat`). Use the `.stg`: it has the current and stacking
error of each reading.

```python
from shallowgeo.datasets import example_data
from shallowgeo.ert import read_stg, read_res2dinv

raw = example_data("2026-11-11-ert-line")
data = read_stg(raw / "111126.stg")         # 184 readings on 21 electrodes
# or, from the RES2DINV export (the same 184 readings):
data = read_res2dinv(raw / "111126.dat")
```

`shallowgeo ert-clean raw/111126.stg -e 21` writes a RES2DINV-ready copy.

## Known problems

- **The command file ran past the cable (removed from these files).** It
  addressed electrodes up to 87, and the SuperSting recorded a value for all
  633 readings. The 449 that used an electrode past 21 were not measurements
  of the ground: apparent resistivities of 10³-10⁴ Ω·m (median 1900-10 000
  Ω·m against 30-150 on the cable), 201 of them zero or negative. They were
  deleted from both files, leaving 184 in each. The `.dat` as converted had
  431 readings, because the converter had already dropped the 201 with
  apparent resistivity ≤ 0 and one with a 16 % stacking error (record 513).
  If you collect a line like this yourself, `read_stg(..., n_electrodes=21)`
  or `ERTData.trim(21)` does the same trimming in software.
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

With readings below 10 mA dropped, a data error of 5 %
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
