# USGS towed resistivity line 3000, Des Moines River, 2018-09-06

One **continuous resistivity profiling (CRP)** line from a USGS data release:
an AGI SuperSting towing an 11-electrode streamer behind a boat on the Des
Moines River, Des Moines, Iowa. It is the shortest of the release's three
lines, at about 0.95 km. The other two (2000, about 7.3 km; 1000, about
13 km) are too long for `shallowgeo.ert` to invert in one go (see
*Processing time*).

It shows a survey a land line cannot: the electrodes move, the same eight
readings are repeated every 2.5 s, and the top of the section is the river.

**Source.** Gruhn, L.R., Bristow, E.L., and White, E.A., 2021, Geophysical
Data Collected in the Des Moines River, Beaver Creek, and the Des Moines
River Floodplain, Des Moines, Iowa, 2018: U.S. Geological Survey data
release, <https://doi.org/10.5066/P9B9AVKJ>. CRP child item:
<https://www.sciencebase.gov/catalog/item/616594fdd34e52f49c82319d>.
Interpretation: Thomas, J.C., Spring, M.A., Gruhn, L.R., and Bristow, E.L.,
2023, U.S. Geological Survey Fact Sheet 2023-3006,
<https://doi.org/10.3133/fs20233006>. USGS data are in the public domain
(the release states no use or access constraints); cite the data release
when you use them.

## The survey

| Field | Value |
|---|---|
| Site | Des Moines River, Polk County, Iowa. Start 41.65241 N, 93.67558 W; end 41.65212 N, 93.66733 W (WGS84, from `Readme_CRP.txt`). The boat travelled roughly east |
| Date, crew | 2018-09-06, 14:17-14:30 instrument time (CDT; see Known problems). USGS crew; contact Eric A. White. "Above average flow for September" |
| Purpose | Resistivity of the water column and the river-bed sediments of the alluvial aquifer, to about 15 m below the water surface. Collected with continuous seismic profiling on the same GPS |
| Instrument, firmware | AGI SuperSting R8-IP, firmware 01.03.15E. GPS: Lowrance, $GPRMC at 1 Hz; 200 kHz echo sounder ($SDDPT) |
| Export format | In the USGS archive: `3000.stg`, the SuperSting's raw measurement file, positions **relative to the streamer**; `3000.gps`, NMEA log (UTC); `3000.cmd`, the command file; `3000.crs`, contact resistances. Here: `3000.dat`, RES2DINV standard dipole-dipole made by `read_crp` |
| Length unit | Metres |
| Electrodes | 11 on a floating streamer at **10 m**, electrode 1 nearest the boat. The release's notes file says 6 m; the `.stg`, the abstract and the command-file scale all say 10 m |
| Measurements | One command, repeated: current on electrodes 2-1, potentials on 3-4, 4-5 ... 10-11 (dipole-dipole, a = 10 m, n = 1-8), eight channels at once. **2560 readings = 320 injections**, one every 2.5 s, the boat moving 0.3-6 m (median 2.5 m) between them. None negative |
| Transmitter | 1 cycle, 1.2 s; current 720-1093 mA |
| Stacking error | 0 on every reading (single cycle): not a data error |
| Line orientation and topography | Water surface (flat). Along-track distance is about 1050 m, against 790 m straight from start to end; the river is unrolled to a straight line |
| Water depth | 0.5-3.4 m (median 1.9 m) below the transducer, from `$SDDPT` |

## Files

Only `3000.dat` is kept here: a RES2DINV file (dipole-dipole, a = 10 m,
n = 1-8, 680 readings on 95 electrode positions) made from the USGS files
by `read_crp`. The instrument files themselves stay in the USGS archive.
To rebuild the `.dat`, or to work from the raw readings (their repeat
scatter, water depth and GPS positions), download `LINE3000_CRP.zip`
(1 MB; SHA-256 `f106f66d63294a293468a22caee61968e36912b369e78ab54675389dbf03c9ee`)
from the [ScienceBase item](https://www.sciencebase.gov/catalog/item/616594fdd34e52f49c82319d),
unzip it, and:

```python
from shallowgeo.ert import read_crp

data = read_crp("LINE3000_CRP/3000.stg", "LINE3000_CRP/3000.gps", utc_offset_hours=5)
data.to_res2dinv("3000.dat")    # the file here: 680 readings, 95 electrodes
```

The zip also holds the command file (`3000.cmd`), contact resistances
(`3000.crs`), a 99 MB picture of the USGS inversion (`CRP3000b.wmf`), and
the USGS's own processing in Aarhus Workbench: the culled data at 87
positions 10 m apart (`LINE3000_proc1_inverted_res2d_CULLED_dat.xyz`) and a
1-D laterally constrained inversion with 20 layers and a depth of
investigation at each (`LINE3000_proc1_inverted_res2d_INVERTED_inv.xyz`).
`Readme_CRP.txt` and `CRP.xml` on the same page describe the files.

`read_crp` matches each reading's time to the GPS track, lays the streamer
behind the boat, shifts the whole streamer (without stretching it) so its
electrodes fall on a 10 m grid along the track, swaps the current and
potential pairs (reciprocity) so each reading is in the usual B A M N
order, and averages the readings that land on the same four electrodes.
That turns the 2560 readings into **680 on 95 grid positions** (2-9 repeats
each, median 4), and keeps the spread of each group (`scatter`) and the
water depth.

In the [ERT notebook](../../../notebooks/ert_supersting.ipynb):

```python
DATA, N_ELECTRODES = example_data("2018-09-06-usgs-crp-line") / "3000.dat", 95
```

Every grid position is "on the cable", so section 1's trimming removes
nothing and section 4's test 2 has nothing to compare.

## Known problems

- **The instrument clock is local daylight time, Central, not Eastern.**
  `Readme_CRP.txt` says the `.stg` times are Eastern daylight time. They
  only fall inside the GPS log with a 5-hour offset to UTC (Central
  daylight time, which is Iowa's), and the same holds for lines 1000 and
  2000. Use `utc_offset_hours=5`.
- **Positions are good to about 5 m.** Snapping to the 10 m grid moves a
  reading up to 5 m along the track, and the distance from the GPS antenna
  to electrode 1 (`lead`) is not recorded (0 is assumed). Neither changes
  the shape of the image, only where along the line it sits.
- **A bent line treated as straight.** Along-track distance is about 260 m
  longer than the straight line from start to end. A 2-D inversion assumes
  the ground does not change across the line; on a river, the banks are
  close by.
- **The water is part of the model.** The electrodes float at the surface
  and the river is 0.5-3.4 m deep, about the thickness of the top row of
  model cells (2.5 m). The inversion has to find the water layer itself; a
  careful CRP inversion fixes its thickness from the echo sounder and its
  resistivity from a conductivity meter (not in this release).
- **Repeat scatter flags bad readings.** Most groups agree to 1-3 %, but a
  few (cavitation, the streamer twisting) scatter by tens of percent. The
  `.dat` keeps their average; `read_crp(...).data["scatter"]` on the USGS
  files has them.

## Processing time

`notebooks/ert_supersting.ipynb` run end to end on `3000.dat`
(Apple M1 Pro, 8 cores, NumPy/SciPy only): **about 2.3 minutes**. One field
inversion takes about 8 s; the field λ sweep (5 inversions) 44 s, and each
DOI index (2 inversions) 10-15 s. Expect two to three times that on Colab.

The cost grows much faster than the line. `resistivity_tomography` keeps
the sensitivity matrix and the normal equations dense, so time and memory
grow roughly as the square (or worse) of the line length. One inversion of
the first 1, 2 and 3 km of line 2000 (same machine, λ = 20, 5 %):

| Length | Readings | Model cells | Time | Peak memory |
|---|---|---|---|---|
| 1 km | 728 | 1836 | 7 s | 2.1 GB |
| 2 km | 1528 | 3636 | 47 s | 8.0 GB |
| 3 km | 2328 | 5436 | 268 s | 14.1 GB |

Lines 2000 (7.3 km) and 1000 (13 km) whole would need far more memory than
a laptop or Colab has. To use them, cut them into overlapping 1 km pieces
with `ERTData.trim(n, first=...)` (about 10 s per inversion each, so
roughly 1 minute and 2 minutes per pass of the whole line).

## Expected result

With the notebook's defaults (5 % error, λ = 20, `z_weight=0.5`), the
first inversion of the 680 readings reaches chi² 0.84 and an RMS misfit of
4.6 % in 2 iterations. It flags 17 outliers (median repeat scatter 5 %,
against 1.3 % for the rest, so they are the readings that disagreed with
themselves in the field). Without them, the 663 left fit to **chi² 0.49
and RMS 3.5 %**: 5 % is a generous error for these data.

The image has a conductive top 5-8 m (30-60 Ω·m: the river, 0.5-3.4 m
deep, and the fine sediment of its bed) over a resistive layer, 100-250
Ω·m, from about 8 m to 25 m: the sand and gravel of the alluvial aquifer.
It is least resistive around x ≈ 300-330 m and 470-500 m. Below about
25 m the coverage is under 1 % of its maximum.

The USGS's own 1-D laterally constrained inversion
(`LINE3000_proc1_inverted_res2d_INVERTED_inv.xyz` in the USGS zip,
positions at the boat) shows the same layers and the same two low-resistivity breaks once it
is shifted 50 m back to the array mid-point, with more lateral detail and a
depth of investigation of 26-32 m. Its first layer, about 40 Ω·m, is the
water.
