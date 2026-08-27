# gnss_emlid_reach.csv

Emlid Reach RS2/RS3 point export, RTK against a local base.

- **Synthetic.** Structure and column set copied from a real Emlid export; the
  site, coordinates, and times are invented. No real survey data here.
- Base at exactly -92.0, 38.0, 320.000 m ellipsoidal; antenna height 2.134 m.

Written to reproduce six pathologies observed in a real export, each of which
a naive reader gets wrong:

| # | Pathology | Rows | Consequence if mishandled |
|---|---|---|---|
| 1 | `Easting`/`Northing`/`Elevation` present but entirely empty | all | reading `Elevation` yields NaN |
| 2 | UTC offset differs between `Averaging start` and `Averaging end` | 7 | 35 s occupation parses as 3635 s |
| 3 | A row **named** `999 - Base` that is a rover ~15 m from the base | 1 | base shift wrong by 15 m / 1 m |
| 4 | Station name `S008` reused at a mark ~40 m away | 2 | keyed joins silently pick one |
| 5 | FLOAT solution with the same vertical RMS as the FIX rows | 1 | gating on RMS admits it |
| 6 | Two names (`S008`, `S015`) at one physical mark, ~1 m apart | 2 | free repeatability check missed |

Row `S006` also has 81 samples where every other occupation has 176 — a short
occupation worth flagging but not rejecting.

The authoritative base coordinate is in the `Base longitude` / `Base latitude` /
`Base ellipsoidal height` columns, identical on every row — **not** in the row
named `999 - Base`.
