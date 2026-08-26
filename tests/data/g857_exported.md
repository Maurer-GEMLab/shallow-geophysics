# g857_exported.asc

Exported G-857 layout, five columns with a commented header line.

- Instrument: Geometrics G-857
- Source: layout specified by the instrument owner (J. Maurer), 2026-08-26
- Firmware: not recorded

Columns: `Line`, `Station`, `Time(HH:MM:SS)`, `Total_Field(nT)`, `Quality/Signal`.

Station identifiers are strings (`S001`), not numbers — the driver must not
coerce them. `Quality/Signal` is the instrument's signal-strength number; it is
carried through as `quality` and can be used to reject readings via
`min_quality=`, but it is *not* a standard deviation and must not be used
directly as an inversion weight.

Row 5 repeats station `S001` about 35 minutes later, which is the base-station
re-occupation pattern that diurnal correction depends on.
