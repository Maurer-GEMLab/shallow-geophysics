"""A made-up magnetics survey, for demonstrations and tests.

Two buried magnetic objects under a grid of lines, read the way a class
reads one: every line walked from its south end, two readings per station,
and a return to the base station before every line and at the end. The
Earth's field drifts through the session, so the table needs a diurnal
correction, and the base-station rows are there to make one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Geomagnetic inclination in central Missouri, degrees below horizontal.
INCLINATION = 64.0


def _dipole_anomaly(x, y, sources, inclination):
    """Total-field anomaly of induced point dipoles, nT, peak scaled per source.

    ``x`` east and ``y`` north along the ground; each source is ``(x, y,
    depth, peak_nT)``. Declination is taken as zero.
    """
    inc = np.radians(inclination)
    f = np.array([0.0, np.cos(inc), -np.sin(inc)])  # unit field: north and down
    total = np.zeros_like(x, dtype=float)
    for sx, sy, depth, peak in sources:
        r = np.stack([x - sx, y - sy, np.full_like(x, depth, dtype=float)], axis=-1)
        dist = np.linalg.norm(r, axis=-1)
        rhat = r / dist[..., None]
        shape = (3 * (rhat @ f) ** 2 - 1) / dist**3
        # Scale so the largest value straight above a source is `peak` nT.
        directly_above = 3 * np.sin(inc) ** 2 - 1
        total += peak * shape * depth**3 / directly_above
    return total


def synthetic_table(
    *,
    n_lines: int = 9,
    n_stations: int = 17,
    station_spacing: float = 10.0,
    line_spacing: float = 10.0,
    regional: float = 50_600.0,
    drift_amplitude: float = 35.0,
    noise: float = 1.5,
    start: str = "2026-10-12 13:00:00",
    seconds_per_station: float = 40.0,
    seed: int = 0,
) -> pd.DataFrame:
    """A station table in the layout the ``mag-summary`` driver reads.

    Distances are in feet. The base station is line 1 station 1, read at
    the start, before every later line, and at the end, so it appears
    ``n_lines + 1`` times. Columns: ``Line_No``, ``Station``,
    ``Dist_from_S_ft``, ``Time``, ``N_Readings``, ``Mean_nT``,
    ``StdDev_nT``. Read it back with ``base_station=(1, 1)``.
    """
    rng = np.random.default_rng(seed)
    width = (n_lines - 1) * line_spacing
    length = (n_stations - 1) * station_spacing
    sources = [
        (0.35 * width, 0.6 * length, 5.0, 1800.0),   # shallow: a buried drum
        (0.75 * width, 0.25 * length, 9.0, 450.0),   # deeper and weaker
    ]

    rows = []
    t = 0.0

    def read(line, station, dist, t):
        x = (line - 1) * line_spacing
        anomaly = _dipole_anomaly(np.array([x]), np.array([dist]), sources, INCLINATION)[0]
        drift = drift_amplitude * np.sin(2 * np.pi * t / 7200.0) + 0.004 * t
        true = regional + 0.15 * dist + anomaly + drift
        readings = true + rng.normal(0, noise, 2)
        rows.append((line, station, dist, t, 2, readings.mean(), readings.std(ddof=1)))

    for line in range(1, n_lines + 1):
        if line > 1:
            t += 180.0  # walk back to the base
            read(1, 1, 0.0, t)
            t += 120.0  # and out to the start of the next line
        for station in range(1, n_stations + 1):
            read(line, station, (station - 1) * station_spacing, t)
            t += seconds_per_station
    t += 180.0
    read(1, 1, 0.0, t)

    table = pd.DataFrame(rows, columns=["Line_No", "Station", "Dist_from_S_ft", "Time",
                                        "N_Readings", "Mean_nT", "StdDev_nT"])
    table["Time"] = (pd.Timestamp(start) + pd.to_timedelta(table["Time"], unit="s")
                     ).dt.strftime("%H:%M:%S")
    # Two stations beside a steel fence post: the readings disagree.
    for i in rng.choice(np.flatnonzero(table["Line_No"] > 1), 2, replace=False):
        table.loc[i, "StdDev_nT"] = rng.uniform(150, 400)
    table[["Mean_nT", "StdDev_nT"]] = table[["Mean_nT", "StdDev_nT"]].round(1)
    return table
