"""Geophex GEM-2 multi-frequency EM: the exported ``.csv``.

The GEM-2 logs to a binary ``.gbf`` file; the Geophex exporter (on the PDA
or the PC utility) turns it into a CSV with one row per sample. The export
this reader was written against (Exporter 1.45, firmware
``ARMEN.230101.31``) starts with one line of survey facts::

    #File:038_7007.GPX,Exporter:1.45,Time:18:56-18:59,526 Samples, 86 GPS,
     sn:G2-000-7007,fw:ARMEN.230101.31,Voltage:10.4V 12.2V,Temperature:31.1C 36.0C

then a header row and the samples:

``Line, Sample``
    Line number and sample number within the line.
``Latitude, Longitude``
    Degrees, interpolated by the instrument between 1 Hz GPS fixes.
    (Exports made with the UTM option carry projected coordinates instead,
    which this reader takes when given ``spatial_ref=``.)
``Mark, Status, GPSStat, GPSalt(m)``
    Operator mark, instrument status, NMEA GGA fix quality (2 = DGPS) and
    the GGA altitude, which is above mean sea level (the receiver's geoid).
``Time(ms), Time(hhmmss.sss)``
    Milliseconds since midnight and clock time, both **UTC** (the GEM-2
    sets its clock from the GPS). No date anywhere in the file.
``PowerLn(nT)``
    Power-line (50/60 Hz) field amplitude: the noise monitor.
``I_<f>Hz(ppm), Q_<f>Hz(ppm)``
    In-phase and quadrature at each frequency, parts per million of the
    primary field. These are the measurements.
``QSum, Econd_*, ECondTotal, MSusc_*, MSuscTotal, R_Layer*, T_Layer*, FitError``
    Products of Geophex's software: apparent conductivity (mS/m) and
    susceptibility (10^-3 SI) per frequency, and a three-layer inversion.
    Kept, but apart, in :attr:`FDEMSurvey.vendor`.

The companion ``*_gps.csv`` holds the raw ``$GPGGA`` sentences with the
instrument's time stamp; :func:`read_gem2_gps` reads it.

The platform must be stated: see :mod:`shallowgeo.em.deployment`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.crs import SpatialRef, local_grid
from ..em.deployment import GEM2, Deployment, Instrument
from ..em.survey import FDEMSurvey
from .base import Driver, _sniff_text

_CHANNEL_RE = re.compile(r"^([IQ])_(\d+(?:\.\d+)?)\s*Hz\s*\(ppm\)$", re.IGNORECASE)
_VENDOR_RE = re.compile(
    r"^(QSum|Econd|ECondTotal|MSusc|MSuscTotal|R_Layer|T_Layer|FitError)", re.IGNORECASE)

#: Exported columns renamed to the survey's names.
_RENAME = {
    "line": "line", "sample": "sample",
    "latitude": "lat", "longitude": "lon",
    "mark": "mark", "status": "status", "gpsstat": "gps_quality",
    "gpsalt": "z", "time_ms": "time_ms", "time_hhmmss": "clock",
    "powerln": "powerline",
    "easting": "x", "northing": "y", "x": "x", "y": "y",
}


def _key(label: str) -> str:
    """``"GPSalt(m)" -> "gpsalt"``, ``"Time(ms)" -> "time_ms"``."""
    text = label.strip()
    if re.match(r"(?i)^time\s*\(ms\)", text):
        return "time_ms"
    if re.match(r"(?i)^time\s*\(hh", text):
        return "time_hhmmss"
    text = re.sub(r"\(.*?\)", "", text).strip().lower()
    return re.sub(r"[^a-z0-9_]", "", text)


def _vendor_name(label: str) -> str:
    """``"Econd_450Hz (mS/m)" -> "econd_450_mS_m"``, keeping the unit."""
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", label.strip())
    stem, unit = (m.group(1), m.group(2)) if m else (label.strip(), "")
    stem = stem.replace("Hz", "").strip()
    unit = {"mS/m": "mS_m", "1/1000": "1e-3SI", "Ohm.m": "ohm_m", "m": "m",
            "%": "pct", "ppm": "ppm"}.get(unit, re.sub(r"\W", "", unit))
    return f"{stem.lower()}_{unit}" if unit else stem.lower()


def parse_header_line(line: str) -> dict:
    """The ``#File:...`` survey-facts line as a dict.

    Unknown fields are kept under their own names, so a newer exporter's
    additions survive.
    """
    text = line.lstrip("#").strip()
    out: dict = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d+)\s+(Samples|GPS)$", part, re.IGNORECASE)
        if m:
            out[f"n_{m.group(2).lower()}"] = int(m.group(1))
            continue
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip().lower()] = v.strip()
        else:
            out.setdefault("extra", []).append(part)
    return out


def _is_gem2(path: Path) -> bool:
    lines = _sniff_text(path, 4)
    text = "".join(lines)
    if "\x00" in text:
        return False
    for line in lines:
        cells = [c.strip() for c in line.split(",")]
        if sum(bool(_CHANNEL_RE.match(c)) for c in cells) >= 2:
            return True
    return False


def _find_header(lines: list[str]) -> int:
    for i, line in enumerate(lines[:20]):
        cells = [c.strip() for c in line.split(",")]
        if any(_CHANNEL_RE.match(c) for c in cells):
            return i
    raise ValueError("no header row with I_<f>Hz(ppm)/Q_<f>Hz(ppm) columns")


def _utm_for(lon: float, lat: float) -> SpatialRef:
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return SpatialRef(crs=epsg, vertical_datum="orthometric")


def read_gem2(
    path: str | Path,
    *,
    platform: str | None = None,
    deployment: Deployment | None = None,
    sensor_height: float | None = None,
    orientation: str = "HCP",
    boom_direction: str = "inline",
    gps_offset: tuple[float, float, float] | None = None,
    tow_distance: float | None = None,
    sling_length: float | None = None,
    carrier: str | None = None,
    height=None,
    date=None,
    spatial_ref: SpatialRef | None = None,
    instrument: Instrument = GEM2,
) -> FDEMSurvey:
    """Read a GEM-2 CSV export.

    Parameters
    ----------
    platform
        ``"handheld"``, ``"sled"`` or ``"drone"``. Required (or pass a whole
        ``deployment=``): the reading is meaningless without the height,
        and the file does not say how the sensor was carried.
    sensor_height, orientation, boom_direction, gps_offset, tow_distance,
    sling_length, carrier
        Passed to :meth:`Deployment.for_platform`; anything not given is
        filled with the platform's nominal value and listed as assumed.
    height
        Per-sample sensor height above ground, metres: a column name in the
        file (a drone altimeter channel), or an array. Stored as ``height``.
    date
        Survey date (UTC). The export has times but no date; without one,
        times are placed on 1970-01-01 and ``metadata["date_known"]`` is
        False -- relative timing, which is all drift and QC need, is right.
    spatial_ref
        Target CRS for ``x``/``y``. Default: the UTM zone of the first fix,
        with GGA altitudes declared orthometric. Required for a UTM or
        dead-reckoning export, which has no latitude/longitude.
    """
    path = Path(path)
    if deployment is None:
        if platform is None:
            raise ValueError(
                f"{path.name}: say how the GEM-2 was carried -- "
                "read_gem2(..., platform='handheld' | 'sled' | 'drone'). "
                "The sensor height sets the size of every reading (a ski at "
                "1 m reads 64 % of what it reads at 0.2 m over the same ground), "
                "and the file does not record it."
            )
        deployment = Deployment.for_platform(
            platform, sensor_height=sensor_height, orientation=orientation,
            boom_direction=boom_direction, gps_offset=gps_offset,
            tow_distance=tow_distance, sling_length=sling_length,
            carrier=carrier, heights_per_sample=height is not None)

    lines = path.read_text(encoding="latin-1", errors="replace").splitlines()
    hdr = _find_header(lines)
    facts = {}
    for line in lines[:hdr]:
        if line.lstrip().startswith("#"):
            facts.update(parse_header_line(line))
    raw = pd.read_csv(path, skiprows=hdr, skipinitialspace=True,
                      encoding="latin-1")
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw.dropna(how="all")

    readings = pd.DataFrame(index=raw.index)
    vendor = pd.DataFrame(index=raw.index)
    other = []
    for col in raw.columns:
        m = _CHANNEL_RE.match(col)
        if m:
            f = float(m.group(2))
            readings[f"{m.group(1).upper()}_{int(f) if f.is_integer() else f}"] = \
                pd.to_numeric(raw[col], errors="coerce")
        elif _VENDOR_RE.match(col):
            vendor[_vendor_name(col)] = pd.to_numeric(raw[col], errors="coerce")
        elif _key(col) in _RENAME:
            name = _RENAME[_key(col)]
            readings[name] = raw[col] if name == "clock" else pd.to_numeric(
                raw[col], errors="coerce")
        else:
            readings[col] = raw[col]
            other.append(col)

    # -- time -------------------------------------------------------------------
    day = pd.Timestamp(date, tz="UTC").normalize() if date is not None else \
        pd.Timestamp("1970-01-01", tz="UTC")
    if "time_ms" in readings:
        ms = readings["time_ms"].to_numpy(float)
        # Roll past midnight UTC if the clock wraps during the survey.
        wraps = np.concatenate([[0], np.cumsum(np.diff(ms) < -43_200_000)])
        readings["time"] = day + pd.to_timedelta(ms + wraps * 86_400_000, unit="ms")
    elif "clock" in readings:
        hms = readings["clock"].astype(float)
        secs = (hms // 10000) * 3600 + (hms // 100 % 100) * 60 + hms % 100
        readings["time"] = day + pd.to_timedelta(secs, unit="s")
    else:
        raise ValueError(f"{path.name}: no Time(ms) or Time(hhmmss) column")

    # -- position ---------------------------------------------------------------
    if {"lat", "lon"} <= set(readings):
        lat, lon = readings["lat"].to_numpy(float), readings["lon"].to_numpy(float)
        ok = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) + np.abs(lon) > 0)
        if not ok.any():
            raise ValueError(f"{path.name}: no valid latitude/longitude")
        if spatial_ref is None:
            spatial_ref = _utm_for(lon[ok][0], lat[ok][0])
        geo = SpatialRef(crs=4326, vertical_datum=spatial_ref.vertical_datum)
        x, y = geo.transformer_to(spatial_ref).transform(lon, lat)
        readings["x"] = np.where(ok, x, np.nan)
        readings["y"] = np.where(ok, y, np.nan)
    elif {"x", "y"} <= set(readings):
        if spatial_ref is None:
            spatial_ref = local_grid(0.0, 0.0)
    else:
        raise ValueError(f"{path.name}: no Latitude/Longitude or X/Y columns")

    if height is not None:
        if isinstance(height, str):
            if height not in raw:
                raise KeyError(f"height column {height!r} not in {list(raw.columns)}")
            readings["height"] = pd.to_numeric(raw[height], errors="coerce")
        else:
            h = np.asarray(height, float)
            if h.shape != (len(readings),):
                raise ValueError(f"height needs {len(readings)} values, got {h.shape}")
            readings["height"] = h

    first = ["line", "sample", "time", "x", "y", "z", "height"]
    chans = sorted([c for c in readings if re.match(r"^[IQ]_\d", c)],
                   key=lambda c: (float(c.split("_")[1]), c[0]))
    order = [c for c in first if c in readings] + chans
    order += [c for c in readings if c not in order]
    readings = readings[order]

    n_declared = facts.get("n_samples")
    metadata = {
        "instrument": f"{instrument.vendor} {instrument.name}".strip(),
        "serial": facts.get("sn"),
        "firmware": facts.get("fw"),
        "exporter": facts.get("exporter"),
        "source_file": str(path),
        "logged_file": facts.get("file"),
        "header": facts,
        "date_known": date is not None,
        "time_zone": "UTC (GPS)",
        "samples_in_header": n_declared,
        "unrecognised_columns": other,
    }
    survey = FDEMSurvey(readings, instrument=instrument, deployment=deployment,
                        spatial_ref=spatial_ref, vendor=vendor, metadata=metadata)
    survey.provenance.record(
        "read", driver="gem2", path=str(path), platform=deployment.platform,
        deployment=deployment.to_dict(), date=None if date is None else str(date))
    return survey


def read_gem2_gps(path: str | Path, *, date=None) -> pd.DataFrame:
    """The GEM-2's raw GPS log: one row per ``$GPGGA`` sentence.

    Columns: ``time_ms`` (instrument clock), ``time`` (UTC from the sentence),
    ``lat``, ``lon``, ``quality`` (0 none, 1 GPS, 2 DGPS, 4 RTK fix, 5 RTK
    float), ``n_sats``, ``hdop``, ``altitude`` (above the geoid, m) and
    ``geoid_separation`` (m). Use it to check the positions the exporter
    interpolated, and to see the fix quality the survey really had.
    """
    rows = []
    day = pd.Timestamp(date, tz="UTC").normalize() if date is not None else \
        pd.Timestamp("1970-01-01", tz="UTC")
    for line in Path(path).read_text(encoding="latin-1").splitlines():
        if "$GPGGA" not in line and "$GNGGA" not in line:
            continue
        head, _, nmea = line.partition("$")
        stamp = [t.strip() for t in head.split(",") if t.strip()]
        f = ("$" + nmea).split("*")[0].split(",")
        if len(f) < 12 or not f[2]:
            continue
        lat = int(f[2][:2]) + float(f[2][2:]) / 60
        lat = -lat if f[3] == "S" else lat
        lon = int(f[4][:3]) + float(f[4][3:]) / 60
        lon = -lon if f[5] == "W" else lon
        hh, mm, ss = int(f[1][:2]), int(f[1][2:4]), float(f[1][4:])
        rows.append({
            "time_ms": float(stamp[0]) if stamp else np.nan,
            "time": day + pd.Timedelta(hours=hh, minutes=mm, seconds=ss),
            "lat": lat, "lon": lon, "quality": int(f[6] or 0),
            "n_sats": int(f[7] or 0), "hdop": float(f[8] or "nan"),
            "altitude": float(f[9] or "nan"),
            "geoid_separation": float(f[11] or "nan"),
        })
    if not rows:
        raise ValueError(f"{path}: no $GPGGA sentences")
    return pd.DataFrame(rows)


driver = Driver(
    name="gem2",
    description="Geophex GEM-2 multi-frequency EM, exported CSV",
    can_open=_is_gem2,
    read=read_gem2,
    extensions=(".csv",),
    methods=("fdem",),
    vendor="Geophex",
    instrument="GEM-2",
    notes="needs platform='handheld'|'sled'|'drone'; date= for absolute times",
)
