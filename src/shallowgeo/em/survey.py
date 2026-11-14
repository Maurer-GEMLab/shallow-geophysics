"""The frequency-domain EM survey container.

A ground-conductivity meter does not fit either Layer 0 shape. It is not a
:class:`~shallowgeo.core.survey.PointSurvey` -- there are no stations,
only a sample every fraction of a second along wherever the sensor was
walked, towed or flown -- and each sample is a whole set of channels, an
in-phase and a quadrature reading at every frequency. :class:`FDEMSurvey`
keeps the samples as rows and the channels as columns, with a table saying
which column is which frequency and component, and the
:class:`~shallowgeo.em.deployment.Deployment` that says how high the
sensor was.

Columns of ``readings``:

``line``, ``sample``
    The instrument's line and sample numbers.
``time``
    Absolute UTC, ``datetime64[ns, UTC]``.
``x``, ``y``
    Sensor position in the survey's projected CRS (metres).
``z``
    Elevation of the GPS antenna as the receiver reported it.
``height``
    Optional: sensor height above ground per sample. When absent the
    deployment's single ``sensor_height`` applies to every sample.
channel columns
    ``I_<f>`` and ``Q_<f>`` in ppm of the primary field, e.g. ``Q_63030``.
anything else
    ``powerline``, ``gps_quality``, ``mark``, ``status``... carried along.

Vendor-computed products (the GEM-2's apparent conductivity, susceptibility
and layered inversion) are kept apart, in :attr:`FDEMSurvey.vendor`, so a
number this package computed and a number the instrument software computed
can never be mistaken for one another.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from ..core.geometry import Geometry
from ..core.provenance import Provenance
from ..core.survey import Survey
from .deployment import Deployment, Instrument

COMPONENTS = {"I": "inphase", "Q": "quadrature"}
_CHANNEL_RE = re.compile(r"^([IQ])_(\d+(?:\.\d+)?)$")


def channel_name(component: str, frequency: float) -> str:
    """``("Q", 63030) -> "Q_63030"``. ``component`` is ``I``/``Q`` or the long name."""
    comp = {"inphase": "I", "quadrature": "Q"}.get(component, component)
    if comp not in COMPONENTS:
        raise ValueError(f"component must be I or Q, got {component!r}")
    f = float(frequency)
    return f"{comp}_{int(f) if f.is_integer() else f}"


def channel_table(columns) -> pd.DataFrame:
    """The channel table for every ``I_<f>``/``Q_<f>`` name in *columns*."""
    rows = []
    for name in columns:
        m = _CHANNEL_RE.match(str(name))
        if m:
            rows.append({"channel": name, "component": COMPONENTS[m.group(1)],
                         "frequency": float(m.group(2)), "units": "ppm"})
    if not rows:
        return pd.DataFrame(columns=["component", "frequency", "units"],
                            index=pd.Index([], name="channel"))
    return (pd.DataFrame(rows).sort_values(["frequency", "component"])
            .set_index("channel"))


class FDEMSurvey(Survey):
    """Multi-frequency EM samples along a track.

    Parameters
    ----------
    readings
        One row per sample; see the module docstring for the columns.
        ``time``, ``x`` and ``y`` are required, and at least one channel.
    instrument
        Coil geometry, e.g. :data:`~shallowgeo.em.deployment.GEM2`.
    deployment
        How the sensor was carried. Required: see
        :mod:`shallowgeo.em.deployment` for why there is no default.
    geometry
        One ``station`` point per sample. Built from ``readings`` when
        omitted, which needs ``spatial_ref``.
    vendor
        Instrument-software products, row-aligned with ``readings``.
    """

    method = "fdem"

    def __init__(
        self,
        readings: pd.DataFrame,
        *,
        instrument: Instrument,
        deployment: Deployment,
        geometry: Geometry | None = None,
        spatial_ref=None,
        vendor: pd.DataFrame | None = None,
        metadata: dict[str, Any] | None = None,
        provenance: Provenance | None = None,
    ) -> None:
        if not isinstance(deployment, Deployment):
            raise TypeError("deployment must be a shallowgeo.em.Deployment")
        missing = {"time", "x", "y"} - set(readings.columns)
        if missing:
            raise ValueError(f"readings missing column(s): {sorted(missing)}")
        channels = channel_table(readings.columns)
        if channels.empty:
            raise ValueError("readings have no I_<f> or Q_<f> channel columns")
        if deployment.sensor_height is None and "height" not in readings:
            raise ValueError(
                "the deployment gives no sensor height and the readings have no "
                "'height' column, so no reading can be interpreted"
            )
        readings = readings.reset_index(drop=True).copy()
        readings["time"] = pd.to_datetime(readings["time"], utc=True)
        if geometry is None:
            if spatial_ref is None:
                raise ValueError("give geometry= or spatial_ref=")
            z = readings["z"] if "z" in readings else np.zeros(len(readings))
            geometry = Geometry(np.arange(len(readings)), readings["x"], readings["y"],
                                z, ["station"] * len(readings), spatial_ref)
        super().__init__(geometry, metadata=metadata, provenance=provenance)
        self.readings = readings
        self.channels = channels
        self.instrument = instrument
        self.deployment = deployment
        if vendor is not None:
            vendor = vendor.reset_index(drop=True)
            if len(vendor) != len(readings):
                raise ValueError("vendor table must be row-aligned with readings")
        self.vendor = vendor if vendor is not None else pd.DataFrame(index=readings.index)

    # -- basic access ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.readings)

    @property
    def n_samples(self) -> int:
        return len(self.readings)

    @property
    def frequencies(self) -> np.ndarray:
        return np.unique(self.channels["frequency"].to_numpy(float))

    @property
    def lines(self) -> list:
        if "line" not in self.readings:
            return [None]
        return list(pd.unique(self.readings["line"]))

    @property
    def coil_separation(self) -> float:
        return self.instrument.coil_separation

    @property
    def orientation(self) -> str:
        return self.deployment.orientation

    def channel(self, component: str, frequency: float) -> np.ndarray:
        """One channel, ppm."""
        name = channel_name(component, frequency)
        if name not in self.readings:
            raise KeyError(f"no channel {name}; have {list(self.channels.index)}")
        return self.readings[name].to_numpy(float)

    def inphase(self, frequency: float) -> np.ndarray:
        return self.channel("I", frequency)

    def quadrature(self, frequency: float) -> np.ndarray:
        return self.channel("Q", frequency)

    def response(self) -> np.ndarray:
        """``(n_samples, n_frequencies)`` complex, I + iQ, ppm."""
        return np.column_stack([self.inphase(f) + 1j * self.quadrature(f)
                                for f in self.frequencies])

    def heights(self) -> np.ndarray:
        """Sensor height above ground per sample, metres."""
        if "height" in self.readings:
            h = self.readings["height"].to_numpy(float)
            if self.deployment.sensor_height is not None:
                h = np.where(np.isnan(h), self.deployment.sensor_height, h)
            return h
        return np.full(len(self), float(self.deployment.sensor_height))

    def xy(self) -> np.ndarray:
        return self.readings[["x", "y"]].to_numpy(float)

    # -- copies ---------------------------------------------------------------

    def _copy(self, readings: pd.DataFrame, vendor: pd.DataFrame,
              deployment: Deployment | None = None) -> FDEMSurvey:
        geom_table = self.geometry.table
        if len(readings) != len(self.readings):
            geom = None
        else:
            geom = Geometry.from_dataframe(geom_table.assign(
                x=readings["x"].to_numpy(float), y=readings["y"].to_numpy(float)),
                self.spatial_ref)
        return FDEMSurvey(
            readings, instrument=self.instrument,
            deployment=deployment or self.deployment,
            geometry=geom, spatial_ref=self.spatial_ref, vendor=vendor,
            metadata=dict(self.metadata), provenance=self.provenance.copy(),
        )

    def with_readings(self, readings: pd.DataFrame, operation: str,
                      **params) -> FDEMSurvey:
        """Copy with new readings (same rows) and one appended provenance step."""
        if len(readings) != len(self.readings):
            raise ValueError("with_readings keeps the rows; use select() to drop some")
        out = self._copy(readings, self.vendor)
        out.provenance.record(operation, **params)
        return out

    def with_deployment(self, deployment: Deployment | None = None,
                        **changes) -> FDEMSurvey:
        """Copy with the deployment replaced or edited, recorded in provenance."""
        dep = deployment or self.deployment.with_changes(**changes)
        out = self._copy(self.readings, self.vendor, deployment=dep)
        out.provenance.record("set_deployment", **dep.to_dict())
        return out

    def select(self, mask, *, reason: str = "select") -> FDEMSurvey:
        """The samples where *mask* is true."""
        mask = np.asarray(mask, dtype=bool)
        out = self._copy(self.readings[mask].reset_index(drop=True),
                         self.vendor[mask].reset_index(drop=True))
        out.provenance.record(reason, kept=int(mask.sum()), dropped=int((~mask).sum()))
        return out

    def line(self, line) -> FDEMSurvey:
        """The samples of one line."""
        if "line" not in self.readings:
            raise KeyError("readings have no 'line' column")
        mask = (self.readings["line"] == line).to_numpy()
        if not mask.any():
            raise KeyError(f"no line {line!r}; have {self.lines}")
        return self.select(mask, reason=f"line {line}")

    def summary(self) -> pd.DataFrame:
        """One row per line: samples, duration, length walked, speed."""
        r = self.readings
        groups = r.groupby("line", sort=False) if "line" in r else [(None, r)]
        rows = []
        for name, g in groups:
            step = np.hypot(np.diff(g["x"]), np.diff(g["y"]))
            dur = (g["time"].iloc[-1] - g["time"].iloc[0]).total_seconds()
            rows.append({"line": name, "samples": len(g), "duration_s": dur,
                         "length_m": step.sum(),
                         "speed_m_s": step.sum() / dur if dur > 0 else np.nan,
                         "spacing_m": np.median(step) if len(step) else np.nan})
        return pd.DataFrame(rows).set_index("line")

    def __repr__(self) -> str:
        f = ", ".join(f"{v:g}" for v in self.frequencies)
        return (f"<FDEMSurvey {self.instrument.name} {self.deployment.platform} "
                f"{self.orientation} samples={len(self)} lines={len(self.lines)} "
                f"f=[{f}] Hz>")
