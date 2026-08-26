"""Method-agnostic survey containers -- the Layer 0 contract.

Everything upstream (drivers) produces one of these; everything downstream
(corrections, meshing, inversion wrappers) consumes one. Two concrete shapes
cover the initial four methods:

``SeismicSurvey``
    Many time series recorded against a source/receiver geometry.
    Refraction (Geode) and MASW, active or passive (ATOM-1C).

``PointSurvey``
    Scalar readings at stations, repeated through time.
    Gravity (CG-5) and magnetics (G-857). The time axis matters: it is what
    drift and diurnal corrections are computed against, so it is a required
    column rather than optional metadata.

The core stores waveforms as a plain ``(n_traces, n_samples)`` array rather
than an ObsPy ``Stream`` so that ``shallowgeo.core`` imports without ObsPy.
:meth:`SeismicSurvey.to_obspy` bridges when you want ObsPy's processing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .geometry import Geometry
from .provenance import Provenance


class Survey:
    """Base container: geometry + provenance + free-form metadata."""

    method: str = "generic"

    def __init__(
        self,
        geometry: Geometry,
        *,
        metadata: dict[str, Any] | None = None,
        provenance: Provenance | None = None,
    ) -> None:
        self.geometry = geometry
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.provenance = provenance or Provenance()

    @property
    def spatial_ref(self):
        return self.geometry.spatial_ref

    def __repr__(self) -> str:
        return f"<{type(self).__name__} method={self.method!r} {self.geometry!r}>"


class SeismicSurvey(Survey):
    """Multichannel time-series data with shot/receiver association.

    Parameters
    ----------
    data
        ``(n_traces, n_samples)``, one row per recorded channel.
    sample_interval
        Seconds between samples.
    trace_map
        One row per trace, aligned with ``data``. Must carry ``receiver_id``;
        ``source_id`` is required for active-source data and absent for
        passive records. Extra columns (``channel``, ``record``, ``delay``)
        pass through untouched.
    """

    method = "seismic"

    def __init__(
        self,
        data: np.ndarray,
        sample_interval: float,
        geometry: Geometry,
        trace_map: pd.DataFrame,
        *,
        start_time=None,
        metadata: dict[str, Any] | None = None,
        provenance: Provenance | None = None,
    ) -> None:
        super().__init__(geometry, metadata=metadata, provenance=provenance)
        data = np.atleast_2d(np.asarray(data))
        if len(trace_map) != data.shape[0]:
            raise ValueError(
                f"trace_map has {len(trace_map)} rows but data has "
                f"{data.shape[0]} traces"
            )
        if "receiver_id" not in trace_map.columns:
            raise ValueError("trace_map requires a 'receiver_id' column")
        if sample_interval <= 0:
            raise ValueError(f"sample_interval must be positive, got {sample_interval}")
        self.data = data
        self.sample_interval = float(sample_interval)
        self.trace_map = trace_map.reset_index(drop=True)
        self.start_time = start_time

    @property
    def n_traces(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def sample_rate(self) -> float:
        return 1.0 / self.sample_interval

    @property
    def duration(self) -> float:
        return self.n_samples * self.sample_interval

    @property
    def is_passive(self) -> bool:
        """No associated source -- ambient noise, as from the ATOM-1C."""
        return "source_id" not in self.trace_map.columns

    def times(self) -> np.ndarray:
        return np.arange(self.n_samples) * self.sample_interval

    def offsets(self) -> np.ndarray:
        """Source-receiver offset per trace, in the geometry's units.

        The single most-used derived quantity in refraction and MASW, and the
        thing most often wrong when SEG-2 geometry has not been checked.
        """
        if self.is_passive:
            raise ValueError("passive survey has no sources, so no offsets")
        pos = self.geometry.table.set_index(["role", "id"])[["x", "y", "z"]]
        src = pos.loc["source"].reindex(self.trace_map["source_id"]).to_numpy(float)
        rec = pos.loc["receiver"].reindex(self.trace_map["receiver_id"]).to_numpy(float)
        return np.linalg.norm(rec - src, axis=1)

    def gather(self, source_id) -> SeismicSurvey:
        """The subset of traces recorded from one shot."""
        if self.is_passive:
            raise ValueError("passive survey has no shot gathers")
        mask = (self.trace_map["source_id"] == source_id).to_numpy()
        if not mask.any():
            raise KeyError(f"no traces for source_id={source_id!r}")
        out = SeismicSurvey(
            self.data[mask],
            self.sample_interval,
            self.geometry,
            self.trace_map.loc[mask],
            start_time=self.start_time,
            metadata=dict(self.metadata),
            provenance=self.provenance.copy(),
        )
        out.provenance.record("gather", source_id=source_id)
        return out

    def to_obspy(self):
        """Convert to an ObsPy ``Stream`` for filtering, picking, plotting.

        Requires the ``seismic`` extra. Trace-map columns are copied into
        ``trace.stats.shallowgeo`` so the association survives the round trip
        through ObsPy processing.
        """
        try:
            from obspy import Stream, Trace, UTCDateTime
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                "to_obspy() needs ObsPy: pip install 'shallow-geophysics[seismic]'"
            ) from exc

        traces = []
        for i, row in self.trace_map.iterrows():
            header = {"delta": self.sample_interval, "channel": str(row.get("channel", ""))}
            if self.start_time is not None:
                header["starttime"] = UTCDateTime(self.start_time)
            tr = Trace(data=np.ascontiguousarray(self.data[i]), header=header)
            tr.stats.shallowgeo = row.to_dict()
            traces.append(tr)
        return Stream(traces)


class PointSurvey(Survey):
    """Scalar readings at stations through time.

    Parameters
    ----------
    readings
        Required columns ``station_id``, ``time`` (datetime-like), ``value``.
        Optional ``sigma`` (per-reading standard deviation, which the CG-5
        reports and which should be carried into inversion weighting rather
        than discarded).
    quantity
        What ``value`` measures, e.g. ``"gravity"`` or ``"total_field"``.
    units
        Units of ``value``, e.g. ``"mGal"`` or ``"nT"``. Stored explicitly
        because vendor exports are inconsistent and unit errors here are
        silent.
    """

    method = "point"

    def __init__(
        self,
        readings: pd.DataFrame,
        geometry: Geometry,
        *,
        quantity: str,
        units: str,
        metadata: dict[str, Any] | None = None,
        provenance: Provenance | None = None,
    ) -> None:
        super().__init__(geometry, metadata=metadata, provenance=provenance)
        missing = {"station_id", "time", "value"} - set(readings.columns)
        if missing:
            raise ValueError(f"readings missing column(s): {sorted(missing)}")
        readings = readings.copy()
        readings["time"] = pd.to_datetime(readings["time"])
        self.readings = readings.sort_values("time").reset_index(drop=True)
        self.quantity = quantity
        self.units = units

    @property
    def n_readings(self) -> int:
        return len(self.readings)

    def station_means(self) -> pd.DataFrame:
        """Collapse repeat occupations to one value per station.

        Standard deviation across occupations is reported as ``repeat_sigma``,
        which is the honest field estimate of measurement error and a better
        inversion weight than the instrument's internal sigma.
        """
        g = self.readings.groupby("station_id")["value"]
        return pd.DataFrame(
            {
                "value": g.mean(),
                "repeat_sigma": g.std(ddof=1),
                "n": g.size(),
            }
        ).reset_index()

    def with_values(self, values: np.ndarray, operation: str, **params) -> PointSurvey:
        """Copy with replaced values and one appended provenance step.

        Corrections use this so that no correction mutates its input and the
        lineage always matches the numbers.
        """
        readings = self.readings.copy()
        readings["value"] = np.asarray(values, dtype=float)
        out = PointSurvey(
            readings,
            self.geometry,
            quantity=self.quantity,
            units=self.units,
            metadata=dict(self.metadata),
            provenance=self.provenance.copy(),
        )
        out.provenance.record(operation, **params)
        return out
