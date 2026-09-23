"""Shared assembly for straight-line active-source shot records.

Both Geometrics exports in scope -- SEG-2 from the Geode and SEG-Y from the
SeisModule controller -- describe the same thing: one shot into a straight
spread, with receiver and source positions written as *distances along the
line* in whatever unit the operator chose in the field software. Everything
that follows the parse is identical, so it lives here once:

* converting the file's length unit to metres (a Geode in the United States
  is very often run in feet, and a factor of 3.28 in every velocity is the
  kind of error that survives to a final report);
* deciding whether the header geometry is usable or whether the caller must
  supply ``spacing`` / ``source_offset``;
* enforcing a single trigger delay across the record;
* placing the spread in a real CRS and building the ``SeismicSurvey``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..core.crs import SpatialRef, local_grid
from ..core.geometry import Geometry
from ..core.survey import SeismicSurvey

#: Metres per unit, keyed by the spellings vendors actually write.
LENGTH_UNITS: dict[str, float] = {
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    # US survey foot -- what a Geode labels FEET is almost certainly the
    # international foot, but SEG-Y's measurement-system flag does not say,
    # and the 2 ppm difference is far below any field positioning error.
    "us_survey_feet": 1200.0 / 3937.0,
}


def length_unit_factor(name: str | None, *, default: str = "m") -> float:
    """Metres per one of *name*; ``default`` when the file is silent."""
    key = (name or default).strip().lower()
    try:
        return LENGTH_UNITS[key]
    except KeyError:
        raise ValueError(
            f"unrecognised length unit {name!r}; known: {sorted(LENGTH_UNITS)}"
        ) from None


def _is_degenerate(positions: np.ndarray) -> bool:
    """True when header positions carry no usable geometry.

    All-NaN, or every receiver at the same place -- both mean the operator
    did not enter the spread, and both must fall back to caller-supplied
    spacing rather than produce a survey with 24 geophones at one point.
    """
    finite = positions[np.isfinite(positions)]
    return finite.size == 0 or np.allclose(finite, finite[0])


def resolve_spread(
    header_receivers: np.ndarray,
    header_source: np.ndarray,
    *,
    unit_factor: float,
    spacing: float | None,
    source_offset: float | None,
    name: str,
) -> tuple[np.ndarray, float, str]:
    """Receiver and source positions in metres along the line.

    Parameters
    ----------
    header_receivers, header_source
        Per-trace positions as written in the file (NaN where absent), in
        the file's own unit.
    unit_factor
        Metres per file unit.
    spacing, source_offset
        Caller overrides, always in **metres**. Either one overrides the
        corresponding header values entirely.

    Returns
    -------
    receivers_m, source_m, geometry_from
        ``geometry_from`` is ``"headers"``, ``"arguments"``, or ``"mixed"``
        and is recorded in provenance so a reader of the result can tell
        whether the geometry was measured or asserted.
    """
    n = header_receivers.size
    from_args = []

    if spacing is not None:
        rec = np.arange(n, dtype=float) * float(spacing)
        from_args.append("receivers")
    elif _is_degenerate(header_receivers):
        raise ValueError(
            f"{name}: receiver positions in the headers are empty or constant, "
            "so the spread geometry is unknown. Pass spacing=<metres>."
        )
    else:
        rec = (
            pd.Series(header_receivers * unit_factor)
            .interpolate(limit_direction="both")
            .to_numpy()
        )

    if source_offset is not None:
        src = float(source_offset)
        from_args.append("source")
    else:
        finite = header_source[np.isfinite(header_source)]
        if finite.size == 0:
            raise ValueError(
                f"{name}: no source position in the headers. "
                "Pass source_offset=<metres along line>."
            )
        src = float(finite[0]) * unit_factor

    if not from_args:
        origin = "headers"
    elif len(from_args) == 2:
        origin = "arguments"
    else:
        origin = "mixed"
    return rec, src, origin


def uniform_delay(delays: Sequence[float | None], *, name: str) -> float:
    """The single trigger delay for a record, in seconds.

    Both formats permit a different delay per trace; a Geode never writes
    one. If a file ever does, that is a genuinely different acquisition and
    the caller should see it rather than have one value silently chosen.
    """
    values = sorted({float(d) for d in delays if d is not None})
    if not values:
        return 0.0
    if len(values) > 1:
        raise ValueError(
            f"{name}: traces carry different delays {values}; per-trace delay "
            "is not supported. Read the traces separately."
        )
    return values[0]


def build_shot_survey(
    *,
    data: np.ndarray,
    sample_interval: float,
    delay: float,
    receivers_m: np.ndarray,
    source_m: float,
    channels: Sequence[int],
    elevations: np.ndarray | None,
    azimuth: float,
    spatial_ref: SpatialRef | None,
    metadata: dict[str, Any],
    start_time=None,
) -> SeismicSurvey:
    """Place a one-shot along-line spread in a CRS and wrap it as a ``SeismicSurvey``.

    With no ``spatial_ref`` the spread goes on a local metric grid with the
    first geophone at the origin: downstream code can then assume a CRS
    always exists without the driver pretending to know where on Earth the
    line was.
    """
    return build_line_survey(
        records=[
            {
                "data": data,
                "receivers_m": receivers_m,
                "source_m": source_m,
                "channels": channels,
            }
        ],
        sample_interval=sample_interval,
        delay=delay,
        elevations=elevations,
        azimuth=azimuth,
        spatial_ref=spatial_ref,
        metadata=metadata,
        start_time=start_time,
    )


def build_line_survey(
    *,
    records: Sequence[dict[str, Any]],
    sample_interval: float,
    delay: float,
    elevations: np.ndarray | None,
    azimuth: float,
    spatial_ref: SpatialRef | None,
    metadata: dict[str, Any],
    start_time=None,
    receiver_tolerance: float = 0.05,
) -> SeismicSurvey:
    """Place one or more shots into one spread and wrap them as a ``SeismicSurvey``.

    Both Geometrics exports write several field records into one file when
    the operator saves a group of shots together, and those records share the
    spread: the same geophones, different hammer positions. That is one
    ``SeismicSurvey`` with several sources, not several surveys and not -- as
    a naive concatenation would give -- one survey with 48 geophones and one
    shot. :meth:`SeismicSurvey.gather` pulls an individual record back out.

    Parameters
    ----------
    records
        One dict per field record, in file order, with ``data``
        ``(n_traces, n_samples)``, ``receivers_m``, ``source_m`` and
        ``channels``. Sources are labelled ``S1``, ``S2``, ... in this order.
    receiver_tolerance
        How far, in metres, a record's geophone may sit from the first
        record's before the records are judged to be different spreads.
        Positions come from the same field-software table, so any real
        difference is a re-lay of the line rather than survey error.

    With no ``spatial_ref`` the spread goes on a local metric grid with the
    first geophone at the origin: downstream code can then assume a CRS
    always exists without the driver pretending to know where on Earth the
    line was.
    """
    if not records:
        raise ValueError("no records to build a survey from")

    base = np.asarray(records[0]["receivers_m"], dtype=float)
    n = base.size
    for k, rec in enumerate(records[1:], start=2):
        other = np.asarray(rec["receivers_m"], dtype=float)
        if other.size != n or not np.allclose(other, base, atol=receiver_tolerance):
            raise ValueError(
                f"record {k} of this file has a different receiver spread than "
                f"record 1, so the records are not one line and cannot share "
                f"one geometry. Read the records separately."
            )

    sref = spatial_ref or local_grid(0.0, 0.0)

    if elevations is None:
        rec_z = np.zeros(n)
    else:
        rec_z = np.asarray(elevations, dtype=float)
        if rec_z.size != n:
            raise ValueError(
                f"elevations has {rec_z.size} entries but the spread has {n} geophones"
            )

    theta = np.deg2rad(azimuth)
    ux, uy = np.sin(theta), np.cos(theta)
    order = np.argsort(base)

    ids: list[Any] = list(range(1, n + 1))
    xs = list(base * ux)
    ys = list(base * uy)
    zs = list(rec_z)
    roles = ["receiver"] * n

    frames = []
    for k, rec in enumerate(records, start=1):
        source_m = float(rec["source_m"])
        source_id = f"S{k}"
        # Shot elevation interpolated onto the spread rather than assumed zero,
        # so an off-end shot on sloping ground does not sit underground.
        src_z = float(np.interp(source_m, base[order], rec_z[order]))
        ids.append(source_id)
        xs.append(source_m * ux)
        ys.append(source_m * uy)
        zs.append(src_z)
        roles.append("source")
        frames.append(
            pd.DataFrame(
                {
                    "receiver_id": range(1, n + 1),
                    "source_id": source_id,
                    "channel": [int(c) for c in rec["channels"]],
                    "record": rec.get("record", k),
                    "delay": delay,
                }
            )
        )

    geometry = Geometry(ids=ids, x=xs, y=ys, z=zs, roles=roles, spatial_ref=sref)
    data = np.concatenate([np.asarray(rec["data"], dtype=float) for rec in records])
    trace_map = pd.concat(frames, ignore_index=True)
    if len(trace_map) != data.shape[0]:
        raise ValueError(
            f"records describe {len(trace_map)} traces but hold {data.shape[0]}"
        )

    return SeismicSurvey(
        data=data,
        sample_interval=sample_interval,
        geometry=geometry,
        trace_map=trace_map,
        delay=delay,
        start_time=start_time,
        metadata=metadata,
    )


def stack_traces(traces: Sequence[np.ndarray]) -> np.ndarray:
    """``(n_traces, n_samples)`` array, zero-padding any short trace."""
    n_samples = max(t.size for t in traces)
    out = np.zeros((len(traces), n_samples))
    for i, t in enumerate(traces):
        out[i, : t.size] = t
    return out
