"""Georeferenced station occupations from a GNSS survey.

A :class:`PositionTable` is what a corrected GNSS export becomes: one row per
*occupation* -- a station, a time window, a position, and a quality flag. It is
not a :class:`~shallowgeo.core.geometry.Geometry` yet, because a geometry has
one point per station and a survey may occupy a station several times.

Two invariants hold for every table this package produces, and both are
enforced at construction rather than left to the caller:

**Heights are at the ground mark.** Never the antenna phase centre. A GNSS
export usually will not say which it holds, and the difference is a constant
offset that propagates straight into the free-air correction -- for a 2 m
antenna that is 0.62 mGal, which can exceed the entire signal on a
low-relief line. :func:`shallowgeo.positions.read_positions` therefore
requires the caller to declare which the file contains.

**Times are absolute UTC.** Vendor exports write local time with an offset
suffix, and at least one writes a *different* offset for the start and end of
the same occupation. Storing anything but UTC makes occupation matching
quietly wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Geod

from ..core.crs import SpatialRef
from ..core.geometry import Geometry
from ..core.provenance import Provenance

_GEOD = Geod(ellps="WGS84")

#: Columns every table carries. Others (rms_h, rms_v, samples, pdop, ...) are
#: preserved when present but never required.
REQUIRED_COLUMNS = ("name", "longitude", "latitude", "height")


class PositionTable:
    """Station occupations with time windows and quality flags.

    Parameters
    ----------
    frame
        One row per occupation. Must carry :data:`REQUIRED_COLUMNS`; ``start``,
        ``end``, and ``status`` are used when present.
    spatial_ref
        Frame of ``longitude``/``latitude``/``height``. Defaults to WGS84 with
        ellipsoidal heights, which is what every receiver in scope emits.
    heights_reduced
        Must be True. The argument exists so that a caller constructing a table
        directly has to make the same declaration the readers require, rather
        than inheriting the invariant by accident.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        spatial_ref: SpatialRef | None = None,
        heights_reduced: bool = False,
        metadata: dict | None = None,
        provenance: Provenance | None = None,
    ) -> None:
        missing = set(REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"position table missing column(s): {sorted(missing)}")
        if not heights_reduced:
            raise ValueError(
                "heights_reduced=True is required. A PositionTable must hold "
                "ground-mark heights, not antenna phase-centre heights. Use "
                "read_positions(..., heights_are=...) which does the reduction, "
                "or pass heights_reduced=True if you have already done it."
            )
        for col in ("start", "end"):
            if col in frame.columns:
                frame = frame.copy()
                frame[col] = pd.to_datetime(frame[col], utc=True)

        self.frame = frame.reset_index(drop=True)
        self.spatial_ref = spatial_ref or SpatialRef(
            "EPSG:4326", vertical_datum="ellipsoidal"
        )
        self.metadata = dict(metadata or {})
        self.provenance = provenance or Provenance()

    # -- inspection ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def stations(self) -> list:
        return list(pd.unique(self.frame["name"]))

    def duplicates(self) -> pd.DataFrame:
        """Occupations sharing a name, with the distance between them.

        A name reused for two different marks is a data-integrity failure that
        any dictionary-keyed join would swallow silently, so it is surfaced as
        a table rather than resolved by a rule.
        """
        rows = []
        for name, group in self.frame.groupby("name"):
            if len(group) < 2:
                continue
            first = group.iloc[0]
            for _, other in group.iloc[1:].iterrows():
                _, _, sep = _GEOD.inv(
                    first["longitude"], first["latitude"],
                    other["longitude"], other["latitude"],
                )
                rows.append(
                    {
                        "name": name,
                        "separation_m": sep,
                        "height_difference_m": other["height"] - first["height"],
                        "status_a": first.get("status"),
                        "status_b": other.get("status"),
                    }
                )
        return pd.DataFrame(rows)

    def coincident(self, tolerance: float = 2.0) -> pd.DataFrame:
        """Differently-named occupations closer together than *tolerance*.

        Usually a mark re-occupied under a new name in a later session. That is
        a free external check on repeatability -- the height difference between
        two occupations of the same physical point is a better error estimate
        than the receiver's own RMS.
        """
        rows = []
        f = self.frame
        for i in range(len(f)):
            for j in range(i + 1, len(f)):
                a, b = f.iloc[i], f.iloc[j]
                if a["name"] == b["name"]:
                    continue
                _, _, sep = _GEOD.inv(
                    a["longitude"], a["latitude"], b["longitude"], b["latitude"]
                )
                if sep <= tolerance:
                    rows.append(
                        {
                            "name_a": a["name"], "name_b": b["name"],
                            "separation_m": sep,
                            "height_difference_m": b["height"] - a["height"],
                        }
                    )
        return pd.DataFrame(rows)

    # -- filtering ----------------------------------------------------------

    def require_status(self, *accepted: str) -> PositionTable:
        """Keep only occupations whose solution status is in *accepted*.

        Filter on status, not on reported RMS: a FLOAT solution has unresolved
        carrier ambiguities, and its RMS is computed under an assumption that
        did not hold, so it can look indistinguishable from a FIX while being
        decimetres out.
        """
        if "status" not in self.frame.columns:
            raise ValueError("table has no 'status' column to filter on")
        accepted_upper = {s.upper() for s in accepted}
        keep = self.frame["status"].astype(str).str.upper().isin(accepted_upper)
        dropped = int((~keep).sum())
        out = self._replace(self.frame[keep])
        out.provenance.record(
            "require_status", accepted=sorted(accepted_upper), dropped=dropped
        )
        return out

    def _replace(self, frame: pd.DataFrame) -> PositionTable:
        return PositionTable(
            frame.reset_index(drop=True),
            spatial_ref=self.spatial_ref,
            heights_reduced=True,
            metadata=dict(self.metadata),
            provenance=self.provenance.copy(),
        )

    # -- export -------------------------------------------------------------

    def mean_positions(self) -> pd.DataFrame:
        """One row per station, averaging repeat occupations.

        ``height_scatter_m`` is the spread across occupations, which is the
        honest vertical error bar to carry into a free-air correction.
        """
        g = self.frame.groupby("name")
        out = pd.DataFrame(
            {
                "longitude": g["longitude"].mean(),
                "latitude": g["latitude"].mean(),
                "height": g["height"].mean(),
                "height_scatter_m": g["height"].std(ddof=1),
                "n": g.size(),
            }
        ).reset_index()
        return out

    def to_geometry(self, spatial_ref: SpatialRef | None = None) -> Geometry:
        """One station per point, ready for ``coordinates=`` on a reader."""
        means = self.mean_positions()
        geom = Geometry(
            ids=means["name"],
            x=means["longitude"],
            y=means["latitude"],
            z=means["height"],
            roles=["station"] * len(means),
            spatial_ref=self.spatial_ref,
        )
        return geom.to_crs(spatial_ref) if spatial_ref else geom

    def to_coordinates(self, spatial_ref: SpatialRef | None = None) -> dict:
        """``{station: (x, y, z)}`` for the CG-5 and G-857 ``coordinates=``."""
        geom = self.to_geometry(spatial_ref)
        return {
            row["id"]: (row["x"], row["y"], row["z"])
            for _, row in geom.table.iterrows()
        }

    def __repr__(self) -> str:
        status = ""
        if "status" in self.frame.columns:
            counts = self.frame["status"].value_counts().to_dict()
            status = " " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        return (
            f"<PositionTable {len(self)} occupation(s), "
            f"{len(self.stations)} station(s){status}>"
        )
