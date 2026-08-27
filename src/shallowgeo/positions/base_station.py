"""Base station coordinates and the PPP base-shift workflow.

An RTK rover position is only as good as the coordinate the base was told it
was standing on. The usual field sequence is: set the base on an unknown mark,
let it self-survey to an approximate coordinate, compute the rovers against
that, then post-process the base's own observations through a PPP service
(CSRS-PPP, OPUS) to get a much better coordinate. Every rover then has to move
by the same vector.

:func:`apply_base_shift` performs exactly that, and records both coordinates in
provenance so the shift is auditable and cannot be applied twice.

A trap this module guards against: a station *named* something like
``999 - Base`` in a rover export is normally a rover occupation of a mark near
the base, not the base coordinate itself. The assumed base is carried in the
export's own ``Base *`` columns. Confusing the two moves every station by the
distance between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pyproj import Geod

from ..core.crs import SpatialRef
from .table import PositionTable

_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class BaseStation:
    """A single known point: longitude, latitude, ellipsoidal height."""

    longitude: float
    latitude: float
    height: float
    name: str = "base"
    source: str = "coordinates"
    spatial_ref: SpatialRef | None = None

    @classmethod
    def from_coordinates(
        cls, longitude: float, latitude: float, height: float, name: str = "base"
    ) -> BaseStation:
        return cls(float(longitude), float(latitude), float(height), name=name)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        heights_are: str,
        antenna_height: float | None = None,
        profile: str = "generic-csv",
        name: str | None = None,
    ) -> BaseStation:
        """Read a one-station CSV, e.g. a CSRS-PPP result exported to a row.

        Goes through the same reader as a rover file, so the antenna-height
        declaration is required here too -- a base whose height still includes
        the tripod shifts every rover vertically by that amount.
        """
        from .readers import read_positions

        table = read_positions(
            path,
            heights_are=heights_are,
            antenna_height=antenna_height,
            profile=profile,
            on_duplicate="keep",
        )
        if len(table) != 1:
            raise ValueError(
                f"{Path(path).name}: expected exactly one station, found "
                f"{len(table)} ({table.stations}). Pass the single-station file "
                "for the base, not the rover export."
            )
        row = table.frame.iloc[0]
        return cls(
            longitude=float(row["longitude"]),
            latitude=float(row["latitude"]),
            height=float(row["height"]),
            name=name or str(row["name"]),
            source=str(path),
        )

    @classmethod
    def from_table(cls, table: PositionTable, name) -> BaseStation:
        """Take one named station out of a position table."""
        rows = table.frame[table.frame["name"] == name]
        if rows.empty:
            raise KeyError(f"no station named {name!r}; have {table.stations}")
        row = rows.iloc[0]
        return cls(
            longitude=float(row["longitude"]),
            latitude=float(row["latitude"]),
            height=float(row["height"]),
            name=str(name),
            source="position table",
        )

    def separation_from(self, other: BaseStation) -> tuple[float, float]:
        """``(horizontal_m, vertical_m)`` from *other* to self."""
        _, _, horizontal = _GEOD.inv(
            other.longitude, other.latitude, self.longitude, self.latitude
        )
        return horizontal, self.height - other.height

    def __repr__(self) -> str:
        return (
            f"<BaseStation {self.name!r} {self.longitude:.8f} "
            f"{self.latitude:.8f} h={self.height:.3f}>"
        )


def assumed_base_from(table: PositionTable) -> BaseStation:
    """The base coordinate the export says the rovers were computed against.

    Read from the ``base_longitude``/``base_latitude``/``base_height`` columns,
    which is the authoritative source -- not from any row whose *name* happens
    to mention a base.
    """
    needed = {"base_longitude", "base_latitude", "base_height"}
    if not needed <= set(table.frame.columns):
        raise ValueError(
            "position table has no base coordinate columns "
            f"({sorted(needed)}); supply the assumed base explicitly"
        )
    unique = table.frame[sorted(needed)].drop_duplicates()
    if len(unique) != 1:
        raise ValueError(
            f"the export references {len(unique)} different base coordinates, "
            "so it mixes sessions; split it before shifting"
        )
    row = unique.iloc[0]
    return BaseStation(
        longitude=float(row["base_longitude"]),
        latitude=float(row["base_latitude"]),
        height=float(row["base_height"]),
        name="assumed base",
        source="export base columns",
    )


def apply_base_shift(
    table: PositionTable,
    corrected: BaseStation,
    assumed: BaseStation | None = None,
    *,
    max_shift_m: float = 50.0,
) -> PositionTable:
    """Shift every occupation by ``corrected - assumed``.

    Parameters
    ----------
    corrected
        The post-processed base coordinate, e.g. from CSRS-PPP.
    assumed
        What the rovers were originally computed against. Defaults to the
        export's own base columns via :func:`assumed_base_from`.
    max_shift_m
        Refuse shifts larger than this. A PPP correction to a self-surveyed
        base is metres at most; anything larger usually means the *wrong point*
        was passed as the assumed base -- most often a rover occupation named
        like a base rather than the base itself.
    """
    if table.provenance.applied("base_shift"):
        raise ValueError(
            "this table has already been base-shifted; applying a second shift "
            "would double-count the correction"
        )
    assumed = assumed or assumed_base_from(table)
    horizontal, vertical = corrected.separation_from(assumed)

    if horizontal > max_shift_m:
        raise ValueError(
            f"base shift is {horizontal:.2f} m horizontally, beyond "
            f"max_shift_m={max_shift_m}. Check that {assumed.name!r} is really "
            "the coordinate the rovers were computed against -- a station "
            "merely named like a base is usually a rover occupation near it. "
            "Raise max_shift_m if the shift is genuine."
        )

    frame = table.frame.copy()
    frame["longitude"] = frame["longitude"] + (corrected.longitude - assumed.longitude)
    frame["latitude"] = frame["latitude"] + (corrected.latitude - assumed.latitude)
    frame["height"] = frame["height"] + (corrected.height - assumed.height)

    out = PositionTable(
        frame,
        spatial_ref=table.spatial_ref,
        heights_reduced=True,
        metadata=dict(table.metadata),
        provenance=table.provenance.copy(),
    )
    out.provenance.record(
        "base_shift",
        assumed=(assumed.longitude, assumed.latitude, assumed.height),
        corrected=(corrected.longitude, corrected.latitude, corrected.height),
        horizontal_m=round(horizontal, 4),
        vertical_m=round(vertical, 4),
        source=corrected.source,
    )
    return out
