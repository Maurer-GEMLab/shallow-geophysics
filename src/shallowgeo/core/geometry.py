"""Source/receiver/station geometry.

Every method in scope reduces to "things at positions in space," so all four
drivers produce one of these. Refraction and MASW populate ``source`` and
``receiver`` roles; gravity and magnetics populate ``station``.

Coordinates are stored as a plain float array plus a :class:`SpatialRef`
rather than as GeoPandas geometry: it keeps the dependency footprint small
enough to install on a locked-down lab machine, and the array is what the
forward operators want anyway.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .crs import SpatialRef

ROLES = ("source", "receiver", "station")


class Geometry:
    """A table of identified, georeferenced points.

    Parameters
    ----------
    ids
        Identifier per point. Unique within a role, not necessarily globally.
    x, y, z
        Coordinates in ``spatial_ref``. ``z`` is positive up.
    roles
        One of :data:`ROLES` per point.
    spatial_ref
        Required. There is no default and no implicit WGS84.
    """

    def __init__(
        self,
        ids: Iterable,
        x: Iterable[float],
        y: Iterable[float],
        z: Iterable[float],
        roles: Iterable[str],
        spatial_ref: SpatialRef,
    ) -> None:
        if not isinstance(spatial_ref, SpatialRef):
            raise TypeError(
                "spatial_ref must be a SpatialRef. Geometry deliberately has no "
                "default CRS -- see shallowgeo.core.crs.local_grid for a "
                "tape-measure survey."
            )
        frame = pd.DataFrame(
            {
                "id": list(ids),
                "x": np.asarray(x, dtype=float),
                "y": np.asarray(y, dtype=float),
                "z": np.asarray(z, dtype=float),
                "role": list(roles),
            }
        )
        bad = set(frame["role"]) - set(ROLES)
        if bad:
            raise ValueError(f"unknown role(s) {sorted(bad)}; expected {ROLES}")
        self.table = frame
        self.spatial_ref = spatial_ref

    # -- construction -------------------------------------------------------

    @classmethod
    def from_dataframe(cls, frame: pd.DataFrame, spatial_ref: SpatialRef) -> Geometry:
        missing = {"id", "x", "y", "z", "role"} - set(frame.columns)
        if missing:
            raise ValueError(f"missing column(s): {sorted(missing)}")
        return cls(
            frame["id"], frame["x"], frame["y"], frame["z"],
            frame["role"], spatial_ref,
        )

    @classmethod
    def from_line(
        cls,
        n: int,
        spacing: float,
        spatial_ref: SpatialRef,
        *,
        role: str = "receiver",
        start: tuple[float, float, float] = (0.0, 0.0, 0.0),
        azimuth: float = 90.0,
        first_id: int = 1,
    ) -> Geometry:
        """Points evenly spaced along a straight line.

        The overwhelmingly common land-survey layout, and the fallback when a
        SEG-2 file carries no usable geometry -- which, for the Geode, is most
        of the time unless the operator filled in the survey setup.
        """
        theta = np.deg2rad(azimuth)
        offs = np.arange(n, dtype=float) * spacing
        return cls(
            ids=np.arange(first_id, first_id + n),
            x=start[0] + offs * np.sin(theta),
            y=start[1] + offs * np.cos(theta),
            z=np.full(n, start[2], dtype=float),
            roles=[role] * n,
            spatial_ref=spatial_ref,
        )

    # -- access -------------------------------------------------------------

    def of_role(self, role: str) -> pd.DataFrame:
        return self.table[self.table["role"] == role]

    @property
    def sources(self) -> pd.DataFrame:
        return self.of_role("source")

    @property
    def receivers(self) -> pd.DataFrame:
        return self.of_role("receiver")

    @property
    def stations(self) -> pd.DataFrame:
        return self.of_role("station")

    def coords(self, role: str | None = None) -> np.ndarray:
        """``(n, 3)`` float array, optionally restricted to one role."""
        sub = self.table if role is None else self.of_role(role)
        return sub[["x", "y", "z"]].to_numpy(dtype=float)

    # -- transforms ---------------------------------------------------------

    def to_crs(self, target: SpatialRef) -> Geometry:
        """Reproject horizontally. Elevations pass through unchanged."""
        self.spatial_ref.assert_vertically_compatible(target)
        tx = self.spatial_ref.transformer_to(target)
        nx, ny = tx.transform(self.table["x"].to_numpy(), self.table["y"].to_numpy())
        out = self.table.copy()
        out["x"], out["y"] = nx, ny
        return Geometry.from_dataframe(out, target)

    def merge(self, other: Geometry) -> Geometry:
        """Concatenate, reprojecting *other* into this geometry's frame."""
        other = other.to_crs(self.spatial_ref)
        return Geometry.from_dataframe(
            pd.concat([self.table, other.table], ignore_index=True), self.spatial_ref
        )

    def bounds(self) -> tuple[float, float, float, float, float, float]:
        c = self.coords()
        return (*c.min(axis=0), *c.max(axis=0))

    def __len__(self) -> int:
        return len(self.table)

    def __repr__(self) -> str:
        counts = self.table["role"].value_counts().to_dict()
        inner = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        return f"<Geometry {inner} crs={self.spatial_ref.crs.name!r}>"
