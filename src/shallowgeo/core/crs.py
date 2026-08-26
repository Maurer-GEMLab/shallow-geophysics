"""Explicit horizontal CRS + vertical datum.

The concept doc flags CRS/vertical-datum mismatch as a top real-world error
source, so this package makes it structurally impossible to build a
:class:`~shallowgeo.core.geometry.Geometry` without declaring one.

The vertical datum is tracked separately from the horizontal CRS because the
practical field workflow -- a total station or RTK giving ellipsoidal heights,
a gravity reduction wanting orthometric heights -- routinely mixes them, and a
compound EPSG code is rarely what the instrument actually wrote out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyproj import CRS as _PyprojCRS
from pyproj import Transformer

#: Vertical references we distinguish. ``local`` covers the very common field
#: case of a tape-and-level survey referenced to an arbitrary benchmark.
VERTICAL_DATUMS = ("ellipsoidal", "orthometric", "local", "unknown")


@dataclass(frozen=True)
class SpatialRef:
    """Horizontal CRS plus an explicit vertical reference."""

    crs: Any
    vertical_datum: str = "unknown"
    vertical_epsg: int | None = None

    def __post_init__(self) -> None:
        if self.vertical_datum not in VERTICAL_DATUMS:
            raise ValueError(
                f"vertical_datum must be one of {VERTICAL_DATUMS}, "
                f"got {self.vertical_datum!r}"
            )
        object.__setattr__(self, "crs", _PyprojCRS.from_user_input(self.crs))

    @property
    def is_projected(self) -> bool:
        return bool(self.crs.is_projected)

    @property
    def horizontal_units(self) -> str:
        return self.crs.axis_info[0].unit_name if self.crs.axis_info else "unknown"

    def transformer_to(self, other: SpatialRef) -> Transformer:
        """Horizontal transformer to *other*.

        Vertical datums are deliberately *not* converted here: a real
        ellipsoidal-to-orthometric conversion needs a geoid model, and silently
        faking it is worse than refusing. :meth:`assert_vertically_compatible`
        is the guard callers should use.
        """
        return Transformer.from_crs(self.crs, other.crs, always_xy=True)

    def assert_vertically_compatible(self, other: SpatialRef) -> None:
        known = {self.vertical_datum, other.vertical_datum} - {"unknown"}
        if len(known) > 1:
            raise ValueError(
                f"Refusing to combine vertical datums {self.vertical_datum!r} and "
                f"{other.vertical_datum!r}. Converting between ellipsoidal and "
                "orthometric heights requires a geoid model; do it explicitly."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "crs": self.crs.to_wkt(),
            "vertical_datum": self.vertical_datum,
            "vertical_epsg": self.vertical_epsg,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SpatialRef:
        return cls(**d)

    def __repr__(self) -> str:
        name = self.crs.name
        return f"<SpatialRef {name!r} vertical={self.vertical_datum}>"


def local_grid(origin_lon: float, origin_lat: float) -> SpatialRef:
    """A metre-based local projection centred on a survey.

    For a class exercise laid out with a tape measure and no GPS, this gives a
    defensible metric CRS without asking students to look up a UTM zone.
    """
    proj = (
        f"+proj=tmerc +lat_0={origin_lat} +lon_0={origin_lon} +k=1 "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    return SpatialRef(crs=proj, vertical_datum="local")
