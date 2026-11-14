"""How the EM sensor was carried: hand, sled, or drone.

A frequency-domain EM reading is a ratio of the field the ground makes to
the field the transmitter makes, and how big that ratio is depends on how
far the coils were above the ground at least as much as on the ground. The
GEM-2 is the same ski whether it is carried at the hip, pulled on a sled a
hand's width off the grass, or slung under a drone several metres up, so
the reading alone does not say which. :class:`Deployment` is the record of
that, and it travels with the data.

The numbers that make it matter, for the GEM-2's 1.66 m horizontal
coplanar (HCP) coil pair over uniform ground:

- the quadrature reading at 1 m above the ground is 64 % of the reading at
  0.2 m (McNeill's cumulative response, :func:`~shallowgeo.em.forward.lin_height_factor`),
  so an apparent conductivity computed with the wrong platform height is off
  by half;
- the in-phase reading of magnetic soil **changes sign** at 0.59 m (0.354 x
  the coil separation): a sled and a hand-carried survey over the same soil
  see opposite signs;
- a drone at 3 m sees 25 % of the ground signal it would at 0.2 m, and its
  height changes from sample to sample, so it needs a height per reading,
  not one number.

That is why :func:`~shallowgeo.drivers.gem2.read_gem2` asks for the platform
and has no default for it, the same way positions ask for ``heights_are``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

#: Platforms the deployment record knows, with what each one assumes.
#: ``height`` is a *nominal* sensor height in metres, used only when the
#: survey supplies none; ``None`` means the platform has no defensible
#: nominal value and a height must be given.
PLATFORMS: dict[str, dict[str, Any]] = {
    "handheld": {
        "height": 1.0,
        "description": "carried by an operator on the shoulder strap, ski at about hip height",
        "notes": (
            "Height wanders by +/-0.1 m with the operator's gait and the ground. "
            "Lines walked in opposite directions put the operator on opposite sides "
            "of the ski, so compare them before trusting small differences (heading error). "
            "The operator must carry no metal: keys, phone, steel toe caps, belt buckle."
        ),
    },
    "sled": {
        "height": 0.25,
        "description": "towed on a non-metallic sled or cart, ski close to the ground",
        "notes": (
            "Measure the height of the ski's centre line on your own sled; 0.25 m is "
            "only a placeholder. Keep the tow vehicle or towing person at least 3-4 m "
            "(two coil separations) ahead, and record that distance as tow_distance. "
            "The GPS antenna is usually on the tow vehicle, so set gps_offset "
            "(the layback) or every anomaly lands in the wrong place."
        ),
    },
    "drone": {
        "height": None,
        "description": "slung below or mounted on an unmanned aircraft",
        "notes": (
            "Height above ground changes every sample and must come from a laser or "
            "radar altimeter, or from the GPS altitude minus a ground elevation "
            "model (heights_from_altitude). Record the sling length: the sensor hangs "
            "below the aircraft and the GPS. Motor and power-electronics noise "
            "depends on throttle; check the noise at altitude before the survey."
        ),
    },
}

#: Coil orientations. ``HCP`` (horizontal coplanar, vertical dipoles) is the
#: ski lying flat; ``VCP`` (vertical coplanar, horizontal dipoles) is the ski
#: turned on its edge.
ORIENTATIONS = ("HCP", "VCP")

#: Which way the boom (the line from transmitter to receiver) points
#: relative to the direction of travel.
BOOM_DIRECTIONS = ("inline", "crossline")


@dataclass(frozen=True)
class Instrument:
    """A frequency-domain EM sensor's fixed geometry.

    Attributes
    ----------
    coil_separation
        Transmitter-to-receiver distance in metres. Every apparent
        conductivity, every depth estimate and every height correction is
        scaled by it.
    bucking_distance
        Transmitter-to-bucking-coil distance in metres, when there is one.
        Informational; the bucking coil cancels the primary field at the
        receiver and does not enter the ground response at the accuracy used
        here.
    """

    name: str
    coil_separation: float
    bucking_distance: float | None = None
    vendor: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Geophex GEM-2 ski: 1.66 m transmitter-receiver separation, bucking coil
#: 1.035 m from the transmitter (Huang & Won 2003, *Geophysics* 68, 1626).
GEM2 = Instrument(name="GEM-2", coil_separation=1.66, bucking_distance=1.035,
                  vendor="Geophex")


@dataclass(frozen=True)
class Deployment:
    """How and at what height the sensor was carried.

    Build one with :meth:`for_platform`, which fills in the platform's
    nominal values and records which numbers were assumed rather than
    measured.

    Attributes
    ----------
    platform
        ``"handheld"``, ``"sled"`` or ``"drone"`` (see :data:`PLATFORMS`).
    sensor_height
        Height of the coil centre line above the ground, metres. A survey
        whose readings carry a ``height`` column uses that per sample instead.
    height_source
        Where ``sensor_height`` came from: ``"measured"``, ``"nominal"`` (the
        platform default, which is a guess), ``"altimeter"``, or
        ``"gps_minus_ground"``.
    orientation
        ``"HCP"`` or ``"VCP"``.
    boom_direction
        ``"inline"`` (boom along the direction of travel) or ``"crossline"``.
    gps_offset
        Antenna position relative to the sensor centre, ``(forward, right,
        up)`` in metres along the direction of travel. A sled towed 4 m
        behind an antenna on the tow vehicle has ``(4.0, 0, ...)``.
    tow_distance
        Sled: sensor centre to the tow vehicle or towing person, metres.
    sling_length
        Drone: aircraft to sensor centre, metres.
    carrier
        Free text: the operator, the vehicle, or the aircraft model.
    """

    platform: str
    sensor_height: float | None
    height_source: str = "measured"
    orientation: str = "HCP"
    boom_direction: str = "inline"
    gps_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tow_distance: float | None = None
    sling_length: float | None = None
    carrier: str | None = None
    notes: str = ""
    assumed: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.platform not in PLATFORMS:
            raise ValueError(
                f"platform must be one of {sorted(PLATFORMS)}, got {self.platform!r}"
            )
        if self.orientation not in ORIENTATIONS:
            raise ValueError(
                f"orientation must be one of {ORIENTATIONS}, got {self.orientation!r}"
            )
        if self.boom_direction not in BOOM_DIRECTIONS:
            raise ValueError(
                f"boom_direction must be one of {BOOM_DIRECTIONS}, "
                f"got {self.boom_direction!r}"
            )
        if self.sensor_height is not None and self.sensor_height < 0:
            raise ValueError(f"sensor_height must be >= 0, got {self.sensor_height}")
        object.__setattr__(self, "gps_offset", tuple(float(v) for v in self.gps_offset))
        if len(self.gps_offset) != 3:
            raise ValueError("gps_offset is (forward, right, up), three numbers")

    @classmethod
    def for_platform(
        cls,
        platform: str,
        *,
        sensor_height: float | None = None,
        height_source: str | None = None,
        orientation: str = "HCP",
        boom_direction: str = "inline",
        gps_offset: tuple[float, float, float] | None = None,
        tow_distance: float | None = None,
        sling_length: float | None = None,
        carrier: str | None = None,
        notes: str = "",
        heights_per_sample: bool = False,
    ) -> Deployment:
        """A deployment record with the platform's defaults filled in.

        Every value taken from a default rather than given is listed in
        :attr:`assumed`, so a report can say which numbers are guesses.

        Parameters
        ----------
        heights_per_sample
            The readings carry their own ``height`` column (an altimeter, or
            :func:`~shallowgeo.em.processing.heights_from_altitude`). A drone
            survey needs either this or ``sensor_height``.
        """
        if platform not in PLATFORMS:
            raise ValueError(
                f"platform must be one of {sorted(PLATFORMS)}, got {platform!r}"
            )
        assumed = []
        if sensor_height is None:
            sensor_height = PLATFORMS[platform]["height"]
            if sensor_height is None and not heights_per_sample:
                raise ValueError(
                    f"a {platform} survey has no nominal sensor height: give "
                    "sensor_height= (metres above ground), or a per-sample height "
                    "column. A drone's height above ground changes every sample "
                    "and controls the size of every reading."
                )
            if sensor_height is not None:
                assumed.append("sensor_height")
            height_source = height_source or ("nominal" if sensor_height is not None
                                              else "per_sample")
        else:
            height_source = height_source or "measured"
        if gps_offset is None:
            gps_offset = (0.0, 0.0, 0.0)
            assumed.append("gps_offset")
        if platform == "sled" and tow_distance is None:
            assumed.append("tow_distance")
        if platform == "drone" and sling_length is None:
            assumed.append("sling_length")
        return cls(
            platform=platform, sensor_height=sensor_height, height_source=height_source,
            orientation=orientation, boom_direction=boom_direction,
            gps_offset=gps_offset, tow_distance=tow_distance,
            sling_length=sling_length, carrier=carrier, notes=notes,
            assumed=tuple(assumed),
        )

    @property
    def description(self) -> str:
        return PLATFORMS[self.platform]["description"]

    @property
    def advice(self) -> str:
        """What to check for this platform."""
        return PLATFORMS[self.platform]["notes"]

    def with_changes(self, **changes) -> Deployment:
        """Copy with fields replaced; a replaced field is no longer assumed."""
        assumed = tuple(a for a in self.assumed if a not in changes)
        if "sensor_height" in changes and "height_source" not in changes:
            changes["height_source"] = "measured"
        return replace(self, assumed=assumed, **changes)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["gps_offset"] = list(self.gps_offset)
        out["assumed"] = list(self.assumed)
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Deployment:
        d = dict(d)
        d["gps_offset"] = tuple(d.get("gps_offset", (0.0, 0.0, 0.0)))
        d["assumed"] = tuple(d.get("assumed", ()))
        return cls(**d)

    def summary(self) -> str:
        h = ("per sample" if self.sensor_height is None
             else f"{self.sensor_height:.2f} m ({self.height_source})")
        lines = [
            f"platform        {self.platform}: {self.description}",
            f"sensor height   {h}",
            f"coils           {self.orientation}, boom {self.boom_direction}",
            (f"GPS offset      forward {self.gps_offset[0]:+.2f} m, right "
             f"{self.gps_offset[1]:+.2f} m, up {self.gps_offset[2]:+.2f} m"),
        ]
        if self.platform == "sled":
            lines.append(f"tow distance    {self.tow_distance} m")
        if self.platform == "drone":
            lines.append(f"sling length    {self.sling_length} m")
        if self.carrier:
            lines.append(f"carrier         {self.carrier}")
        if self.assumed:
            lines.append(f"ASSUMED, not recorded: {', '.join(self.assumed)}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
