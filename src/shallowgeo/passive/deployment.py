"""Turning a raw passive deployment into something a transform can be run on.

Everything here is the unglamorous part that decides whether the dispersion
step has any chance: which nodes actually recorded, over what interval they
overlap, where they were, and what wavelengths that array can resolve.

The array-capability functions are not decoration. A five-node deployment has
a narrow window of wavelengths it can measure, bounded below by spatial
aliasing and above by the aperture, and a dispersion image will cheerfully
draw a confident-looking ridge outside that window that is an artefact of the
geometry. :func:`resolution_limits` is what lets the notebook grey that region
out instead of inviting a student to pick it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.survey import SeismicSurvey


def _as_utc(when) -> pd.Timestamp:
    """Accept a naive or already-localised timestamp, return a UTC one.

    Naive input is read as UTC, which is what the nodes record; localising an
    already-aware timestamp raises rather than converting, hence the split.
    """
    stamp = pd.Timestamp(when)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def node_coverage(survey: SeismicSurvey) -> pd.DataFrame:
    """Per-node recorded fraction, amplitude and time span.

    Coverage is the fraction of the survey's time axis that is not NaN.
    A node that stopped early shows up here as coverage well below 1.0,
    which is the cue to either trim the window or drop the node.
    """
    dt = survey.sample_interval
    rows = []
    for i, node in enumerate(survey.trace_map["receiver_id"]):
        trace = survey.data[i]
        good = np.isfinite(trace)
        idx = np.flatnonzero(good)
        rows.append(
            {
                "node": node,
                "coverage": float(good.mean()),
                "n_samples": int(good.sum()),
                "first_sample": (
                    survey.start_time + pd.Timedelta(seconds=idx[0] * dt)
                    if idx.size
                    else pd.NaT
                ),
                "last_sample": (
                    survey.start_time + pd.Timedelta(seconds=(idx[-1] + 1) * dt)
                    if idx.size
                    else pd.NaT
                ),
                "rms": float(np.std(trace[good])) if idx.size else np.nan,
                "abs_max": float(np.abs(trace[good]).max()) if idx.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def common_window(
    survey: SeismicSurvey, *, nodes: list[str] | None = None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Interval over which every requested node has data.

    Dispersion processing needs simultaneous records; a window where one node
    is missing does not give a shorter answer, it gives a wrong one, because
    the NaN samples have to be filled with something before an FFT.
    """
    dt = survey.sample_interval
    ids = list(survey.trace_map["receiver_id"])
    keep = [ids.index(n) for n in (nodes if nodes is not None else ids)]
    good = np.isfinite(survey.data[keep]).all(axis=0)
    idx = np.flatnonzero(good)
    if idx.size == 0:
        raise ValueError(
            "the requested nodes never recorded simultaneously. Check "
            "node_coverage() and drop the node that stopped early."
        )
    # Longest run of simultaneous coverage, not first-to-last: a node that
    # drops out in the middle would otherwise give a window full of holes.
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    stops = np.r_[idx[breaks], idx[-1]]
    longest = np.argmax(stops - starts)
    return (
        survey.start_time + pd.Timedelta(seconds=starts[longest] * dt),
        survey.start_time + pd.Timedelta(seconds=(stops[longest] + 1) * dt),
    )


def trim(
    survey: SeismicSurvey,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    *,
    nodes: list[str] | None = None,
) -> SeismicSurvey:
    """Subset a deployment in time and by node, keeping geometry consistent."""
    dt = survey.sample_interval
    ids = list(survey.trace_map["receiver_id"])
    keep = [ids.index(n) for n in (nodes if nodes is not None else ids)]

    i0 = 0 if start is None else int(
        round((_as_utc(start) - survey.start_time).total_seconds() / dt)
    )
    i1 = survey.n_samples if end is None else int(
        round((_as_utc(end) - survey.start_time).total_seconds() / dt)
    )
    i0, i1 = max(i0, 0), min(i1, survey.n_samples)
    if i1 <= i0:
        raise ValueError("the requested window is empty")

    kept_ids = [ids[i] for i in keep]
    table = survey.geometry.table
    geometry = type(survey.geometry).from_dataframe(
        table[table["id"].isin(kept_ids)].copy(), survey.geometry.spatial_ref
    )
    out = SeismicSurvey(
        data=survey.data[keep, i0:i1],
        sample_interval=dt,
        geometry=geometry,
        trace_map=survey.trace_map.iloc[keep].reset_index(drop=True),
        start_time=survey.start_time + pd.Timedelta(seconds=i0 * dt),
        metadata=dict(survey.metadata),
        provenance=survey.provenance,
    )
    out.provenance.record(
        "trim", nodes=kept_ids, start=str(out.start_time), n_samples=i1 - i0
    )
    return out


# -- array geometry ---------------------------------------------------------


@dataclass
class ArrayLayout:
    """Station positions in a local metric frame, plus what they imply.

    ``along`` is the projection onto the array's best-fit line. It is only
    meaningful when ``linearity`` is small; for a 2D array use ``east`` and
    ``north`` and the pair separations instead.
    """

    nodes: list[str]
    east: np.ndarray
    north: np.ndarray
    along: np.ndarray
    linearity: float
    azimuth: float
    aperture: float
    separations: pd.DataFrame

    @property
    def is_linear(self) -> bool:
        """True when the stations lie close enough to a line to be treated as one."""
        return self.linearity < 0.15

    def __repr__(self) -> str:
        kind = "linear" if self.is_linear else "2-D"
        return (
            f"ArrayLayout({len(self.nodes)} nodes, {kind}, "
            f"aperture {self.aperture:.1f} m, "
            f"separations {self.separations['distance'].min():.1f}"
            f"-{self.separations['distance'].max():.1f} m)"
        )


def array_layout(survey: SeismicSurvey) -> ArrayLayout:
    """Local east/north coordinates, pair separations and array shape.

    Geographic coordinates are converted to metres on a local tangent plane
    rather than reprojected to UTM: the arrays here are tens of metres across,
    the difference is millimetres, and it keeps the function usable without a
    CRS lookup on a Colab kernel with no network.
    """
    table = survey.geometry.table.set_index("id")
    nodes = [n for n in survey.trace_map["receiver_id"] if n in table.index]
    x = table.loc[nodes, "x"].to_numpy(float)
    y = table.loc[nodes, "y"].to_numpy(float)

    if not survey.geometry.spatial_ref.is_projected:
        lat0, lon0 = float(np.mean(y)), float(np.mean(x))
        east = (x - lon0) * 111320.0 * np.cos(np.radians(lat0))
        north = (y - lat0) * 111320.0
    else:
        east, north = x - x.mean(), y - y.mean()

    points = np.c_[east, north]
    centre = points.mean(axis=0)
    _, singular, basis = np.linalg.svd(points - centre)
    linearity = float(singular[1] / singular[0]) if singular[0] > 0 else 0.0
    along = (points - centre) @ basis[0]
    if along.size and along[0] > along[-1]:
        along, basis = -along, -basis
    azimuth = float(np.degrees(np.arctan2(basis[0, 0], basis[0, 1])) % 180.0)

    rows = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            rows.append(
                {
                    "node_a": nodes[i],
                    "node_b": nodes[j],
                    "index_a": i,
                    "index_b": j,
                    "distance": float(np.hypot(east[i] - east[j], north[i] - north[j])),
                    "along_offset": float(along[j] - along[i]),
                }
            )
    separations = pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)

    return ArrayLayout(
        nodes=nodes,
        east=east,
        north=north,
        along=along,
        linearity=linearity,
        azimuth=azimuth,
        aperture=float(along.max() - along.min()) if along.size else 0.0,
        separations=separations,
    )


def suggest_array(
    n_nodes: int,
    *,
    depth_min: float = 2.0,
    depth_max: float = 30.0,
    velocity: float = 250.0,
) -> pd.DataFrame:
    """Propose node positions for a passive survey of a given depth range.

    Returns a table of local east/north offsets in metres, centred on the
    first node, to be laid out with a tape and compass.

    The design follows from three facts and one choice.

    The wavelengths needed are set by the depth range, at roughly three times
    the depth. The *smallest* separation has to be a fraction of the shortest
    of those wavelengths, and the *largest* a good fraction of the longest, so
    the separations must span the same ratio as the depths -- which means
    spacing them geometrically, not evenly. An evenly spaced array wastes most
    of its stations measuring nearly the same separation, and that is the
    single most common way a passive deployment comes home useless.

    The choice is to spread azimuths as widely as the node count allows, using
    the golden angle so that no two stations line up however many there are.
    Azimuthal spread is what lets the array cope when the noise arrives mostly
    from one direction, which at a quiet site with one nearby road is the
    normal case rather than the exception.

    With very few nodes there is nothing to spare: every station is carrying a
    separation the others cannot, so nothing is held back for redundancy, and
    a single failed node costs a band of the curve.
    """
    if n_nodes < 4:
        raise ValueError(
            "a passive array needs at least 4 nodes to give the three distinct "
            "separations that the Bessel fit solves for velocity and coherent "
            "fraction from"
        )
    lambda_min, lambda_max = 3.0 * depth_min, 3.0 * depth_max
    # Smallest separation inside the first Bessel lobe at the highest useful
    # frequency; largest a third of the longest wavelength.
    r_min = max(0.3 * lambda_min, 1.0)
    r_max = lambda_max / 3.0
    radii = np.geomspace(r_min, r_max, n_nodes - 1)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    azimuths = np.arange(n_nodes - 1) * golden

    rows = [{"node": 1, "east": 0.0, "north": 0.0, "radius": 0.0, "azimuth": 0.0}]
    for i, (r, a) in enumerate(zip(radii, azimuths, strict=True), start=2):
        rows.append(
            {
                "node": i,
                "east": float(r * np.sin(a)),
                "north": float(r * np.cos(a)),
                "radius": float(r),
                "azimuth": float(np.degrees(a) % 360.0),
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["lambda_range"] = (lambda_min, lambda_max)
    frame.attrs["depth_range"] = (depth_min, depth_max)
    frame.attrs["frequency_range"] = (velocity / lambda_max, velocity / lambda_min)
    return frame


@dataclass
class ResolutionLimits:
    """The wavelength band an array can honestly measure, and the depths it maps.

    ``lambda_min`` is the spatial-aliasing floor: a wave sampled at less than
    two stations per wavelength is indistinguishable from a longer one, so
    anything the transform draws below this is a grating lobe.

    ``lambda_max`` is the aperture ceiling. Over an array shorter than a
    wavelength the phase barely changes between the end stations, and the
    apparent velocity runs away to infinity.

    Depths follow the usual engineering rule of thumb that a Rayleigh wave
    samples to roughly a third to a half of its wavelength (Foti et al., 2018,
    section 4). They are indicative, not a resolution kernel.
    """

    lambda_min: float
    lambda_max: float
    min_spacing: float
    aperture: float
    depth_min: float
    depth_max: float

    def frequency_band(self, velocity: float) -> tuple[float, float]:
        """Frequencies the array can resolve, given a nominal phase velocity."""
        return velocity / self.lambda_max, velocity / self.lambda_min

    def __repr__(self) -> str:
        return (
            f"ResolutionLimits(lambda {self.lambda_min:.1f}-{self.lambda_max:.1f} m, "
            f"depth ~{self.depth_min:.1f}-{self.depth_max:.1f} m)"
        )


def resolution_limits(
    layout: ArrayLayout, *, aperture_factor: float = 2.0
) -> ResolutionLimits:
    """Wavelength and depth limits for an array.

    ``aperture_factor`` is how many array lengths the longest trustworthy
    wavelength is taken to be. Published practice ranges from 1 (strict, Foti
    et al. 2018) to 3 (optimistic, common in SPAC work); 2 is the middle and
    the notebook says which it used.
    """
    distances = layout.separations["distance"].to_numpy()
    min_spacing = float(distances.min()) if distances.size else np.nan
    lambda_min = 2.0 * min_spacing
    lambda_max = aperture_factor * layout.aperture
    return ResolutionLimits(
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        min_spacing=min_spacing,
        aperture=layout.aperture,
        depth_min=lambda_min / 3.0,
        depth_max=lambda_max / 3.0,
    )
