"""Passive MASW: using traffic and footfalls as the source.

When ambient noise at a site is too weak to support SPAC -- quiet rural
ground, where each node mostly records its own wind noise -- there is usually
still a usable signal in the transients. A vehicle on a nearby road or a
person walking the line puts a broadband surface-wave train through the array
from a known-ish direction, and that is an active-source experiment in
everything but the trigger. Park and Miller (2008) formalised this as roadside
passive MASW; the processing below is that method.

The differences from an active shot are all in the bookkeeping:

*There is no trigger*, so the windows have to be found. :func:`detect_events`
looks for times when every node gets loud together, which rejects the far more
common case of one node getting loud by itself.

*The source direction is unknown and varies between events.* Each window is
therefore transformed twice, once for each direction along the line, and the
stronger result is kept. Stacking without doing this smears the ridge across
both signs of slowness.

*Amplitude differs between nodes by orders of magnitude* -- coupling, gain,
distance to the road. Each trace is normalised before the transform so that
the stack is not simply a picture of the loudest node.

The limiting factor is the array, not the method. A five-node line at 20 m
spacing aliases above a 40 m wavelength no matter how clean the events are, so
read :func:`shallowgeo.passive.resolution_limits` before believing the
high-frequency end of any image produced here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from ..core.survey import SeismicSurvey
from ..surfacewave.dispersion import DispersionCurve, DispersionImage, phase_shift
from .deployment import ArrayLayout, array_layout


@dataclass
class EventWindow:
    """One window of the record that contains a transient.

    ``start_sample`` is the index into the survey, and is what the transforms
    read. It is carried explicitly rather than derived from ``index`` because
    windows do not always lie on a grid: :func:`detect_events` scans on fixed
    blocks, but :func:`shot_windows` places windows at arbitrary shot times.
    """

    index: int
    start_sample: int
    start_time: pd.Timestamp
    duration: float
    score: float
    rms: np.ndarray
    label: str = ""

    def __repr__(self) -> str:
        name = f"{self.label}, " if self.label else ""
        return (
            f"EventWindow({name}{self.start_time:%H:%M:%S}, "
            f"score {self.score:.1f}x background)"
        )


def detect_events(
    survey: SeismicSurvey,
    *,
    window: float = 4.0,
    threshold: float = 3.0,
    max_events: int | None = None,
    band: tuple[float, float] | None = (2.0, 40.0),
    layout: ArrayLayout | None = None,
) -> list[EventWindow]:
    """Find windows where every node is simultaneously above its background.

    The score of a window is the *minimum* over nodes of that node's RMS
    divided by its own median RMS. Taking the minimum rather than the mean is
    the whole point: it requires all nodes to participate, so a single node
    being thumped -- which is most of what a loud window turns out to be --
    scores near 1 and is never selected.

    Scoring each node against its own median handles the 30-fold differences
    in background level between nodes without any manual gain balancing.
    """
    layout = layout or array_layout(survey)
    fs = survey.sample_rate
    data = np.nan_to_num(survey.data)
    if band is not None:
        sos = butter(4, band, btype="band", fs=fs, output="sos")
        data = sosfiltfilt(sos, data, axis=1)

    n = int(round(window * fs))
    count = data.shape[1] // n
    if count < 2:
        raise ValueError(f"record holds fewer than two {window:g} s windows")
    blocks = data[:, : count * n].reshape(data.shape[0], count, n)
    rms = blocks.std(axis=2)

    background = np.median(rms, axis=1, keepdims=True)
    normalised = rms / np.where(background > 0, background, 1.0)
    score = normalised.min(axis=0)

    order = np.argsort(score)[::-1]
    order = order[score[order] >= threshold]
    if max_events is not None:
        order = order[:max_events]

    return [
        EventWindow(
            index=int(i),
            start_sample=int(i) * n,
            start_time=survey.start_time + pd.Timedelta(seconds=i * window),
            duration=window,
            score=float(score[i]),
            rms=rms[:, i].copy(),
        )
        for i in order
    ]


def shot_windows(
    survey: SeismicSurvey,
    shot_times,
    *,
    window: float = 8.0,
    pre: float = 0.5,
    search: float = 3.0,
    band: tuple[float, float] | None = (2.0, 40.0),
    labels=None,
) -> list[EventWindow]:
    """Windows at *known* shot times, for nodes used as an active-source array.

    Free-running nodes recording through an active survey capture every shot,
    and because they are not tied to the seismograph's cable they can sit at
    offsets the spread cannot reach. All that is needed to use them is the
    shot times, which the seismograph already wrote into its own file headers.

    The times only have to be approximate. Seismograph headers carry a
    one-second stamp from a clock that is usually not GPS-disciplined, so each
    window is refined by looking for the actual amplitude jump within
    ``search`` seconds of the nominal time. Getting this exactly right matters
    less than it seems: a phase-shift transform reads the *relative* phase
    between receivers, so a common timing error moves the whole wavetrain
    within the window without changing the measured velocity.

    For the same reason the **source position is not needed**. Offsetting
    every receiver by the same distance multiplies the transform by a phase
    factor of unit modulus and leaves the image identical. What the source
    position does decide is which receivers are usable: the wave must reach
    them all travelling the same way, so a shot inside the array means
    discarding the receivers on the far side.

    Parameters
    ----------
    shot_times
        Timestamps, UTC. Naive values are read as UTC. Times outside the
        record are skipped.
    search
        Half-width of the refinement search. Set to 0 to trust the times.
    """
    fs = survey.sample_rate
    data = np.nan_to_num(survey.data)
    if band is not None:
        sos = butter(4, band, btype="band", fs=fs, output="sos")
        detect = sosfiltfilt(sos, data, axis=1)
    else:
        detect = data

    envelope = np.abs(detect)
    background = np.median(envelope, axis=1, keepdims=True)
    n = int(round(window * fs))
    pre_n = int(round(pre * fs))
    labels = list(labels) if labels is not None else [""] * len(list(shot_times))

    out = []
    for index, (stamp, label) in enumerate(zip(shot_times, labels, strict=False)):
        stamp = pd.Timestamp(stamp)
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        nominal = int(round((stamp - survey.start_time).total_seconds() * fs))

        onset = nominal
        if search > 0:
            half = int(round(search * fs))
            lo, hi = max(nominal - half, 0), min(nominal + half, data.shape[1])
            if hi - lo > 1:
                # Earliest sample where every node is simultaneously above its
                # background: a shot is seen by the whole array at once.
                hot = (envelope[:, lo:hi] > 3.0 * background).all(axis=0)
                found = np.flatnonzero(hot)
                if found.size:
                    onset = lo + int(found[0])

        start = onset - pre_n
        if start < 0 or start + n > data.shape[1]:
            continue
        block = envelope[:, start : start + n]
        out.append(
            EventWindow(
                index=index,
                start_sample=start,
                start_time=survey.start_time + pd.Timedelta(seconds=start / fs),
                duration=window,
                score=float((block.max(axis=1) / background.ravel()).min()),
                rms=data[:, start : start + n].std(axis=1),
                label=str(label),
            )
        )
    if not out:
        raise ValueError(
            "no shot time fell inside the record. Check the time zone: "
            "seismograph headers are usually local time and node records UTC."
        )
    return out


def event_dispersion_image(
    survey: SeismicSurvey,
    events: list[EventWindow],
    *,
    fmin: float = 2.0,
    fmax: float = 40.0,
    vmin: float = 50.0,
    vmax: float = 800.0,
    dv: float = 2.0,
    layout: ArrayLayout | None = None,
    both_directions: bool = True,
) -> DispersionImage:
    """Stack phase-shift transforms of many event windows into one image.

    Each window is normalised per trace, transformed with
    :func:`~shallowgeo.surfacewave.dispersion.phase_shift` for each assumed
    propagation direction, and the per-window maximum over directions is
    added to the stack. Stacking the *normalised* image of each event rather
    than raw power stops one loud vehicle from being the entire answer.
    """
    if not events:
        raise ValueError("no event windows given; lower the detection threshold")
    layout = layout or array_layout(survey)
    if not layout.is_linear:
        raise ValueError(
            f"phase-shift imaging assumes a line array, but this one has "
            f"linearity {layout.linearity:.2f}. Use SPAC for a 2-D array."
        )

    fs = survey.sample_rate
    data = np.nan_to_num(survey.data)
    n = int(round(events[0].duration * fs))
    along = layout.along
    directions = [along - along.min()]
    if both_directions:
        directions.append(along.max() - along)

    stack = None
    image = None
    for event in events:
        start = event.start_sample
        block = data[:, start : start + n]
        if block.shape[1] < n:
            continue
        block = block - block.mean(axis=1, keepdims=True)
        scale = block.std(axis=1, keepdims=True)
        block = block / np.where(scale > 0, scale, 1.0)

        best = None
        for offsets in directions:
            image = phase_shift(
                block, offsets, 1.0 / fs,
                fmin=fmin, fmax=fmax, vmin=vmin, vmax=vmax, dv=dv,
                normalize=True,
            )
            best = image.image if best is None else np.maximum(best, image.image)
        stack = best if stack is None else stack + best

    if stack is None or image is None:
        raise ValueError("no complete event window could be transformed")
    stack = stack / stack.max(axis=1, keepdims=True)

    return DispersionImage(
        frequency=image.frequency,
        velocity=image.velocity,
        image=stack,
        offsets=along,
        metadata={
            "method": "passive_masw_event_stack",
            "n_events": len(events),
            "both_directions": both_directions,
            "aperture": layout.aperture,
        },
    )


def two_station_dispersion(
    survey: SeismicSurvey,
    events: list[EventWindow],
    *,
    pair: tuple[int, int] | None = None,
    fmin: float = 2.0,
    fmax: float = 40.0,
    min_coherence: float = 0.3,
    layout: ArrayLayout | None = None,
) -> DispersionCurve:
    """Phase velocity from the cross-spectral phase of one station pair.

    The oldest surface-wave method there is: stack the cross-spectrum over
    events, unwrap its phase, and divide. Its value here is that it is not
    bounded by the spatial-sampling limit that stops the multichannel
    transform -- two stations cannot alias each other -- so it reaches to
    shorter wavelengths than :func:`event_dispersion_image` on the same array.

    What it buys in bandwidth it pays for in the cycle ambiguity: the unwrap
    starts from the lowest frequency and any error there propagates to every
    point above it as a whole extra cycle. Treat a curve from a single pair as
    a cross-check on the image, not as a replacement for it, and compare pairs
    against each other before trusting either.
    """
    layout = layout or array_layout(survey)
    if pair is None:
        widest = layout.separations.iloc[-1]
        pair = (int(widest["index_a"]), int(widest["index_b"]))
    i, j = pair
    distance = abs(layout.along[j] - layout.along[i])
    if distance <= 0:
        raise ValueError("the chosen pair is co-located along the array line")

    fs = survey.sample_rate
    data = np.nan_to_num(survey.data)
    n = int(round(events[0].duration * fs))
    taper = np.hanning(n)

    cross = power_i = power_j = None
    for event in events:
        start = event.start_sample
        block = data[:, start : start + n]
        if block.shape[1] < n:
            continue
        a = np.fft.rfft((block[i] - block[i].mean()) * taper)
        b = np.fft.rfft((block[j] - block[j].mean()) * taper)
        product, pa, pb = a * np.conj(b), np.abs(a) ** 2, np.abs(b) ** 2
        if cross is None:
            cross, power_i, power_j = product, pa, pb
        else:
            cross, power_i, power_j = cross + product, power_i + pa, power_j + pb
    if cross is None:
        raise ValueError("no complete event window could be transformed")

    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    band = (freqs >= fmin) & (freqs <= fmax)
    freqs, cross = freqs[band], cross[band]
    coherence = np.abs(cross) / np.sqrt(power_i[band] * power_j[band] + 1e-30)
    phase = np.unwrap(np.angle(cross))

    with np.errstate(divide="ignore", invalid="ignore"):
        # Passive events arrive from either end of the line, so the sign of
        # the phase carries direction, not information about the ground. The
        # magnitude is the measurement.
        velocity = 2.0 * np.pi * freqs * distance / np.abs(phase)
    good = (
        np.isfinite(velocity)
        & (velocity > 0)
        & (coherence >= min_coherence)
        & (np.abs(phase) > np.radians(20.0))  # below this the unwrap is noise
    )
    if good.sum() < 3:
        raise ValueError(
            f"pair {layout.nodes[i]}-{layout.nodes[j]} has too few frequencies with "
            f"coherence above {min_coherence}; try another pair or more events"
        )

    return DispersionCurve(
        frequency=freqs[good],
        velocity=velocity[good],
        mode=0,
        wave="rayleigh",
        metadata={
            "method": "two_station_phase",
            "distance": distance,
            "nodes": [layout.nodes[i], layout.nodes[j]],
            "n_events": len(events),
            "coherence": coherence[good],
            "warning": "cycle ambiguity: verify against another pair",
        },
    )
