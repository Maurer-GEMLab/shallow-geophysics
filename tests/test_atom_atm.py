"""ATOM-1C native ``.atm`` reader.

The format has no published specification, so these tests pin down what was
reverse-engineered from the field corpus. Several of them look pedantic and
are not: the sample codec and the coordinate convention are both places where
a plausible misreading produces numbers that survive casual inspection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from atm_writer import write_atm

from shallowgeo.drivers.atom_atm import (
    _decode_samples,
    _dm_to_degrees,
    _is_atm,
    group_deployments,
    read_atm,
    read_atm_deployment,
    read_atm_file,
    scan_atm,
)

FS = 250.0


@pytest.fixture
def one_minute(tmp_path):
    rng = np.random.default_rng(0)
    counts = rng.integers(-3000, 3000, size=15000)
    path = write_atm(tmp_path / "10152500.atm", counts)
    return path, counts


def test_recognises_its_own_files(one_minute, tmp_path):
    path, _ = one_minute
    assert _is_atm(path)
    other = tmp_path / "not_atom.dat"
    other.write_bytes(b"SEG2\x00" + b"\x00" * 500)
    assert not _is_atm(other)


def test_samples_round_trip_including_negatives(one_minute):
    path, counts = one_minute
    header = read_atm_file(path)
    assert header["data"].size == counts.size
    np.testing.assert_array_equal(header["data"], counts)
    assert (counts < 0).any(), "fixture must exercise the sign-extension byte"


def test_sample_at_the_24_bit_extremes(tmp_path):
    extremes = np.array([0, 1, -1, (1 << 23) - 1, -(1 << 23), 255, -256])
    path = write_atm(tmp_path / "10152501.atm", extremes)
    np.testing.assert_array_equal(read_atm_file(path)["data"], extremes)


def test_plain_int32_reading_would_be_wrong(one_minute):
    """The four bytes are not a little-endian int32.

    The sign byte comes first, so reading the group as an ordinary int32 puts
    it in the least significant position and shifts the real sample up a byte.
    Every count comes out 256 times too large, with a further 255 on the
    negatives from the sign byte itself.

    A uniform factor of 256 is exactly the kind of error that survives
    inspection -- the waveforms and the spectra keep their shape, and only an
    absolute amplitude would give it away.
    """
    path, counts = one_minute
    naive = np.frombuffer(path.read_bytes()[512:], dtype="<i4")
    expected = 256 * counts + np.where(counts < 0, 255, 0)
    np.testing.assert_array_equal(naive, expected)
    assert not np.array_equal(naive, counts)


def test_sign_byte_disagreement_is_an_error(tmp_path):
    counts = np.array([-5, 10, -20, 30])
    path = write_atm(tmp_path / "10152502.atm", counts, corrupt_sign_byte=True)
    with pytest.raises(ValueError, match="sign byte"):
        read_atm_file(path)
    # ...but it is a check, not a codec: skipping it still decodes.
    assert read_atm_file(path, check_sign_byte=False)["data"].size == 4


def test_trailing_partial_sample_is_dropped():
    payload = b"\x00\x01\x00\x00" + b"\x00\x02\x00"  # one sample plus 3 bytes
    np.testing.assert_array_equal(_decode_samples(payload), [1])


def test_coordinates_are_degrees_and_decimal_minutes():
    """``N3755.5010`` is 37 deg 55.5010 min, not 37.755010 deg.

    Both readings land in Missouri, so a map check does not catch the error;
    only the arithmetic does.
    """
    assert _dm_to_degrees("N3755.5010") == pytest.approx(37.925017, abs=1e-6)
    assert _dm_to_degrees("W09146.5853") == pytest.approx(-91.776422, abs=1e-6)
    assert _dm_to_degrees("S3755.5010") == pytest.approx(-37.925017, abs=1e-6)
    assert _dm_to_degrees("E09146.5853") == pytest.approx(91.776422, abs=1e-6)


def test_header_fields(one_minute):
    path, _ = one_minute
    header = read_atm_file(path)
    assert header["unit_id"] == "101525"
    assert header["adc_bits"] == 24
    assert header["start_time"] == pd.Timestamp("2026-09-18 17:00:00", tz="UTC")
    assert header["latitude"] == pytest.approx(37.925017, abs=1e-5)
    assert header["longitude"] == pytest.approx(-91.776422, abs=1e-5)


def test_read_atm_builds_a_passive_survey(one_minute):
    path, counts = one_minute
    survey = read_atm(path)
    assert survey.is_passive, "a node record has no source"
    assert survey.n_traces == 1
    assert survey.n_samples == counts.size
    assert survey.duration == pytest.approx(60.0)
    assert survey.sample_rate == pytest.approx(FS)


def _deployment(tmp_path, *, nodes=("101525", "101527", "101528"), minutes=6):
    """Three nodes on a 20 m line, split across two hourly folders."""
    rng = np.random.default_rng(1)
    for index, node in enumerate(nodes):
        for minute in range(minutes):
            hour, mm = (16, 57 + minute) if minute < 3 else (17, minute - 3)
            folder = tmp_path / f"260918{hour:02d}"
            folder.mkdir(exist_ok=True)
            write_atm(
                folder / f"{node}{mm:02d}.atm",
                rng.integers(-500, 500, size=15000),
                unit_id=node,
                start=f"2026/09/18 {hour:02d}:{mm:02d}:00",
                latitude=37.925017 + index * 20.0 / 111320.0,
            )
    return tmp_path


def test_deployment_merges_hourly_folders(tmp_path):
    root = _deployment(tmp_path)
    survey = read_atm_deployment(root)
    assert survey.n_traces == 3
    assert survey.metadata["n_files"] == 18
    assert survey.metadata["folders"] == ["26091816", "26091817"]
    # Six contiguous minutes across the hour boundary, not two separate runs.
    assert survey.duration == pytest.approx(360.0)
    assert survey.start_time == pd.Timestamp("2026-09-18 16:57:00", tz="UTC")
    assert not np.isnan(survey.data).any()


def test_gaps_are_nan_not_zero(tmp_path):
    """A silent zero-fill is a loud impulse at both edges of the gap."""
    root = _deployment(tmp_path)
    (root / "26091717").mkdir(exist_ok=True)
    missing = next(root.rglob("10152501.atm"))
    missing.unlink()
    survey = read_atm_deployment(root)
    row = list(survey.trace_map["receiver_id"]).index("101525")
    assert np.isnan(survey.data[row]).any()
    assert survey.metadata["coverage"]["101525"] < 1.0
    assert survey.metadata["coverage"]["101527"] == pytest.approx(1.0)


def test_nodes_that_start_at_different_times_are_aligned(tmp_path):
    """Alignment is on GPS time, not on file order."""
    rng = np.random.default_rng(2)
    folder = tmp_path / "26091817"
    folder.mkdir()
    write_atm(folder / "10152500.atm", rng.integers(-100, 100, 15000),
              unit_id="101525", start="2026/09/18 17:00:00")
    write_atm(folder / "10152702.atm", rng.integers(-100, 100, 15000),
              unit_id="101527", start="2026/09/18 17:02:00",
              latitude=37.9252)
    survey = read_atm_deployment(folder)
    assert survey.duration == pytest.approx(180.0)
    late = list(survey.trace_map["receiver_id"]).index("101527")
    # The first two minutes of the late node are absent, not shifted earlier.
    assert np.isnan(survey.data[late, : int(120 * FS)]).all()
    assert not np.isnan(survey.data[late, int(120 * FS):]).any()


def test_applesingle_stubs_are_ignored(tmp_path):
    root = _deployment(tmp_path, nodes=("101525",), minutes=3)
    stub = root / "26091816" / "._10152557.atm"
    stub.write_bytes(b"\x00\x05\x16\x07" + b"\x00" * 200)
    assert read_atm_deployment(root).n_traces == 1


def test_scan_and_group_deployments(tmp_path):
    _deployment(tmp_path)
    scan = scan_atm(tmp_path.rglob("*.atm"))
    assert len(scan) == 18
    assert set(scan["node"]) == {"101525", "101527", "101528"}

    groups = group_deployments(scan)
    assert len(groups) == 1, "one afternoon in two folders is one deployment"
    assert groups.iloc[0]["n_nodes"] == 3
    assert groups.iloc[0]["folders"] == ["26091816", "26091817"]


def test_group_deployments_splits_on_time_not_folder(tmp_path):
    rng = np.random.default_rng(3)
    for hour, day in ((17, "18"), (17, "19")):
        folder = tmp_path / f"2609{day}{hour}"
        folder.mkdir()
        write_atm(folder / "10152500.atm", rng.integers(-50, 50, 15000),
                  start=f"2026/09/{day} {hour}:00:00")
    groups = group_deployments(scan_atm(tmp_path.rglob("*.atm")))
    assert len(groups) == 2, "same hour on different days is two deployments"


def test_node_subset_and_explicit_window(tmp_path):
    root = _deployment(tmp_path)
    survey = read_atm_deployment(
        root, nodes=["101525", "101528"],
        start="2026-09-18 17:00:00", end="2026-09-18 17:02:00",
    )
    assert survey.n_traces == 2
    assert survey.duration == pytest.approx(120.0)


def test_sample_rate_is_a_parameter(one_minute):
    """A firmware change must not silently rescale every derived velocity."""
    path, _ = one_minute
    assert read_atm(path, sample_rate=500.0).duration == pytest.approx(30.0)
