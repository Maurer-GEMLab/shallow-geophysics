import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from seg2_writer import write_seg2  # noqa: E402


@pytest.fixture
def geode_file(tmp_path):
    """A 24-channel Geode shot record with geometry written in the headers."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(24, 512)).astype("f4")
    path = tmp_path / "LINE1.DAT"
    write_seg2(
        path,
        data,
        sample_interval=0.000125,
        file_header={
            "ACQUISITION_DATE": "12/MAR/2026",
            "ACQUISITION_TIME": "10:15:00",
            "INSTRUMENT": "GEOMETRICS GEODE",
            "NOTE": "GEODE 24CH",
        },
        trace_headers=[
            {
                "RECEIVER_LOCATION": f"{i * 2.0:.2f}",
                "SOURCE_LOCATION": "-1.00",
                "DELAY": "0.0",
            }
            for i in range(24)
        ],
    )
    return path


@pytest.fixture
def geode_file_no_geometry(tmp_path):
    """The common teaching-lab case: geometry never entered, all zeros."""
    data = np.zeros((24, 256), dtype="f4")
    path = tmp_path / "NOGEOM.DAT"
    write_seg2(
        path,
        data,
        sample_interval=0.00025,
        file_header={"INSTRUMENT": "GEOMETRICS GEODE"},
        trace_headers=[
            {"RECEIVER_LOCATION": "0.00", "SOURCE_LOCATION": "0.00"} for _ in range(24)
        ],
    )
    return path


def _atom_node(tmp_path, name, lat, lon, elev, start_time, n=2000):
    rng = np.random.default_rng(abs(hash(name)) % 2**31)
    path = tmp_path / f"{name}.DAT"
    write_seg2(
        path,
        rng.normal(size=(1, n)).astype("f4"),
        sample_interval=0.004,
        file_header={
            "INSTRUMENT": "GEOMETRICS ATOM-1C",
            "ACQUISITION_DATE": "12/MAR/2026",
            "ACQUISITION_TIME": start_time,
            "UNIT_NUMBER": name,
        },
        trace_headers=[{"RECEIVER_GPS": f"{lat} {lon} {elev}"}],
    )
    return path


@pytest.fixture
def atom_node(tmp_path):
    return _atom_node(tmp_path, "N001", 37.9514, -91.7724, 350.0, "10:00:00")


@pytest.fixture
def atom_array(tmp_path):
    """Three nodes with staggered start times, to exercise alignment."""
    return [
        _atom_node(tmp_path, "N001", 37.9514, -91.7724, 350.0, "10:00:00"),
        _atom_node(tmp_path, "N002", 37.9515, -91.7724, 351.0, "10:00:01"),
        _atom_node(tmp_path, "N003", 37.9516, -91.7724, 352.0, "10:00:02"),
    ]


@pytest.fixture
def cg5_file(tmp_path):
    path = tmp_path / "GRAV.TXT"
    path.write_text(
        "/\t\tCG-5 SURVEY\n"
        "/\t\tSurvey name:\tROLLA_TEST\n"
        "/\t\tInstrument S/N:\t40234\n"
        "/\t\tOperator:\tJM\n"
        "/\t\tCG-5 OPTIONS\n"
        "/\t\tTide Correction: YES\n"
        "/\t\tCont. Tilt: YES\n"
        "/\t\tTerrain Corr.: NO\n"
        "/-------------------------------------------------------------\n"
        "/Line\tStation\tAlt.\tGrav.\tSD.\tTilt x\tTilt y\tTemp.\tTide\tDur\tRej\tTime\tDecTimeDate\tTerrain\tDate\n"
        "1.0\t100.0\t350.0\t3123.456\t0.012\t-1.2\t0.5\t0.01\t0.023\t60\t5\t10:01:02\t46000.4174\t0.000\t2026/03/12\n"
        "1.0\t101.0\t352.0\t3123.501\t0.010\t-1.1\t0.6\t0.01\t0.024\t60\t4\t10:11:02\t46000.4243\t0.000\t2026/03/12\n"
        "1.0\t102.0\t355.0\t3123.602\t0.015\t-1.3\t0.4\t0.02\t0.025\t60\t6\t10:21:02\t46000.4313\t0.000\t2026/03/12\n"
        "1.0\t100.0\t350.0\t3123.470\t0.011\t-1.2\t0.5\t0.01\t0.026\t60\t5\t10:31:02\t46000.4382\t0.000\t2026/03/12\n",
        encoding="latin-1",
    )
    return path


@pytest.fixture
def g857_file(tmp_path):
    path = tmp_path / "MAG.ASC"
    path.write_text(
        "G-857 MEMORY-MAG DATA DUMP\n"
        "100 52341.2 09:00:00 1\n"
        "101 52339.8 09:02:00 1\n"
        "102 52344.1 09:04:00 1\n"
        "100 52340.5 09:30:00 1\n",
        encoding="latin-1",
    )
    return path
