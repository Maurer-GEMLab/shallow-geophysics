from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import shallowgeo as sg
from shallowgeo.core import SpatialRef
from shallowgeo.drivers import DriverError, identify, read, registry
from shallowgeo.drivers._seg2 import read_seg2
from shallowgeo.drivers.atom_seg2 import read_atom_array
from shallowgeo.drivers.g857 import inspect_g857


class TestRegistry:
    def test_all_four_drivers_are_discovered(self):
        names = {d.name for d in registry.all()}
        assert {"geode-seg2", "atom-seg2", "cg5", "g857"} <= names

    def test_unknown_driver_lists_alternatives(self):
        with pytest.raises(DriverError, match="Registered:"):
            registry.get("nope")

    def test_unrecognised_file_raises(self, tmp_path):
        p = tmp_path / "junk.bin"
        p.write_bytes(b"\x00" * 4096)
        with pytest.raises(DriverError, match="no registered driver"):
            read(p)


class TestSEG2Parser:
    def test_round_trips_headers_and_samples(self, geode_file):
        seg2 = read_seg2(geode_file)
        assert seg2.n_traces == 24
        assert seg2.traces[0].data.size == 512
        assert seg2.header["INSTRUMENT"] == "GEOMETRICS GEODE"
        assert seg2.traces[3].header["RECEIVER_LOCATION"] == "6.00"

    def test_rejects_non_seg2(self, tmp_path):
        p = tmp_path / "x.dat"
        p.write_bytes(b"NOTSEG2" + b"\x00" * 100)
        with pytest.raises(ValueError, match="file descriptor"):
            read_seg2(p)


class TestGeode:
    def test_identified(self, geode_file):
        assert identify(geode_file)[0].name == "geode-seg2"

    def test_geometry_from_headers(self, geode_file):
        s = read(geode_file)
        assert s.n_traces == 24
        assert s.sample_interval == pytest.approx(0.000125)
        np.testing.assert_allclose(s.geometry.receivers["x"].to_numpy(),
                                   np.arange(24) * 2.0)
        assert s.geometry.sources["x"].iloc[0] == pytest.approx(-1.0)

    def test_offsets_include_shot_standoff(self, geode_file):
        np.testing.assert_allclose(read(geode_file).offsets(),
                                   np.arange(24) * 2.0 + 1.0)

    def test_missing_geometry_raises_actionable_error(self, geode_file_no_geometry):
        with pytest.raises(ValueError, match="Pass spacing"):
            read(geode_file_no_geometry)

    def test_spacing_argument_supplies_missing_geometry(self, geode_file_no_geometry):
        s = read(geode_file_no_geometry, spacing=1.5, source_offset=-3.0)
        np.testing.assert_allclose(s.geometry.receivers["x"].to_numpy(),
                                   np.arange(24) * 1.5)
        assert s.provenance[0].parameters["geometry_from"] == "arguments"

    def test_arguments_override_header_geometry(self, geode_file):
        s = read(geode_file, spacing=5.0, source_offset=0.0)
        assert s.geometry.receivers["x"].iloc[1] == pytest.approx(5.0)

    def test_shot_elevation_follows_sloping_ground(self, geode_file):
        elev = np.linspace(0, 23, 24)
        s = read(geode_file, spacing=1.0, source_offset=5.0, elevations=elev)
        # Shot at 5 m along a 1 m/m slope should sit on the surface, not at z=0.
        assert s.geometry.sources["z"].iloc[0] == pytest.approx(5.0)

    def test_elevation_length_is_checked(self, geode_file):
        with pytest.raises(ValueError, match="elevations has"):
            read(geode_file, spacing=1.0, source_offset=0.0,
                 elevations=np.zeros(3))


class TestAtom:
    def test_identified_over_geode(self, atom_node):
        assert identify(atom_node)[0].name == "atom-seg2"

    def test_single_node_is_passive_and_positioned(self, atom_node):
        s = read(atom_node)
        assert s.is_passive
        assert s.n_traces == 1
        assert s.geometry.receivers["y"].iloc[0] == pytest.approx(37.9514)
        assert s.geometry.receivers["x"].iloc[0] == pytest.approx(-91.7724)

    def test_array_aligns_on_absolute_time(self, atom_array):
        s = read_atom_array(atom_array)
        assert s.n_traces == 3
        assert s.is_passive
        # Nodes start 1 s apart; the common window loses 2 s at 4 ms sampling.
        assert s.n_samples == 2000 - 500
        assert s.provenance[-1].operation == "read_array"

    def test_array_reprojects_to_requested_crs(self, atom_array):
        utm = SpatialRef("EPSG:32615", vertical_datum="ellipsoidal")
        s = read_atom_array(atom_array, spatial_ref=utm)
        assert s.geometry.coords()[:, 0].min() > 1000

    def test_node_without_gps_raises(self, tmp_path):
        from seg2_writer import write_seg2

        p = tmp_path / "NOGPS.DAT"
        write_seg2(p, np.zeros((1, 100), dtype="f4"), 0.004,
                   file_header={"INSTRUMENT": "GEOMETRICS ATOM-1C"})
        with pytest.raises(ValueError, match="no GPS position"):
            read(p)


class TestCG5:
    def test_identified(self, cg5_file):
        assert identify(cg5_file)[0].name == "cg5"

    def test_readings_and_units(self, cg5_file):
        s = read(cg5_file)
        assert s.quantity == "gravity" and s.units == "mGal"
        assert s.n_readings == 4
        assert s.readings["value"].iloc[0] == pytest.approx(3123.456)
        assert s.readings["sigma"].iloc[0] == pytest.approx(0.012)

    def test_instrument_applied_corrections_are_recorded(self, cg5_file):
        s = read(cg5_file)
        assert s.provenance.applied("tide_correction")
        assert s.provenance.applied("tilt_correction")
        # Terrain Corr. was NO in the fixture and must not be claimed.
        assert not s.provenance.applied("terrain_correction")

    def test_metadata_from_header(self, cg5_file):
        s = read(cg5_file)
        assert s.metadata["serial_number"] == "40234"
        assert s.metadata["survey_name"] == "ROLLA_TEST"

    def test_repeat_occupation_scatter(self, cg5_file):
        means = read(cg5_file).station_means().set_index("station_id")
        assert means.loc[100.0, "n"] == 2

    def test_coordinates_require_spatial_ref(self, cg5_file):
        with pytest.raises(ValueError, match="spatial_ref is required"):
            read(cg5_file, coordinates={100.0: (0, 0, 350)})

    def test_missing_station_coordinate_raises(self, cg5_file):
        utm = SpatialRef("EPSG:32615", vertical_datum="orthometric")
        with pytest.raises(ValueError, match="no coordinates given"):
            read(cg5_file, coordinates={100.0: (0, 0, 350)}, spatial_ref=utm)

    def test_coordinates_are_used(self, cg5_file):
        utm = SpatialRef("EPSG:32615", vertical_datum="orthometric")
        coords = {100.0: (0, 0, 350), 101.0: (10, 0, 352), 102.0: (20, 0, 355)}
        s = read(cg5_file, coordinates=coords, spatial_ref=utm)
        assert s.geometry.stations["x"].tolist() == [0, 10, 20]


DATA = Path(__file__).parent / "data"
EXPORTED = DATA / "g857_exported.asc"
MANUAL = DATA / "g857_manual.txt"


class TestFixtureCorpus:
    """Guard against the corpus being present locally but untracked.

    A `data/` line in .gitignore once matched tests/data/ as well, so these
    files existed on disk and passed locally while CI checked out a repo
    without them. Fail with a message that names the actual cause.
    """

    @pytest.mark.parametrize("sample", [EXPORTED, MANUAL])
    def test_sample_is_present(self, sample):
        assert sample.exists(), (
            f"{sample.name} is missing. If it exists locally but not in CI, "
            "it is untracked -- check .gitignore for a rule matching tests/data/."
        )


class TestG857Exported:
    """Five-column exported layout, against the real sample in tests/data."""

    def test_identified(self):
        assert identify(EXPORTED)[0].name == "g857"

    def test_columns_come_from_header(self):
        s = read(EXPORTED, date="2026-03-12")
        assert s.metadata["column_mapping"] == [
            "line", "station", "time", "field", "quality"
        ]
        assert s.provenance[0].parameters["columns_from"] == "header"
        assert s.metadata["column_mapping_verified"] is True

    def test_values_and_units(self):
        s = read(EXPORTED, date="2026-03-12")
        assert s.quantity == "total_field" and s.units == "nT"
        assert s.n_readings == 5
        assert s.readings["value"].iloc[0] == pytest.approx(52431.2)
        assert s.readings["value"].max() == pytest.approx(52450.1)

    def test_string_station_ids_are_not_coerced(self):
        s = read(EXPORTED, date="2026-03-12")
        assert s.readings["station_id"].iloc[0] == "S001"
        assert set(s.geometry.stations["id"]) == {"S001", "S002", "S003", "S004"}

    def test_line_and_quality_carried_through(self):
        s = read(EXPORTED, date="2026-03-12")
        assert s.readings["line"].iloc[0] == "L001"
        assert s.readings["quality"].iloc[2] == pytest.approx(2.8)

    def test_times_parsed_with_supplied_date(self):
        s = read(EXPORTED, date="2026-03-12")
        first = s.readings["time"].iloc[0]
        assert (first.year, first.month, first.day) == (2026, 3, 12)
        assert first.strftime("%H:%M:%S") == "10:30:15"

    def test_repeat_occupation_detected(self):
        # S001 is read twice, 35 minutes apart: the diurnal control pair.
        s = read(EXPORTED, date="2026-03-12")
        means = s.station_means().set_index("station_id")
        assert means.loc["S001", "n"] == 2

    def test_base_station_tagged(self):
        s = read(EXPORTED, date="2026-03-12", base_station="S001")
        assert s.readings["is_base"].sum() == 2

    def test_min_quality_filters_and_records(self):
        s = read(EXPORTED, date="2026-03-12", min_quality=2.9)
        assert s.n_readings == 4          # only S003, at 2.8, falls below
        assert s.provenance.applied("quality_filter")
        assert s.provenance[-1].parameters["dropped"] == 1

    def test_min_quality_rejecting_everything_raises(self):
        with pytest.raises(ValueError, match="rejected every reading"):
            read(EXPORTED, date="2026-03-12", min_quality=99.0)


class TestG857Manual:
    """Three-column hand-entered layout, uncommented header."""

    def test_identified(self):
        assert identify(MANUAL)[0].name == "g857"

    def test_uncommented_header_is_detected(self):
        s = read(MANUAL, date="2026-03-12")
        assert s.metadata["column_mapping"] == ["station", "time", "field"]
        assert s.provenance[0].parameters["columns_from"] == "header"

    def test_values(self):
        s = read(MANUAL, date="2026-03-12")
        assert s.n_readings == 3
        assert s.readings["value"].iloc[0] == pytest.approx(52431.2)
        assert s.readings["station_id"].iloc[2] == "S003"

    def test_no_quality_column(self):
        s = read(MANUAL, date="2026-03-12")
        assert "quality" not in s.readings
        with pytest.raises(ValueError, match="no quality column"):
            read(MANUAL, date="2026-03-12", min_quality=2.0)


class TestG857LayoutResolution:
    def test_layout_name_accepted(self, tmp_path):
        p = tmp_path / "bare.txt"
        p.write_text("S001 10:30:15 52431.2\nS002 10:30:20 52428.5\n")
        s = read(p, driver="g857", columns="manual", date="2026-03-12")
        assert s.provenance[0].parameters["columns_from"] == "layout_name"
        assert s.readings["value"].iloc[0] == pytest.approx(52431.2)

    def test_unknown_layout_name_raises(self, tmp_path):
        p = tmp_path / "bare.txt"
        p.write_text("S001 10:30:15 52431.2\n")
        with pytest.raises(ValueError, match="unknown layout"):
            read(p, driver="g857", columns="nope")

    def test_headerless_file_falls_back_to_column_count(self, tmp_path):
        p = tmp_path / "bare.txt"
        p.write_text("S001 10:30:15 52431.2\nS002 10:30:20 52428.5\n")
        s = read(p, driver="g857", date="2026-03-12")
        assert s.provenance[0].parameters["columns_from"] == "column_count"
        # Inferred from width alone is the one route we do not vouch for.
        assert s.metadata["column_mapping_verified"] is False

    def test_unrecognised_width_raises_actionable_error(self, tmp_path):
        p = tmp_path / "odd.txt"
        p.write_text("S001 10:30:15 52431.2 3.0\nS002 10:30:20 52428.5 3.0\n")
        with pytest.raises(ValueError, match="layout cannot be inferred"):
            read(p, driver="g857")

    def test_explicit_columns_override_header(self):
        s = read(EXPORTED, date="2026-03-12",
                 columns=["line", "station", "time", "field", "signal"])
        assert s.provenance[0].parameters["columns_from"] == "argument"

    def test_wrong_mapping_is_caught(self):
        with pytest.raises(ValueError, match="probably wrong"):
            read(EXPORTED, date="2026-03-12",
                 columns=["line", "station", "field", "time", "quality"])

    def test_inspect_reports_detected_columns(self):
        frame = inspect_g857(EXPORTED)
        assert list(frame.columns) == [
            "line", "station", "time", "field", "quality"
        ]
        assert len(frame) == 5


class TestTopLevelAPI:
    def test_read_is_exported(self, geode_file):
        assert sg.read(geode_file).n_traces == 24

    def test_diagnostics_runs(self, capsys):
        sg.print_diagnostics()
        out = capsys.readouterr().out
        assert "Registered drivers:" in out
        assert "geode-seg2" in out
