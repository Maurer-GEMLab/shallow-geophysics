from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import shallowgeo as sg
from shallowgeo.core import SpatialRef
from shallowgeo.drivers import DriverError, identify, read, registry
from shallowgeo.drivers._seg2 import read_seg2
from shallowgeo.drivers._segy import ibm_to_float, looks_like_segy, read_segy
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


class TestGeodeUnitsAndDelay:
    """Against the header layout of real SeisModule Controller exports."""

    def test_identified_by_seismodule_marker(self, geode_file_feet):
        assert identify(geode_file_feet)[0].name == "geode-seg2"

    def test_feet_are_converted_to_metres(self, geode_file_feet):
        s = read(geode_file_feet)
        np.testing.assert_allclose(s.geometry.receivers["x"].to_numpy(),
                                   np.arange(24) * 4.0 * 0.3048)
        assert s.geometry.sources["x"].iloc[0] == pytest.approx(-12.0 * 0.3048)
        assert s.offsets()[0] == pytest.approx(12.0 * 0.3048)

    def test_conversion_is_recorded(self, geode_file_feet):
        s = read(geode_file_feet)
        params = s.provenance[0].parameters
        assert params["header_units"] == "FEET"
        assert params["unit_factor"] == pytest.approx(0.3048)
        assert params["geometry_from"] == "headers"
        assert s.metadata["header_units"] == "FEET"

    def test_units_override(self, geode_file_feet):
        s = read(geode_file_feet, units="meters")
        assert s.geometry.receivers["x"].iloc[1] == pytest.approx(4.0)

    def test_unknown_units_raise(self, geode_file_feet):
        with pytest.raises(ValueError, match="unrecognised length unit"):
            read(geode_file_feet, units="cubits")

    def test_arguments_are_metres_regardless_of_header(self, geode_file_feet):
        s = read(geode_file_feet, spacing=2.0, source_offset=-1.0)
        assert s.geometry.receivers["x"].iloc[1] == pytest.approx(2.0)
        assert s.provenance[0].parameters["geometry_from"] == "arguments"

    def test_mixed_origin_is_labelled(self, geode_file_feet):
        s = read(geode_file_feet, source_offset=-1.0)
        assert s.provenance[0].parameters["geometry_from"] == "mixed"

    def test_delay_lands_on_time_axis(self, geode_file_feet):
        s = read(geode_file_feet)
        assert s.delay == pytest.approx(0.001)
        assert s.times()[0] == pytest.approx(0.001)
        assert (s.trace_map["delay"] == 0.001).all()

    def test_acquisition_metadata(self, geode_file_feet):
        s = read(geode_file_feet)
        assert s.metadata["shot_sequence_number"] == "26"
        assert s.metadata["stack"] == 8
        assert s.metadata["fixed_gain_db"] == 36
        assert s.metadata["descaling_factor"] == pytest.approx(4.2704e-5)
        assert s.metadata["note"]["BASE_INTERVAL"] == "4.00"

    def test_fixture_without_units_defaults_to_metres(self, geode_file):
        s = read(geode_file)
        assert s.provenance[0].parameters["unit_factor"] == 1.0
        assert s.metadata["header_units"] is None


class TestSEGYParser:
    def test_ibm_float_known_values(self):
        raw = np.array([0x41100000, 0xC2760000, 0x00000000, 0x40800000], dtype=np.uint32)
        np.testing.assert_allclose(ibm_to_float(raw), [1.0, -118.0, 0.0, 0.5])

    def test_ibm_round_trip_through_writer(self, segy_file):
        path, data = segy_file
        segy = read_segy(path)
        np.testing.assert_allclose(segy.traces[0].data, data[0], rtol=1e-6)

    def test_headers(self, segy_file):
        segy = read_segy(segy_file[0])
        assert segy.n_traces == 24
        assert segy.binary_header["format_code"] == 1
        assert segy.measurement_unit == "ft"
        assert segy.sample_interval == pytest.approx(0.000125)
        assert segy.traces[0].header["field_record"] == 4001
        assert segy.traces[5].header["group_x"] == 20
        assert "GEOMETRICS SEISMODULE" in segy.text_header

    def test_little_endian_ieee(self, segy_file_ieee_metric):
        path, data = segy_file_ieee_metric
        segy = read_segy(path)
        assert segy.endian == "<"
        assert segy.measurement_unit == "m"
        np.testing.assert_allclose(segy.traces[2].data, data[2], rtol=1e-6)

    def test_sniffer_rejects_junk(self, tmp_path):
        p = tmp_path / "junk.sgy"
        p.write_bytes(b"\x00" * 4000)
        assert not looks_like_segy(p.read_bytes())
        assert identify(p) == []


class TestGeodeSEGY:
    def test_identified(self, segy_file):
        assert identify(segy_file[0])[0].name == "geode-segy"

    def test_geometry_in_metres(self, segy_file):
        s = read(segy_file[0])
        assert s.n_traces == 24 and s.n_samples == 256
        np.testing.assert_allclose(s.geometry.receivers["x"].to_numpy(),
                                   np.arange(24) * 4.0 * 0.3048)
        assert s.geometry.sources["x"].iloc[0] == pytest.approx(112.0 * 0.3048)
        assert s.offsets()[-1] == pytest.approx(20.0 * 0.3048)

    def test_provenance_and_metadata(self, segy_file):
        s = read(segy_file[0])
        params = s.provenance[0].parameters
        assert params["driver"] == "geode-segy"
        assert params["header_units"] == "ft"
        assert params["units_assumed"] is False
        assert s.metadata["format"] == "SEG-Y"
        assert s.metadata["field_record"] == 4001
        assert s.sample_interval == pytest.approx(0.000125)

    def test_start_time_from_trace_header(self, segy_file):
        s = read(segy_file[0])
        assert s.start_time == pd.Timestamp("2026-09-11 13:19:34")

    def test_unbranded_metric_file_is_claimed_by_extension(self, segy_file_ieee_metric):
        path, _ = segy_file_ieee_metric
        assert identify(path)[0].name == "geode-segy"
        s = read(path)
        assert s.geometry.receivers["x"].iloc[1] == pytest.approx(2.0)
        assert s.geometry.sources["x"].iloc[0] == pytest.approx(-4.0)
        assert s.provenance[0].parameters["unit_factor"] == 1.0

    def test_missing_geometry_needs_spacing(self, segy_file_no_geometry):
        with pytest.raises(ValueError, match="Pass spacing"):
            read(segy_file_no_geometry)
        s = read(segy_file_no_geometry, spacing=1.0, source_offset=-2.0)
        assert s.geometry.receivers["x"].iloc[5] == pytest.approx(5.0)

    def test_registry_lists_five_drivers(self):
        names = {d.name for d in registry.all()}
        assert {"geode-seg2", "geode-segy", "atom-seg2", "cg5", "g857"} <= names


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


class TestGeodeSEGYMultiRecord:
    """A SEG-Y holding several field records is one spread with several shots.

    Concatenating the traces instead would attach every trace to the first
    record's source, which silently corrupts every offset in the second
    record and doubles the apparent number of geophones.
    """

    def test_records_become_sources_not_extra_receivers(self, segy_file_two_records):
        s = read(segy_file_two_records[0])
        assert s.n_traces == 48
        assert len(s.geometry.receivers) == 24
        assert s.geometry.sources["id"].tolist() == ["S1", "S2"]
        assert s.metadata["field_records"] == [4017, 4018]

    def test_second_record_offsets_use_its_own_source(self, segy_file_two_records):
        s = read(segy_file_two_records[0])
        off = s.offsets()
        # 112 ft and 92 ft sources, geophones from 0 ft at 4 ft spacing.
        assert off[0] == pytest.approx(112 * 0.3048)
        assert off[24] == pytest.approx(92 * 0.3048)

    def test_gather_recovers_one_record(self, segy_file_two_records):
        path, data = segy_file_two_records
        s = read(path)
        g = s.gather("S2")
        assert g.n_traces == 24
        np.testing.assert_allclose(g.data, data[24:], rtol=1e-6)
        assert g.offsets()[0] == pytest.approx(92 * 0.3048)

    def test_trace_map_labels_the_record(self, segy_file_two_records):
        s = read(segy_file_two_records[0])
        assert s.trace_map["record"].unique().tolist() == [4017, 4018]

    def test_source_offset_override_is_refused(self, segy_file_two_records):
        with pytest.raises(ValueError, match="field records"):
            read(segy_file_two_records[0], source_offset=0.0)

    def test_mismatched_spreads_are_refused(self, tmp_path):
        from tests.segy_writer import write_segy

        a, b = tmp_path / "_a.sgy", tmp_path / "_b.sgy"
        write_segy(a, np.zeros((6, 32)), 0.0005,
                   group_x=[2.0 * i for i in range(6)], source_x=10.0, field_record=1)
        write_segy(b, np.zeros((6, 32)), 0.0005,
                   group_x=[9.0 * i for i in range(6)], source_x=10.0, field_record=2)
        path = tmp_path / "mixed.sgy"
        path.write_bytes(a.read_bytes() + b.read_bytes()[3600:])
        with pytest.raises(ValueError, match="different receiver spread"):
            read(path)
