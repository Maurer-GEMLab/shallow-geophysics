"""GNSS positioning: readers, base shifts, and occupation matching."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shallowgeo.core import SpatialRef, local_grid
from shallowgeo.positions import (
    BaseStation,
    PositionTable,
    apply_base_shift,
    assumed_base_from,
    attach_positions,
    match_occupations,
    read_positions,
)
from shallowgeo.positions.readers import parse_offset_times

DATA = Path(__file__).parent / "data"
GNSS = DATA / "gnss_emlid_reach.csv"

# The synthetic fixture's true base, and the antenna pole length.
BASE = (-92.0, 38.0, 320.0)
ANTENNA = 2.134


def _read(**kw):
    kw.setdefault("heights_are", "antenna_phase_center")
    kw.setdefault("on_duplicate", "keep")
    return read_positions(GNSS, **kw)


class TestFixturePresent:
    def test_present(self):
        assert GNSS.exists(), (
            f"{GNSS.name} missing; if present locally but not in CI it is "
            "untracked -- check .gitignore."
        )


class TestOffsetTimeParsing:
    """The 'UTC-05:00' suffix, and offsets that change mid-occupation."""

    def test_all_occupations_are_35_seconds_in_utc(self):
        t = _read()
        duration = (t.frame["end"] - t.frame["start"]).dt.total_seconds()
        assert (duration == 35.0).all()

    def test_naive_parsing_would_have_been_wrong(self):
        # Guards the actual trap: 7 rows change offset mid-occupation, and
        # reading the local clock turns 35 s into an hour.
        raw = pd.read_csv(GNSS)
        changed = raw["Averaging start"].str[-6:] != raw["Averaging end"].str[-6:]
        assert changed.sum() == 7
        naive = (
            pd.to_datetime(raw["Averaging end"].str[:21])
            - pd.to_datetime(raw["Averaging start"].str[:21])
        ).dt.total_seconds()
        assert set(naive[changed]) == {3635.0}

    def test_parser_strips_the_literal_utc(self):
        out = parse_offset_times(pd.Series(["2026-05-14 12:07:32.6 UTC-05:00"]))
        assert out.iloc[0] == pd.Timestamp("2026-05-14 17:07:32.6", tz="UTC")

    def test_times_are_timezone_aware_utc(self):
        assert str(_read().frame["start"].dt.tz) == "UTC"


class TestAntennaHeight:
    """The highest-stakes number in a GNSS export."""

    def test_declaration_is_required(self):
        with pytest.raises(TypeError):
            read_positions(GNSS)  # heights_are is keyword-only and required

    def test_invalid_declaration_rejected(self):
        with pytest.raises(ValueError, match="cannot be defaulted"):
            read_positions(GNSS, heights_are="whatever")

    def test_phase_center_heights_are_reduced(self):
        raw = pd.read_csv(GNSS)["Ellipsoidal height"].to_numpy()
        t = _read(heights_are="antenna_phase_center")
        np.testing.assert_allclose(t.frame["height"], raw - ANTENNA)

    def test_ground_mark_heights_pass_through(self):
        raw = pd.read_csv(GNSS)["Ellipsoidal height"].to_numpy()
        t = _read(heights_are="ground_mark")
        np.testing.assert_allclose(t.frame["height"], raw)

    def test_reduction_is_recorded_in_provenance(self):
        t = _read(heights_are="antenna_phase_center")
        assert t.provenance.applied("antenna_height_reduction")
        assert t.provenance[-1].parameters["subtracted_m"] == pytest.approx(ANTENNA)

    def test_ground_mark_records_a_zero_reduction(self):
        t = _read(heights_are="ground_mark")
        step = [s for s in t.provenance
                if s.operation == "antenna_height_reduction"][0]
        assert step.parameters["subtracted_m"] == 0.0

    def test_override_conflicts_with_ground_mark(self):
        with pytest.raises(ValueError, match="nothing to reduce"):
            _read(heights_are="ground_mark", antenna_height=2.0)

    def test_override_wins_over_file_column(self):
        raw = pd.read_csv(GNSS)["Ellipsoidal height"].to_numpy()
        t = _read(antenna_height=1.5)
        np.testing.assert_allclose(t.frame["height"], raw - 1.5)

    def test_table_refuses_unreduced_heights(self):
        frame = pd.DataFrame(
            {"name": ["A"], "longitude": [-92.0], "latitude": [38.0],
             "height": [300.0]}
        )
        with pytest.raises(ValueError, match="ground-mark heights"):
            PositionTable(frame)


class TestColumnHandling:
    def test_all_empty_projected_columns_are_dropped(self):
        t = _read()
        # Easting/Northing/Elevation are present in the header but never filled.
        assert "elevation" not in t.frame.columns
        assert "easting" not in t.frame.columns
        assert {"easting", "northing", "elevation"} <= set(
            t.metadata["dropped_empty_columns"]
        )

    def test_geographic_columns_survive(self):
        assert {"longitude", "latitude", "height"} <= set(_read().frame.columns)

    def test_unknown_profile_rejected(self):
        with pytest.raises(ValueError, match="unknown profile"):
            _read(profile="trimble-something")

    def test_generic_profile_reads_the_same_file(self):
        # Header aliases alone should carry an unfamiliar receiver.
        t = _read(profile="generic-csv")
        assert len(t) == 12

    def test_non_metre_antenna_units_rejected(self, tmp_path):
        raw = pd.read_csv(GNSS)
        raw["Antenna height units"] = "ft"
        p = tmp_path / "feet.csv"
        raw.to_csv(p, index=False)
        with pytest.raises(ValueError, match="not metres"):
            read_positions(p, heights_are="antenna_phase_center",
                           on_duplicate="keep")


class TestDuplicatesAndQuality:
    def test_duplicate_name_at_two_marks_raises_by_default(self):
        with pytest.raises(ValueError, match="two different marks"):
            read_positions(GNSS, heights_are="ground_mark")

    def test_duplicates_reports_separation(self):
        d = _read().duplicates()
        assert len(d) == 1
        assert d.iloc[0]["name"] == "S008"
        assert d.iloc[0]["separation_m"] > 30

    def test_on_duplicate_first_keeps_one(self):
        t = _read(on_duplicate="first")
        assert len(t) == 11
        assert t.provenance.applied("drop_duplicate_names")

    def test_invalid_duplicate_policy(self):
        with pytest.raises(ValueError, match="on_duplicate must be"):
            _read(on_duplicate="whatever")

    def test_require_status_drops_float(self):
        t = _read().require_status("FIX")
        assert len(t) == 10
        assert t.provenance[-1].parameters["dropped"] == 2

    def test_float_row_can_have_deceptively_good_rms(self):
        # The point of gating on status rather than RMS: one FLOAT row reports
        # the same vertical RMS as the FIX rows.
        f = _read().frame
        floats = f[f["status"] == "FLOAT"]
        assert floats["rms_v"].min() == pytest.approx(0.012)
        assert f[f["status"] == "FIX"]["rms_v"].min() == pytest.approx(0.012)

    def test_coincident_marks_found(self):
        # S008 (morning) and S015 (afternoon) are the same physical point.
        c = _read().coincident(tolerance=2.0)
        assert {"S008", "S015"} == set(c.iloc[0][["name_a", "name_b"]])
        assert abs(c.iloc[0]["height_difference_m"]) < 0.2


class TestBaseStation:
    def test_from_coordinates(self):
        b = BaseStation.from_coordinates(*BASE, name="B1")
        assert (b.longitude, b.latitude, b.height) == BASE

    def test_from_single_row_csv(self, tmp_path):
        p = tmp_path / "base.csv"
        p.write_text(
            "Name,Longitude,Latitude,Ellipsoidal height\n"
            "BASE1,-92.00000100,38.00000200,320.150\n"
        )
        b = BaseStation.from_csv(p, heights_are="ground_mark")
        assert b.name == "BASE1"
        assert b.height == pytest.approx(320.150)

    def test_multi_row_csv_rejected(self):
        with pytest.raises(ValueError, match="expected exactly one station"):
            BaseStation.from_csv(GNSS, heights_are="ground_mark")

    def test_base_csv_also_requires_height_declaration(self, tmp_path):
        p = tmp_path / "base.csv"
        p.write_text(
            "Name,Longitude,Latitude,Ellipsoidal height,Antenna height,"
            "Antenna height units\nBASE1,-92.0,38.0,320.0,2.0,m\n"
        )
        reduced = BaseStation.from_csv(p, heights_are="antenna_phase_center")
        assert reduced.height == pytest.approx(318.0)

    def test_assumed_base_comes_from_base_columns_not_a_row_name(self):
        t = _read()
        assumed = assumed_base_from(t)
        assert (assumed.longitude, assumed.latitude) == pytest.approx(BASE[:2])
        # The row *named* "999 - Base" is a rover occupation, ~15 m away.
        named = BaseStation.from_table(t, "999 - Base")
        horizontal, _ = named.separation_from(assumed)
        assert horizontal > 10


class TestBaseShift:
    def test_shift_moves_every_station(self):
        t = _read()
        corrected = BaseStation.from_coordinates(
            BASE[0] + 0.0000100, BASE[1] + 0.0000200, BASE[2] + 0.150
        )
        out = apply_base_shift(t, corrected)
        np.testing.assert_allclose(
            out.frame["height"] - t.frame["height"], 0.150, atol=1e-9
        )
        assert out.provenance.applied("base_shift")
        assert out.provenance[-1].parameters["vertical_m"] == pytest.approx(0.15)

    def test_double_shift_refused(self):
        t = _read()
        c = BaseStation.from_coordinates(BASE[0], BASE[1], BASE[2] + 0.1)
        with pytest.raises(ValueError, match="already been base-shifted"):
            apply_base_shift(apply_base_shift(t, c), c)

    def test_wrong_assumed_base_is_caught(self):
        # Passing the row named "999 - Base" as the assumed base is the classic
        # blunder; the shift is then ~15 m and must not pass silently.
        t = _read()
        wrong = BaseStation.from_table(t, "999 - Base")
        corrected = BaseStation.from_coordinates(*BASE)
        with pytest.raises(ValueError, match="beyond max_shift_m"):
            apply_base_shift(t, corrected, assumed=wrong, max_shift_m=5.0)

    def test_large_shift_allowed_when_explicit(self):
        t = _read()
        wrong = BaseStation.from_table(t, "999 - Base")
        out = apply_base_shift(
            t, BaseStation.from_coordinates(*BASE), assumed=wrong, max_shift_m=100.0
        )
        assert out.provenance.applied("base_shift")

    def test_original_table_not_mutated(self):
        t = _read()
        before = t.frame["height"].copy()
        apply_base_shift(t, BaseStation.from_coordinates(BASE[0], BASE[1], 330.0))
        pd.testing.assert_series_equal(t.frame["height"], before)


class TestExport:
    def test_mean_positions_reports_scatter(self):
        m = _read().mean_positions().set_index("name")
        assert m.loc["S008", "n"] == 2
        assert m.loc["S008", "height_scatter_m"] > 0

    def test_to_coordinates_feeds_a_reader(self):
        coords = _read(on_duplicate="first").to_coordinates()
        assert "S001" in coords and len(coords["S001"]) == 3

    def test_to_geometry_reprojects(self):
        utm = SpatialRef("EPSG:32615", vertical_datum="ellipsoidal")
        g = _read(on_duplicate="first").to_geometry(utm)
        assert g.coords()[:, 0].min() > 1000


class TestOccupationMatching:
    def _survey(self, times, stations):
        from shallowgeo.core import Geometry, PointSurvey

        geom = Geometry(
            ids=stations, x=np.arange(len(stations), dtype=float),
            y=np.zeros(len(stations)), z=np.zeros(len(stations)),
            roles=["station"] * len(stations), spatial_ref=local_grid(-92, 38),
        )
        return PointSurvey(
            pd.DataFrame({
                "station_id": stations,
                "time": pd.to_datetime(times, utc=True),
                "value": np.arange(len(stations), dtype=float) + 3000.0,
            }),
            geom, quantity="gravity", units="mGal",
        )

    def test_readings_matched_to_windows(self):
        t = _read(on_duplicate="first")
        windows = t.frame.set_index("name")
        times = [windows.loc[s, "start"] + pd.Timedelta(seconds=10)
                 for s in ["S001", "S002", "S003"]]
        survey = self._survey(times, ["S001", "S002", "S003"])
        out = match_occupations(survey, t)
        assert list(out["matched_station"]) == ["S001", "S002", "S003"]
        assert out["name_agrees"].all()

    def test_mislabelled_station_is_detected(self):
        # Time says S002, the operator typed S009: exactly the cross-check
        # that matching on time rather than name buys you.
        t = _read(on_duplicate="first")
        w = t.frame.set_index("name")
        survey = self._survey([w.loc["S002", "start"] + pd.Timedelta(seconds=5)],
                              ["S009"])
        out = match_occupations(survey, t)
        assert out["matched_station"].iloc[0] == "S002"
        agrees = out["name_agrees"].iloc[0]
        assert agrees is not pd.NA and not agrees

    def test_reading_outside_every_window_raises(self):
        t = _read(on_duplicate="first")
        survey = self._survey(["2026-05-14 20:00:00"], ["S001"])
        with pytest.raises(ValueError, match="outside every occupation window"):
            match_occupations(survey, t)

    def test_tolerance_admits_a_near_miss(self):
        t = _read(on_duplicate="first")
        w = t.frame.set_index("name")
        late = w.loc["S001", "end"] + pd.Timedelta(seconds=20)
        survey = self._survey([late], ["S001"])
        with pytest.raises(ValueError):
            match_occupations(survey, t)
        out = match_occupations(survey, t, tolerance_s=30)
        assert out["matched_station"].iloc[0] == "S001"

    def test_unmatched_warn_keeps_going(self):
        t = _read(on_duplicate="first")
        survey = self._survey(["2026-05-14 20:00:00"], ["S001"])
        with pytest.warns(RuntimeWarning, match="outside every occupation"):
            out = match_occupations(survey, t, unmatched="warn")
        assert out["matched_station"].iloc[0] is None
        # No match is pd.NA, not False -- a missed reading is not a mislabel.
        assert out["name_agrees"].iloc[0] is pd.NA

    def test_attach_positions_georeferences_a_survey(self):
        t = _read(on_duplicate="first")
        w = t.frame.set_index("name")
        times = [w.loc[s, "start"] + pd.Timedelta(seconds=10)
                 for s in ["S001", "S002"]]
        survey = self._survey(times, ["S001", "S002"])
        out = attach_positions(survey, t)
        assert out.provenance.applied("attach_positions")
        assert out.geometry.stations["z"].iloc[0] == pytest.approx(
            w.loc["S001", "height"]
        )
