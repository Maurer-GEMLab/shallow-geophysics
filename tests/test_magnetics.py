"""The mag-summary driver, diurnal correction, and the magnetics plots."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from shallowgeo import identify, read
from shallowgeo import magnetics as mag
from shallowgeo.drivers.mag_summary import inspect_mag_summary

HEADER = ("Line_No,Station,Dist_from_S_ft,N_Readings,Mean_nT,StdDev_nT,"
          "Correction_nT,Leveled_Mean_nT\n")
# Made-up values in the column layout of a class hand-in table. The
# correction column is rounded, as a spreadsheet shows it, so leveled is
# off from mean + correction by 0.3 nT on line 1.
ROWS = """\
1,1,0.0,2,50100.0,4.1,12.0,50112.3
1,2,13.0,2,50180.0,2.7,12.0,50192.3
1,3,26.0,2,50410.5,3.5,12.0,50422.8
1,4,39.0,2,50320.0,9.8,12.0,50332.3
3,1,0.0,2,50060.5,546.6,-30.0,50030.5
3,2,13.0,2,50140.5,8.2,-30.0,50110.5
3,3,26.0,2,51380.0,9.9,-30.0,51350.0
3,4,39.0,2,50700.5,434.9,-30.0,50670.5
6,1,0.0,3,50020.0,805.0,5.0,50025.0
"""
EXPORTED = Path(__file__).parent / "data" / "g857_exported.asc"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture
def summary(tmp_path):
    p = tmp_path / "mag_summary.csv"
    p.write_text(HEADER + ROWS)
    return p


@pytest.fixture
def with_base(tmp_path):
    """Synthetic survey with times and base-station re-occupations."""
    p = tmp_path / "synthetic.csv"
    mag.synthetic_table(n_lines=5, n_stations=9).to_csv(p, index=False)
    return p


def _write(tmp_path, text, name="mag.csv"):
    p = tmp_path / name
    p.write_text(text)
    return p


class TestMagSummaryDriver:
    def test_identified_and_not_claimed_by_g857(self, summary):
        assert [d.name for d in identify(summary)] == ["mag-summary"]

    def test_g857_dump_not_claimed(self):
        assert "mag-summary" not in [d.name for d in identify(EXPORTED)]

    def test_columns_and_units_from_labels(self, summary):
        cols = inspect_mag_summary(summary).set_index("label")
        assert cols.loc["Dist_from_S_ft", "name"] == "distance"
        assert cols.loc["Dist_from_S_ft", "unit"] == "ft"
        assert cols.loc["Leveled_Mean_nT", "name"] == "leveled"
        assert cols.loc["StdDev_nT", "name"] == "repeat_sigma"

    def test_station_ids_combine_line_and_station(self, summary):
        s = read(summary)
        # Station numbers restart on each line; the id must not collide.
        assert s.readings["station_id"].tolist()[:5] == [
            "L1-S1", "L1-S2", "L1-S3", "L1-S4", "L3-S1"]

    def test_file_order_kept_without_times(self, summary):
        s = read(summary)
        assert s.readings["time"].isna().all()
        assert s.readings["line"].tolist() == [1, 1, 1, 1, 3, 3, 3, 3, 6]

    def test_feet_converted_to_metres(self, summary):
        s = read(summary)
        assert s.readings["distance"].iloc[1] == pytest.approx(13 * 0.3048)
        assert s.metadata["distance_units_in_file"] == "ft"
        assert s.metadata["distance_from"] == "S"

    def test_line_spacing_positions_lines(self, summary):
        x = read(summary, line_spacing=10).geometry.stations.set_index("id")["x"]
        assert x["L1-S1"] == 0.0
        assert x["L3-S1"] == pytest.approx(20 * 0.3048)
        assert np.isnan(read(summary).geometry.stations["x"]).all()

    def test_leveling_arithmetic_checked(self, summary):
        assert read(summary).metadata["leveling_residual_max"] == pytest.approx(0.3, abs=0.05)

    def test_leveling_mismatch_warns(self, tmp_path):
        p = _write(tmp_path, HEADER + "1,1,0,2,50000,1,10,50010\n1,2,10,2,50000,1,10,50110\n")
        with pytest.warns(UserWarning, match="up to 100.0 nT"):
            read(p)

    def test_unitless_distance_needs_distance_units(self, tmp_path):
        p = _write(tmp_path, "Line,Station,Distance,N,Mean_nT\n1,1,0,2,50000\n1,2,5,2,50020\n")
        with pytest.raises(ValueError, match="distance_units"):
            read(p, driver="mag-summary")
        s = read(p, driver="mag-summary", distance_units="m")
        assert s.readings["distance"].iloc[1] == 5.0

    def test_implausible_field_column_raises(self, tmp_path):
        p = _write(tmp_path, HEADER + "1,1,0,2,12,1,1,13\n1,2,10,2,14,1,1,15\n")
        with pytest.raises(ValueError, match="probably not"):
            read(p, driver="mag-summary")

    def test_repeated_station_needs_base_station(self, tmp_path):
        p = _write(tmp_path, HEADER + "1,1,0,2,50000,1,0,50000\n1,1,0,2,50003,1,0,50003\n")
        with pytest.raises(ValueError, match="base_station="):
            read(p)


class TestDiurnalOnRead:
    def test_file_uses_the_correction_column(self, summary):
        s = read(summary, diurnal="file")
        assert s.readings["value"].iloc[0] == pytest.approx(50112.3)
        assert s.provenance.applied("file_correction")
        assert not s.provenance.applied("diurnal")

    def test_file_is_the_default_without_a_base_station(self, summary):
        assert read(summary).metadata["diurnal"] == "file"

    def test_remove_uses_the_station_means(self, summary):
        s = read(summary, diurnal="remove")
        assert s.readings["value"].iloc[0] == pytest.approx(50100.0)
        assert s.readings["value"].iloc[4] == pytest.approx(50060.5)
        assert s.provenance.applied("file_correction_removed")
        assert not s.provenance.applied("file_correction")
        assert s.readings["correction"].iloc[0] == 12.0  # still there to look at

    def test_remove_recovers_means_from_leveled_minus_correction(self, tmp_path):
        p = _write(tmp_path, "Line,Station,Dist_ft,N,Correction_nT,Leveled_nT\n"
                             "1,1,0,2,12,50112\n1,2,10,2,12,50132\n")
        assert read(p, diurnal="remove").readings["value"].tolist() == [50100, 50120]

    def test_remove_without_means_or_correction_raises(self, tmp_path):
        p = _write(tmp_path, "Line,Station,Dist_ft,N,Leveled_nT\n1,1,0,2,50112\n1,2,10,2,50132\n")
        with pytest.raises(ValueError, match="uncorrected"):
            read(p, diurnal="remove")

    def test_base_needs_a_base_station(self, summary):
        with pytest.raises(ValueError, match="base_station"):
            read(summary, diurnal="base")

    def test_unknown_base_station_raises(self, with_base):
        with pytest.raises(KeyError, match="not in the table"):
            read(with_base, base_station="L9-S9")

    def test_unknown_mode_raises(self, summary):
        with pytest.raises(ValueError, match="'file', 'remove' or 'base'"):
            read(summary, diurnal="yes")

    def test_times_read_from_the_table(self, with_base):
        s = read(with_base, base_station=(1, 1), date="2026-10-12", diurnal="remove")
        assert s.readings["time"].notna().all()
        assert s.readings["time"].iloc[0] == pd.Timestamp("2026-10-12 13:00:00")

    def test_base_station_correction_removes_the_drift(self, with_base):
        s = read(with_base, base_station=(1, 1), date="2026-10-12")
        assert s.metadata["diurnal"] == "base"
        step = s.provenance[-1]
        assert step.operation == "diurnal" and step.parameters["by"] == "time"
        assert step.parameters["n_base"] == 6
        base = s.readings[s.readings["is_base"]]
        # Base readings reduce to the first one, to within reading noise.
        assert np.abs(base["value"] - base["value"].iloc[0]).max() < 5
        assert np.abs(base["raw"] - base["raw"].iloc[0]).max() > 15
        assert np.allclose(s.readings["value"], s.readings["raw"] + s.readings["diurnal"])

    def test_base_station_by_id_or_pair(self, with_base):
        a = read(with_base, base_station="L1-S1", date="2026-10-12")
        b = read(with_base, base_station=(1, 1), date="2026-10-12")
        assert np.allclose(a.readings["value"], b.readings["value"])


class TestDiurnalCorrection:
    def test_g857_base_station_by_time(self):
        s = read(EXPORTED, date="2026-03-12", base_station="S001")
        c = mag.diurnal_correction(s)
        # S001: 52431.2 at 10:30:15, 52433.8 at 11:05:02 -- 2.6 nT of drift.
        assert c.readings["value"].iloc[-1] == pytest.approx(52431.2)
        assert c.readings["raw"].iloc[-1] == pytest.approx(52433.8)
        assert c.provenance[-1].parameters["max_drift_nT"] == pytest.approx(2.6)

    def test_g857_base_station_named_in_the_call(self):
        c = mag.diurnal_correction(read(EXPORTED, date="2026-03-12"), "S001")
        assert c.metadata["base_station"] == "S001"

    def test_without_times_falls_back_to_order_with_warning(self, tmp_path):
        p = _write(tmp_path, HEADER.replace(",Correction_nT,Leveled_Mean_nT", "")
                   + "1,1,0,2,50000,1\n1,2,10,2,50100,1\n1,3,20,2,50200,1\n1,1,0,2,50010,1\n")
        with pytest.warns(UserWarning, match="reading order"):
            s = read(p, base_station="L1-S1")
        assert s.provenance[-1].parameters["by"] == "order"
        # Drift of 10 nT over three steps, interpolated by row.
        assert s.readings["diurnal"].round(3).tolist() == [0, -3.333, -6.667, -10]

    def test_refuses_to_stack_on_the_file_correction(self, summary):
        with pytest.raises(ValueError, match="diurnal='remove'"):
            mag.diurnal_correction(read(summary, diurnal="file"), "L1-S1")

    def test_refuses_to_apply_twice(self, with_base):
        s = read(with_base, base_station=(1, 1), date="2026-10-12")
        with pytest.raises(ValueError, match="already been applied"):
            mag.diurnal_correction(s, (1, 1))

    def test_needs_two_base_readings(self, summary):
        with pytest.raises(ValueError, match="at least two"):
            mag.diurnal_correction(read(summary, diurnal="remove"), "L1-S1")

    def test_no_base_station_at_all(self, summary):
        with pytest.raises(ValueError, match="base_station="):
            mag.diurnal_correction(read(summary, diurnal="remove"))

    def test_plot_diurnal(self, with_base):
        mag.plot_diurnal(read(with_base, base_station=(1, 1), date="2026-10-12"))
        with pytest.raises(ValueError, match="no base-station correction"):
            mag.plot_diurnal(read(with_base, base_station=(1, 1), diurnal="remove"))


class TestMagnetics:
    def test_flag_noisy(self, summary):
        s = read(summary)
        assert mag.flag_noisy(s, 100).sum() == 3
        assert mag.flag_noisy(s, None).sum() == 0

    def test_station_table_in_file_units(self, summary):
        t = mag.station_table(read(summary))
        assert "distance_ft" in t and t["distance_ft"].iloc[1] == 13.0
        assert t["noisy"].sum() == 3

    def test_base_station_is_one_row_in_tables(self, with_base):
        s = read(with_base, base_station=(1, 1), date="2026-10-12")
        t = mag.station_table(s)
        assert t["station_id"].is_unique and len(t) == 5 * 9
        assert t.set_index("station_id").loc["L1-S1", "occupations"] == 6

    def test_remove_regional(self, summary):
        s = read(summary)
        a = mag.remove_regional(s, "median", exclude=mag.flag_noisy(s))
        assert a.quantity == "total_field_anomaly"
        level = a.provenance[-1].parameters["level_nT"]
        assert np.allclose(a.readings["value"] + level, s.readings["value"])
        with pytest.raises(ValueError, match="already been removed"):
            mag.remove_regional(a)
        assert mag.remove_regional(s, 50000.0).readings["value"].iloc[0] == pytest.approx(112.3)

    def test_plots_draw(self, summary):
        s = read(summary, line_spacing=10)
        fig = mag.plot_profiles(s, show_raw=True)
        assert len([ax for ax in fig.axes if ax.get_visible()]) == 3
        mag.plot_stacked_profiles(s)
        mag.plot_map(s, drop_noisy=False)
        mag.plot_map(s, anomaly=False)
        mag.plot_corrections(s)
        mag.plot_repeatability(s)

    def test_plots_draw_with_base_station_repeats(self, with_base):
        s = read(with_base, base_station=(1, 1), line_spacing=10, date="2026-10-12")
        mag.plot_profiles(s, show_raw=True)
        mag.plot_stacked_profiles(s)
        mag.plot_map(s)

    def test_profiles_for_chosen_lines(self, summary):
        fig = mag.plot_profiles(read(summary), lines=[3])
        assert fig.axes[0].get_title(loc="left") == "Line 3"
        with pytest.raises(KeyError):
            mag.plot_profiles(read(summary), lines=[99])

    def test_map_needs_line_spacing(self, summary):
        with pytest.raises(ValueError, match="line_spacing"):
            mag.plot_map(read(summary))
        mag.plot_stacked_profiles(read(summary))  # works on line number

    def test_survey_without_line_columns_is_refused(self):
        with pytest.raises(ValueError, match="distance"):
            mag.plot_profiles(read(EXPORTED, date="2026-03-12"))


class TestSynthetic:
    def test_layout_and_base_occupations(self):
        t = mag.synthetic_table(n_lines=4, n_stations=6)
        assert list(t.columns[:4]) == ["Line_No", "Station", "Dist_from_S_ft", "Time"]
        assert ((t["Line_No"] == 1) & (t["Station"] == 1)).sum() == 5
        assert len(t) == 4 * 6 + 4
