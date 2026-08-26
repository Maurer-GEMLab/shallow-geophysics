import numpy as np
import pandas as pd
import pytest

from shallowgeo.core import (
    Geometry,
    PointSurvey,
    Provenance,
    SeismicSurvey,
    SpatialRef,
    local_grid,
)


class TestSpatialRef:
    def test_rejects_unknown_vertical_datum(self):
        with pytest.raises(ValueError, match="vertical_datum"):
            SpatialRef("EPSG:4326", vertical_datum="msl")

    def test_refuses_to_mix_vertical_datums(self):
        a = SpatialRef("EPSG:4326", vertical_datum="ellipsoidal")
        b = SpatialRef("EPSG:4326", vertical_datum="orthometric")
        with pytest.raises(ValueError, match="geoid model"):
            a.assert_vertically_compatible(b)

    def test_unknown_datum_is_permissive(self):
        a = SpatialRef("EPSG:4326", vertical_datum="unknown")
        b = SpatialRef("EPSG:4326", vertical_datum="orthometric")
        a.assert_vertically_compatible(b)

    def test_local_grid_is_metric(self):
        ref = local_grid(-91.77, 37.95)
        assert ref.is_projected
        assert "metre" in ref.horizontal_units.lower()


class TestGeometry:
    def test_requires_explicit_crs(self):
        with pytest.raises(TypeError, match="no default CRS"):
            Geometry([1], [0], [0], [0], ["receiver"], "EPSG:4326")

    def test_rejects_unknown_role(self):
        with pytest.raises(ValueError, match="unknown role"):
            Geometry([1], [0], [0], [0], ["sensor"], local_grid(0, 0))

    def test_from_line_spacing_and_azimuth(self):
        g = Geometry.from_line(5, 2.0, local_grid(0, 0), azimuth=90.0)
        assert len(g) == 5
        np.testing.assert_allclose(g.coords()[:, 0], [0, 2, 4, 6, 8], atol=1e-9)
        np.testing.assert_allclose(g.coords()[:, 1], 0.0, atol=1e-9)

    def test_reproject_round_trip(self):
        wgs = SpatialRef("EPSG:4326", vertical_datum="ellipsoidal")
        utm = SpatialRef("EPSG:32615", vertical_datum="ellipsoidal")
        g = Geometry([1], [-91.77], [37.95], [350.0], ["station"], wgs)
        back = g.to_crs(utm).to_crs(wgs)
        np.testing.assert_allclose(back.coords(), g.coords(), atol=1e-7)

    def test_reproject_preserves_elevation(self):
        wgs = SpatialRef("EPSG:4326", vertical_datum="ellipsoidal")
        utm = SpatialRef("EPSG:32615", vertical_datum="ellipsoidal")
        g = Geometry([1], [-91.77], [37.95], [350.0], ["station"], wgs)
        assert g.to_crs(utm).coords()[0, 2] == 350.0


class TestProvenance:
    def test_applied_detects_recorded_operation(self):
        p = Provenance()
        assert not p.applied("drift_correction")
        p.record("drift_correction", method="linear")
        assert p.applied("drift_correction")

    def test_copy_is_independent(self):
        p = Provenance()
        p.record("read")
        q = p.copy()
        q.record("filter")
        assert len(p) == 1 and len(q) == 2

    def test_round_trips_through_dict(self):
        p = Provenance()
        p.record("read", path="a.dat", spacing=2.0)
        assert Provenance.from_list(p.to_list())[0].parameters["spacing"] == 2.0


class TestSeismicSurvey:
    def _survey(self, n=4):
        geom = Geometry(
            ids=[*range(1, n + 1), "S1"],
            x=[*(np.arange(n) * 2.0), -1.0],
            y=[0.0] * (n + 1),
            z=[0.0] * (n + 1),
            roles=[*["receiver"] * n, "source"],
            spatial_ref=local_grid(0, 0),
        )
        tmap = pd.DataFrame({"receiver_id": range(1, n + 1), "source_id": "S1"})
        return SeismicSurvey(np.zeros((n, 100)), 0.001, geom, tmap)

    def test_rejects_trace_map_length_mismatch(self):
        geom = Geometry([1], [0], [0], [0], ["receiver"], local_grid(0, 0))
        with pytest.raises(ValueError, match="trace_map has"):
            SeismicSurvey(np.zeros((3, 10)), 0.001, geom,
                          pd.DataFrame({"receiver_id": [1]}))

    def test_rejects_nonpositive_sample_interval(self):
        geom = Geometry([1], [0], [0], [0], ["receiver"], local_grid(0, 0))
        with pytest.raises(ValueError, match="sample_interval"):
            SeismicSurvey(np.zeros((1, 10)), 0.0, geom,
                          pd.DataFrame({"receiver_id": [1]}))

    def test_offsets(self):
        s = self._survey()
        np.testing.assert_allclose(s.offsets(), [1.0, 3.0, 5.0, 7.0])

    def test_derived_timing(self):
        s = self._survey()
        assert s.sample_rate == 1000.0
        assert s.duration == pytest.approx(0.1)

    def test_gather_records_provenance(self):
        g = self._survey().gather("S1")
        assert g.n_traces == 4
        assert g.provenance[-1].operation == "gather"

    def test_gather_unknown_source_raises(self):
        with pytest.raises(KeyError):
            self._survey().gather("S9")

    def test_passive_survey_has_no_offsets(self):
        geom = Geometry([1], [0], [0], [0], ["receiver"], local_grid(0, 0))
        s = SeismicSurvey(np.zeros((1, 10)), 0.004, geom,
                          pd.DataFrame({"receiver_id": [1]}))
        assert s.is_passive
        with pytest.raises(ValueError, match="no sources"):
            s.offsets()


class TestPointSurvey:
    def _survey(self):
        geom = Geometry([100, 101], [0, 10], [0, 0], [350, 352],
                        ["station"] * 2, local_grid(0, 0))
        readings = pd.DataFrame({
            "station_id": [100, 101, 100],
            "time": pd.to_datetime(["2026-03-12 10:00", "2026-03-12 10:10",
                                    "2026-03-12 10:20"]),
            "value": [3123.4, 3123.5, 3123.6],
        })
        return PointSurvey(readings, geom, quantity="gravity", units="mGal")

    def test_requires_core_columns(self):
        geom = Geometry([1], [0], [0], [0], ["station"], local_grid(0, 0))
        with pytest.raises(ValueError, match="missing column"):
            PointSurvey(pd.DataFrame({"value": [1.0]}), geom,
                        quantity="gravity", units="mGal")

    def test_station_means_reports_repeat_scatter(self):
        means = self._survey().station_means().set_index("station_id")
        assert means.loc[100, "n"] == 2
        assert means.loc[100, "value"] == pytest.approx(3123.5)
        assert means.loc[100, "repeat_sigma"] > 0

    def test_with_values_does_not_mutate_input(self):
        s = self._survey()
        out = s.with_values([0.0, 0.0, 0.0], "drift_correction", method="linear")
        assert s.readings["value"].iloc[0] == 3123.4
        assert out.provenance.applied("drift_correction")
