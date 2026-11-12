"""ERT: RES2DINV reading and repair, forward accuracy, sensitivities, inversion."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import k0

from shallowgeo import cli
from shallowgeo.ert import (
    ERTData,
    apparent_resistivity,
    dipole_dipole,
    forward,
    geometric_factor,
    lambda_sweep,
    make_mesh,
    median_depth,
    read_res2dinv,
    read_stg,
    resistivity_tomography,
    schlumberger,
    wenner,
)
from shallowgeo.ert.forward import wavenumbers

EXAMPLE = (Path(__file__).resolve().parents[1]
           / "examples/data/2610234-ert-line/raw/2610234.dat")

SUPERSTING_HEADER = "Sting/Swift prg: test\n  0.91440\n3\n{n}\n1\n0\n"
# Dipole-dipole rows as the SuperSting writes them: single-precision noise
# in a and n. Rows 3, 6 and 7 use electrodes past electrode 12.
SUPERSTING_ROWS = """\
  1.37160    0.91440    1.00000    88.39630
  1.82880    0.91440    2.00000   241.99699
  6.40080    1.82880    5.00000  3662.82007
  4.57200    0.91440    8.00001  3191.91992
  5.02920    0.91440    7.00000  3029.26001
 35.20440    6.40079    9.00001 53345.10156
 10.97280    1.82880    8.99999 11907.90039
"""


def _supersting_file(tmp_path, rows=SUPERSTING_ROWS):
    n = len(rows.strip().splitlines())
    p = tmp_path / "sting.dat"
    p.write_text(SUPERSTING_HEADER.format(n=n) + rows + "0\n0\n0\n0\n")
    return p


def _image_series_rhoa(data, rho1, rho2, h):
    """Analytic apparent resistivity over two layers (image series)."""
    kk = (rho2 - rho1) / (rho2 + rho1)

    def pot(r):
        s = 1 / r
        for i in range(1, 3000):
            s = s + 2 * kk ** i / np.sqrt(r ** 2 + (2 * i * h) ** 2)
        return rho1 / (2 * np.pi) * s

    e = data.electrodes[:, 0]
    d = data.data

    def p(i, j):
        out = np.zeros(len(i))
        ok = (i >= 0) & (j >= 0)
        out[ok] = pot(np.abs(e[i[ok]] - e[j[ok]]))
        return out

    a, b, m, n = (d[c].to_numpy() for c in "abmn")
    return (p(a, m) - p(a, n) - p(b, m) + p(b, n)) * d["k"].to_numpy()


class TestGeometry:
    def test_geometric_factors_of_standard_arrays(self):
        a = 2.0
        assert geometric_factor([0], [3 * a], [a], [2 * a])[0] == pytest.approx(2 * np.pi * a)
        for n in (1, 3, 6):  # dipole-dipole, C2 C1 P1 P2
            k = geometric_factor([a], [0], [(n + 1) * a], [(n + 2) * a])[0]
            assert k == pytest.approx(np.pi * a * n * (n + 1) * (n + 2))
        k = geometric_factor([0], [np.nan], [a], [np.nan])[0]  # pole-pole
        assert k == pytest.approx(2 * np.pi * a)

    def test_edwards_median_depths(self):
        # Edwards (1977): Wenner 0.519 a; dipole-dipole n = 1..4.
        assert median_depth([0], [3], [1], [2])[0] == pytest.approx(0.519, abs=2e-3)
        expected = [0.416, 0.697, 0.962, 1.220]
        for n, z in zip(range(1, 5), expected):
            assert median_depth([1], [0], [n + 1], [n + 2])[0] == pytest.approx(z, abs=2e-3)
        assert median_depth([0], [np.nan], [1], [np.nan])[0] == pytest.approx(0.867, abs=2e-3)

    def test_schemes(self):
        dd = dipole_dipole(10, 1.0, n_max=3)
        assert len(dd) == 7 + 6 + 5
        assert (dd.data["k"] > 0).all()
        assert len(wenner(10, 1.0)) == 7 + 4 + 1
        assert (schlumberger(12, 1.0, n_max=2).data["k"] > 0).all()


class TestRes2dinv:
    def test_non_integer_n_is_snapped_and_written_as_integers(self, tmp_path):
        d = read_res2dinv(_supersting_file(tmp_path))
        assert d.array == "dipole-dipole"
        assert d.metadata["snapped_max_m"] < 0.01
        out = d.to_res2dinv(tmp_path / "out.dat")
        lines = out.read_text().splitlines()
        assert int(lines[3]) == len(d)
        n_col = [ln.split()[2] for ln in lines[6:6 + len(d)]]
        assert all(s.isdigit() for s in n_col)
        assert n_col[3] == "8" and n_col[5] == "9"
        back = read_res2dinv(out)
        assert np.allclose(back.data["rhoa"], d.data["rhoa"])
        assert (back.data[list("abmn")].to_numpy() == d.data[list("abmn")].to_numpy()).all()

    def test_apparent_resistivity_is_kept_as_read(self, tmp_path):
        d = read_res2dinv(_supersting_file(tmp_path))
        assert d.data["rhoa"].iloc[0] == pytest.approx(88.3963)
        # x mid-point 1.3716 with a = n = 1: electrodes 0..3
        assert d.data.loc[0, list("abmn")].tolist() == [1, 0, 2, 3]

    def test_trim_drops_readings_past_the_cable(self, tmp_path):
        d = read_res2dinv(_supersting_file(tmp_path))
        t = d.trim(12)
        assert len(t) == 4
        assert t.n_electrodes == 12
        assert t.data[list("abmn")].to_numpy().max() <= 11
        assert t.provenance[-1].operation == "trim_electrodes"
        assert t.provenance[-1].parameters["dropped"] == 3
        # the same through the reader, and the header count follows
        t2 = read_res2dinv(_supersting_file(tmp_path), n_electrodes=12)
        assert len(t2) == 4
        assert int(t2.to_res2dinv(tmp_path / "t.dat").read_text().splitlines()[3]) == 4

    def test_trim_with_offset_first_electrode_renumbers(self, tmp_path):
        d = read_res2dinv(_supersting_file(tmp_path))
        t = d.trim(12, first=1)
        assert len(t) == 1  # only the n = 7 reading starts at electrode 1
        assert t.data.loc[0, list("abmn")].tolist() == [1, 0, 8, 9]
        assert t.electrodes[0, 0] == pytest.approx(0.9144)

    def test_off_grid_electrode_is_an_error(self, tmp_path):
        rows = "  1.37160    0.91440    1.00000    88.0\n  1.50000    0.91440    1.00000    88.0\n"
        p = _supersting_file(tmp_path, rows)
        with pytest.raises(ValueError, match="off the"):
            read_res2dinv(p)

    def test_general_array_round_trip(self, tmp_path):
        d = dipole_dipole(12, 1.0, n_max=3)
        d.data["rhoa"] = np.linspace(10, 50, len(d))
        p = d.to_res2dinv(tmp_path / "g.dat", general=True)
        assert p.read_text().splitlines()[2] == "11"
        back = read_res2dinv(p)
        assert back.array == "general"
        assert np.allclose(back.data["rhoa"], d.data["rhoa"])
        assert np.allclose(back.data["k"], d.data["k"])

    def test_resistances_are_converted(self, tmp_path):
        d = dipole_dipole(8, 1.0, n_max=2)
        txt = d.to_res2dinv(tmp_path / "g.dat", general=True).read_text().splitlines()
        txt[5] = "1"  # measurement type: resistance
        rows = [f"{ln.rsplit(' ', 1)[0]} 2.0" for ln in txt[9:9 + len(d)]]
        (tmp_path / "r.dat").write_text("\n".join(txt[:9] + rows + txt[9 + len(d):]))
        back = read_res2dinv(tmp_path / "r.dat")
        assert np.allclose(back.data["rhoa"], 2.0 * back.data["k"])

    def test_cli_cleans_a_file(self, tmp_path, capsys):
        p = _supersting_file(tmp_path)
        out = tmp_path / "clean.dat"
        assert cli.main(["ert-clean", str(p), "-e", "12", "-o", str(out)]) == 0
        printed = capsys.readouterr().out
        assert "dropped 3" in printed
        # four kept readings, four configurations: (a, n) = (1, 1) fits at 9 positions on 12
        # electrodes and was measured at 1, and so on
        assert "are not in the file" in printed
        assert "dipole 1 x 0.9144 m: n=1 1/9" in printed
        assert len(read_res2dinv(out)) == 4


class TestCoverage:
    def test_a_complete_scheme_has_nothing_missing(self):
        dd = dipole_dipole(15, 1.0, n_max=4, a_max=2)
        cov = dd.coverage()
        assert (cov["missing"] == 0).all()
        assert cov["measured"].sum() == len(dd)
        assert cov[["dipole", "sep"]].drop_duplicates().shape[0] == 8
        assert dd.missing_readings().empty

    def test_removed_readings_are_reported_where_they_were(self):
        dd = dipole_dipole(15, 1.0, n_max=4, a_max=2)
        gone = [3, 4, 20, len(dd) - 1]
        cut = dd.select(~np.isin(np.arange(len(dd)), gone))
        miss = cut.missing_readings()
        want = dd.data.loc[gone, list("abmn")].to_numpy()
        assert sorted(map(tuple, miss[list("abmn")].to_numpy())) == sorted(map(tuple, want))
        assert np.allclose(np.sort(miss["x"]), np.sort(dd.data.loc[gone, "x"]))
        assert cut.coverage()["missing"].sum() == 4
        row = miss.set_index(["a", "b"]).loc[(4, 3)]  # gone[0]: a = 1, n = 1 starting at 3
        assert (row["dipole"], row["sep"]) == (1, 1)

    def test_other_arrays_and_remote_electrodes(self):
        w = wenner(12, 1.0)
        w2 = w.select(w.data.index != 0)
        assert "dipole" not in w2.coverage()
        assert w2.missing_readings()[list("abmn")].values.tolist() == [[0, 3, 1, 2]]
        # pole-dipole: the remote B (-1) stays remote and does not set the shape's width
        pdp = ERTData(np.column_stack([np.arange(8.0), np.zeros(8)]),
                      pd.DataFrame({"a": [0, 2], "b": [-1, -1], "m": [1, 3], "n": [2, 4],
                                    "rhoa": [1.0, 1.0]}), 1.0, "pole-dipole")
        cov = pdp.coverage()
        assert cov.loc[0, "possible"] == 6 and cov.loc[0, "missing"] == 4
        assert (pdp.missing_readings()["b"] == -1).all()


@pytest.mark.skipif(not EXAMPLE.exists(), reason="example dataset not present")
class TestExampleLine:
    def test_raw_file_reaches_past_the_cable(self):
        d = read_res2dinv(EXAMPLE)
        assert len(d) == 323
        assert d.n_electrodes == 136
        assert d.spacing == pytest.approx(0.9144)

    def test_trim_to_42_electrodes(self):
        t = read_res2dinv(EXAMPLE, n_electrodes=42)
        assert len(t) == 214
        assert t.used_electrodes().tolist() == list(range(42))
        assert t.electrodes[-1, 0] == pytest.approx(41 * 0.9144)

    def test_gaps_inside_the_cable_are_in_the_file(self):
        t = read_res2dinv(EXAMPLE, n_electrodes=42)
        cov = t.coverage()
        assert cov[["measured", "possible", "missing"]].sum().tolist() == [214, 466, 252]
        first = cov.iloc[0]
        assert (first["dipole"], first["sep"], first["measured"], first["possible"]) == (1, 1, 15, 39)
        # none of the missing readings is in the raw file under another numbering
        raw = read_res2dinv(EXAMPLE).data[list("abmn")]
        raw_rows = set(map(tuple, raw.to_numpy()))
        assert not raw_rows & set(map(tuple, t.missing_readings()[list("abmn")].to_numpy()))


class TestForward:
    def test_wavenumbers_reproduce_a_point_source(self):
        k, w = wavenumbers(0.5, 80.0)
        r = np.geomspace(0.5, 80.0, 100)
        approx = (2 / np.pi) * (k0(np.outer(r, k)) @ w)
        assert np.abs(approx * r - 1).max() < 1e-3

    def test_uniform_ground_is_exact(self):
        d = dipole_dipole(16, 1.0, n_max=5, a_max=2)
        mesh = make_mesh(d)
        f = forward(mesh, mesh.uniform(123.0), d)
        assert np.allclose(f["rhoa"], 123.0, rtol=1e-10)
        # and the mesh's own geometric factor is within a few percent of k
        assert np.abs(f["k_num"] / d.data["k"] - 1).max() < 0.05

    @pytest.mark.parametrize("rho1, rho2, h", [(50.0, 500.0, 2.0), (500.0, 50.0, 1.5)])
    def test_two_layers_match_the_image_series(self, rho1, rho2, h):
        d = dipole_dipole(24, 1.0, n_max=8, a_max=3)
        mesh = make_mesh(d)
        h = mesh.z_edges[np.argmin(np.abs(mesh.z_edges - h))]
        _, Z = mesh.mesh()
        rhoa = forward(mesh, np.where(Z < h, rho1, rho2), d)["rhoa"]
        exact = _image_series_rhoa(d, rho1, rho2, h)
        assert np.abs(rhoa / exact - 1).max() < 0.015

    def test_jacobian_matches_finite_differences(self):
        d = dipole_dipole(12, 1.0, n_max=4, a_max=2)
        mesh = make_mesh(d)
        rng = np.random.default_rng(3)
        rho = np.exp(np.log(100) + 0.4 * rng.standard_normal(mesh.shape))
        f = forward(mesh, rho, d, jacobian=True)
        assert np.allclose(forward(mesh, mesh.uniform(30.0), d, jacobian=True)["J"].sum(1), 1)
        for c in rng.choice(mesh.n_cells, 4, replace=False):
            r2 = rho.copy().ravel()
            r2[c] *= np.exp(1e-4)
            f2 = forward(mesh, r2.reshape(mesh.shape), d)
            fd = (np.log(f2["rhoa"]) - np.log(f["rhoa"])) / 1e-4
            assert np.allclose(fd, f["J"][:, c], atol=1e-5)

    def test_mesh_rejects_topography(self):
        d = dipole_dipole(8, 1.0, n_max=2)
        d.electrodes[3, 1] = 0.5
        with pytest.raises(NotImplementedError, match="flat"):
            make_mesh(d)


def _block(X, Z):
    rho = np.where(Z < 1.0, 80.0, 300.0)
    return np.where((np.abs(X - 11.5) < 2.0) & (Z > 1.2) & (Z < 3.2), 20.0, rho)


class TestInversion:
    @pytest.fixture(scope="class")
    @classmethod
    def synthetic(cls):
        d = dipole_dipole(24, 1.0, n_max=6, a_max=2)
        mesh = make_mesh(d)
        return apparent_resistivity(d, _block, mesh=mesh, noise=0.03, seed=1), mesh

    def test_recovers_a_conductive_block(self, synthetic):
        d, mesh = synthetic
        res = resistivity_tomography(d, mesh=mesh)
        assert res.chi2 < 1.5
        assert res.history["chi2"].iloc[0] > 10 * res.chi2
        X, Z = mesh.mesh()
        inside = (np.abs(X - 11.5) < 1.0) & (Z > 1.5) & (Z < 2.8)
        beside = (np.abs(X - 4.0) < 1.0) & (Z > 1.5) & (Z < 2.8)
        assert np.median(res.resistivity[inside]) < 0.5 * np.median(res.resistivity[beside])
        top = Z < 0.4
        assert np.median(res.resistivity[top]) == pytest.approx(80, rel=0.25)
        assert {"predicted", "residual", "error"} <= set(res.data.data.columns)
        assert res.covered()[0].all() and not res.covered()[-1].all()

    def test_lambda_trades_fit_for_smoothness(self, synthetic):
        d, mesh = synthetic
        sweep = lambda_sweep(d, lams=(3, 300), mesh=mesh, max_iter=4)
        assert sweep["chi2"].iloc[0] < sweep["chi2"].iloc[1]
        assert sweep["roughness"].iloc[0] > sweep["roughness"].iloc[1]

    def test_drops_non_positive_readings(self):
        d = dipole_dipole(10, 1.0, n_max=3)
        d.data["rhoa"] = 50.0
        d.data.loc[0, "rhoa"] = -1.0
        res = resistivity_tomography(d, max_iter=2)
        assert len(res.data) == len(d) - 1


def test_ertdata_requires_columns():
    with pytest.raises(ValueError, match="missing"):
        ERTData(np.zeros((4, 2)), pd.DataFrame({"a": [0]}), 1.0)


def _stg_line(rec, b, a, m, nn, r, err=3, cur=200.0, spacing=1.0, unit_scale=1.0):
    """One SuperSting .stg record for electrodes at ``index * spacing``."""
    pos = [i * spacing * unit_scale for i in (a, b, m, nn)]  # file order A, B, M, N
    k = float(geometric_factor(*(np.array([[i * spacing, 0.0]]) for i in (a, b, m, nn)))[0])
    xyz = ",".join(f"{x:12.5E}, 0.00000E+00, 0.00000E+00" for x in pos)
    return (f"{rec:4d},USER   ,20261111,12:06:20,{r:12.5E},{err:4d},{cur:.0f},"
            f"{k * r:12.5E},test     ,{xyz},Cmd={rec},HV=258,Cyk=2,MTime=0.8,Gain=1,Ch=1")


def _stg_file(tmp_path, rows, *, unit="meter", records=None, spacing=1.0, unit_scale=1.0):
    head = ("Advanced Geosciences, Inc. SuperSting R8-IP Resistivity meter. S/N: SS0 Type: 3D\n"
            f"Firmware version: 01.33.78E Survey period: 20261111 Records: "
            f"{len(rows) if records is None else records}\nUnit: {unit}\n")
    body = "\n".join(_stg_line(i + 1, *r, spacing=spacing, unit_scale=unit_scale)
                      for i, r in enumerate(rows))
    p = tmp_path / "line.stg"
    p.write_text(head + body + "\n")
    return p


# (b, a, m, n, V/I): dipole-dipole as the SuperSting lays it out, B left of A.
# Row 3 is negative; row 5 reaches electrode 8, past an 8-electrode cable.
STG_ROWS = [(0, 1, 2, 3, 4.0), (0, 1, 3, 4, 1.2), (1, 2, 4, 5, -0.05),
            (1, 2, 6, 7, 0.3), (2, 4, 6, 8, 0.2)]


class TestReadStg:
    def test_reads_positions_resistance_and_extras(self, tmp_path):
        d = read_stg(_stg_file(tmp_path, STG_ROWS))
        assert d.array == "dipole-dipole"
        assert d.n_electrodes == 9 and d.spacing == 1.0
        assert list(d.data["b"]) == [0, 0, 1, 1, 2]
        assert np.allclose(d.data["r"], [r[4] for r in STG_ROWS])
        assert np.allclose(d.data["rhoa"], d.data["k"] * d.data["r"])
        assert d.metadata["nonpositive"] == 1
        assert d.metadata["rhoa_file_max_rel_diff"] < 1e-5
        assert np.allclose(d.data["stack_error"], 0.003)
        assert (d.data["current_ma"] == 200).all()
        assert list(d.data["record"]) == [1, 2, 3, 4, 5]

    def test_trim_and_res2dinv_round_trip(self, tmp_path):
        d = read_stg(_stg_file(tmp_path, STG_ROWS), n_electrodes=8)
        assert len(d) == 4  # row 5 used electrode 8
        out = d.drop_nonpositive().to_res2dinv(tmp_path / "line.dat")
        back = read_res2dinv(out)
        assert len(back) == 3
        assert np.allclose(back.data["rhoa"], d.drop_nonpositive().data["rhoa"], rtol=1e-6)

    def test_numbering_is_absolute(self, tmp_path):
        # Nothing uses electrode 0 or 1: a .dat would renumber from the left-most
        # used position, the .stg keeps the cable's own numbering.
        d = read_stg(_stg_file(tmp_path, [(2, 3, 4, 5, 1.0), (2, 3, 5, 6, 0.5)]))
        assert d.data["b"].iloc[0] == 2 and d.n_electrodes == 7

    def test_feet_are_converted(self, tmp_path):
        m = read_stg(_stg_file(tmp_path, STG_ROWS, spacing=0.9144))
        ft = read_stg(_stg_file(tmp_path, STG_ROWS, unit="feet", spacing=0.9144,
                                unit_scale=1 / 0.3048))
        assert ft.spacing == pytest.approx(0.9144)
        assert np.allclose(ft.data["rhoa"], m.data["rhoa"])

    def test_record_count_mismatch_is_an_error(self, tmp_path):
        with pytest.raises(ValueError, match="header says 9"):
            read_stg(_stg_file(tmp_path, STG_ROWS, records=9))

    def test_cli_cleans_an_stg_file(self, tmp_path, capsys):
        p = _stg_file(tmp_path, STG_ROWS)
        out = tmp_path / "clean.dat"
        assert cli.main(["ert-clean", str(p), "-e", "8", "-o", str(out)]) == 0
        printed = capsys.readouterr().out
        assert "1 with zero or negative" in printed
        assert "dropped 1 kept readings" in printed
        assert len(read_res2dinv(out)) == 3
