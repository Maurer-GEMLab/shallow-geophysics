"""Cross-check shallowgeo.ert against ResIPy's R2, run through wine in a linux/amd64 container.

ResIPy and R2 are used only here; neither is a shallowgeo dependency. See README.md in this
directory for the environment, the container and how to run all jobs (``run_all.sh``).

    python r2_comparison.py <job> [--out DIR] [--pygimli-csv FILE]

R2 jobs (each runs R2.exe through ``dockerwine``, which must be on PATH):
    fwd               forward models (uniform, two-layer, block) on both schemes, two mesh densities
    syn_dd824         invert 3 % noisy synthetic data, full dipole-dipole (824 readings)
    syn_field214      the same on the field line's own 214 readings
    field214          invert the field line, 3 % error, no reweighting
    field214_default  invert the field line with ResIPy's default error model
Then:
    compare           tables, figures and results_r2.json from whatever jobs have finished
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
try:
    import shallowgeo.ert as sge
except ImportError:  # not installed in the ResIPy environment: use this checkout
    sys.path.insert(0, str(REPO / "src"))
    import shallowgeo.ert as sge

FIELD = REPO / "examples" / "data" / "2610234-ert-line" / "raw" / "2610234.dat"
SPACING = 0.9144
NOISE = 0.03
OUT = HERE / "output"  # set from --out

dd = sge.dipole_dipole(42, SPACING, n_max=8, a_max=4)
field = sge.read_res2dinv(FIELD, n_electrodes=42)
SCHEMES = {"dd824": dd, "field214": field}

# Models are evaluated at cell centres, so on shallowgeo's grid the interface and block edges
# move to the nearest cell edges ("effective" geometry). R2 gets mesh nodes at those positions.
mesh0 = sge.make_mesh(dd)
ze, zc = mesh0.z_edges, mesh0.z
H_NOM = 1.5
H_EFF = float(ze[np.argmax(zc > H_NOM)])
xe, xc = mesh0.x_edges, mesh0.x
in_x = np.abs(xc - 20) < 2.5
in_z = (zc > 1.5) & (zc < 4)
BLK_EFF = (float(xe[np.argmax(in_x)]), float(xe[len(in_x) - np.argmax(in_x[::-1])]),
           float(ze[np.argmax(in_z)]), float(ze[len(in_z) - np.argmax(in_z[::-1])]))


def rho_two_layer(X, Z):
    return np.where(Z < H_NOM, 80.0, 400.0)


def rho_block(X, Z):
    r = rho_two_layer(X, Z)
    return np.where((np.abs(X - 20) < 2.5) & (Z > 1.5) & (Z < 4), 20.0, r)


def rho_uniform(X, Z):
    return np.full_like(X, 100.0)


MODELS = {"uniform": rho_uniform, "two_layer": rho_two_layer, "block": rho_block}


def rho_eff(mname, X, Z):
    """The same models with their edges at the effective (shallowgeo-grid) positions, for R2."""
    if mname == "uniform":
        return np.full_like(X, 100.0)
    r = np.where(Z < H_EFF, 80.0, 400.0)
    if mname == "block":
        x0, x1, z0, z1 = BLK_EFF
        r = np.where((X > x0) & (X < x1) & (Z > z0) & (Z < z1), 20.0, r)
    return r


def chi2_log(obs, pred):
    return float(np.mean(((np.log(obs) - np.log(pred)) / NOISE) ** 2))


def rms_pct(obs, pred):
    return float(np.sqrt(np.mean((100 * (obs - pred) / pred) ** 2)))


# ---------------------------------------------------------------------------
# Running R2 through ResIPy
# ---------------------------------------------------------------------------

def resipy_project():
    """ResIPy's Project class, with its macOS wine lookup pointed at the ``dockerwine`` shim."""
    from resipy import Project

    # `resipy.Project` is the class once the package is imported; patch the module itself.
    # ResIPy then runs `which dockerwine` and starts R2 as `dockerwine R2.exe` in the job dir.
    sys.modules["resipy.Project"].whichWineMac = lambda: "dockerwine"
    if shutil.which("dockerwine") is None:
        sys.exit("dockerwine is not on PATH (see README.md)")
    return Project


def make_mesh(k, depth: float, elemx: int, refine=False):
    kw = {}
    if refine:  # nodes on the interface and block edges (x edges already fall on the node grid)
        zs = [H_EFF, BLK_EFF[2], BLK_EFF[3]]
        kw = {"refine_x": [20.0] * len(zs), "refine_z": [-z for z in zs]}
    # ResIPy's padding is elemx * pad columns; keep its elemx=4 default of 8 at any elemx,
    # or the side boundary sits ~7 m past the outer electrodes and long offsets read low.
    k.createMesh(typ="quad", elemx=elemx, fmd=depth, pad=8 // elemx, **kw)


def set_electrodes(k, data):
    e = data.electrodes
    k.setElec(pd.DataFrame({"x": e[:, 0], "y": 0.0, "z": 0.0}))


def r2_forward(sname, mname, elemx):
    Project = resipy_project()
    data = SCHEMES[sname]
    depth = float(sge.make_mesh(data).z_edges[-1])
    wd = OUT / f"fwd_{sname}_{mname}_e{elemx}"
    if wd.exists():
        shutil.rmtree(wd)
    k = Project(str(wd), typ="R2")
    set_electrodes(k, data)
    make_mesh(k, depth, elemx, refine=True)
    c = k.mesh.elmCentre
    k.mesh.df["res0"] = rho_eff(mname, c[:, 0], -c[:, 2])
    k.sequence = (data.data[["a", "b", "m", "n"]].to_numpy() + 1).astype(int)
    t = time.time()
    k.forward(noise=0.0)
    dt = time.time() - t
    f = pd.read_csv(Path(k.dirname) / "fwd" / "R2_forward.dat", sep=r"\s+", skiprows=1, header=None)
    # column 5 is the transfer resistance; R2's own rho_a column uses the same analytic k
    return f.iloc[:, 5].to_numpy() * data.data["k"].to_numpy(), dt, k.mesh.numel


def job_fwd():
    rows = []
    for sname, data in SCHEMES.items():
        for mname, fn in MODELS.items():
            sg = sge.apparent_resistivity(data, fn, noise=0).data["rhoa"].to_numpy()
            for elemx in (4, 2):
                r2, dt, nel = r2_forward(sname, mname, elemx)
                a = np.abs(100 * (r2 - sg) / sg)
                rows.append({"scheme": sname, "model": mname, "elemx": elemx, "r2_elements": nel,
                             "n": len(sg), "median_abs_pct": float(np.median(a)),
                             "max_abs_pct": float(a.max()), "r2_time_s": dt})
                np.save(OUT / f"fwd_{sname}_{mname}_e{elemx}.npy", np.c_[sg, r2])
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "forward_raw.csv", index=False)


# Log data, 3 % error, no reweighting, target RMS 1. alpha_aniso 4 is the squared equivalent of
# the vertical roughness weight 0.5 that shallowgeo and pyGIMLi used. R2 picks its own alpha.
PARAM = {"a_wgt": 0.0, "b_wgt": NOISE, "error_mod": 0, "max_iter": 20, "tolerance": 1.0,
         "data_type": 1, "alpha_aniso": 4.0}


def r2_invert(name, data, rhoa, param, elemx=2):
    Project = resipy_project()
    depth = float(sge.make_mesh(data).z_edges[-1])
    wd = OUT / name
    if wd.exists():
        shutil.rmtree(wd)
    # ResIPy's RES2DINV reader mangles mid-point dipole-dipole files; give it a, b, m, n.
    pf = OUT / f"{name}_protocol.dat"
    d = data.data
    lines = [f"{i} {a + 1} {b + 1} {m + 1} {n + 1} {ra / kk:.8e}\n"
             for i, (a, b, m, n, ra, kk) in enumerate(zip(d.a, d.b, d.m, d.n, rhoa, d.k), 1)]
    pf.write_text(f"{len(d)}\n" + "".join(lines))
    k = Project(str(wd), typ="R2")
    k.createSurvey(str(pf), ftype="ProtocolDC")
    set_electrodes(k, data)
    make_mesh(k, depth, elemx)
    print(name, "elements", k.mesh.numel, "data", len(d), flush=True)
    t = time.time()
    k.invert(param=dict(param))
    dt = time.time() - t
    inv = Path(k.dirname)
    res = pd.read_csv(inv / "f001_res.dat", sep=r"\s+", header=None).to_numpy()
    err = pd.read_csv(inv / "f001_err.dat", sep=r"\s+", header=0)
    out_txt = (inv / "R2.out").read_text()
    np.save(OUT / f"{name}_res.npy", res[:, :3])
    err.to_csv(OUT / f"{name}_err.csv", index=False)
    shutil.copy(inv / "R2.out", OUT / f"{name}_R2.out")
    shutil.copy(inv / "R2.in", OUT / f"{name}_R2.in")
    s = {"name": name, "elements": k.mesh.numel, "time_s": dt, "err_columns": list(err.columns),
         "r2_final_rms_lines": [ln.strip() for ln in out_txt.splitlines() if "Final RMS" in ln],
         "iterations": out_txt.count("Iteration")}
    (OUT / f"{name}_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1), flush=True)


def job_syn(sname):
    data = SCHEMES[sname]
    syn = sge.apparent_resistivity(data, rho_block, noise=NOISE, seed=1)  # as in the pyGIMLi run
    rhoa = syn.data["rhoa"].to_numpy()
    np.save(OUT / f"syn_{sname}_rhoa.npy", rhoa)
    r2_invert(f"syn_{sname}", data, rhoa, PARAM)


def job_field(default=False):
    rhoa = field.data["rhoa"].to_numpy()
    if default:  # ResIPy's defaults: 0.01 ohm + 2 %, error reweighting, 10 iterations, isotropic
        r2_invert("field214_default", field, rhoa, {})
    else:
        r2_invert("field214", field, rhoa, PARAM)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def r2_predicted(name, data):
    """Observed and R2-predicted rho_a from f001_err.dat (protocol order)."""
    err = pd.read_csv(OUT / f"{name}_err.csv")
    cols = {c.lower(): c for c in err.columns}
    calc = next(cols[c] for c in cols if c.startswith("calc"))
    obs = next(cols[c] for c in cols if c.startswith("obs"))
    # R2 v4 writes apparent resistivities; older versions wrote resistances
    k = 1.0 if "rho" in obs.lower() else data.data["k"].to_numpy()
    return err[obs].to_numpy() * k, err[calc].to_numpy() * k


def sample_r2(name, X, Z):
    """R2's model (element centres, z as elevation) interpolated in log10 onto (X, Z) depths."""
    from scipy.interpolate import griddata

    xyz = np.load(OUT / f"{name}_res.npy")
    v = griddata(np.c_[xyz[:, 0], -xyz[:, 1]], np.log10(xyz[:, 2]), (X, Z), method="linear")
    return 10 ** v


def r2_info(name):
    s = json.loads((OUT / f"{name}_summary.json").read_text())
    return {"r2_elements": s["elements"], "r2_time_min": s["time_s"] / 60,
            "r2_iterations": s["iterations"],
            "r2_final_rms_reported": s["r2_final_rms_lines"][-1] if s["r2_final_rms_lines"] else None}


def compare_forward(R):
    rows = []
    for f in sorted(OUT.glob("fwd_*_e*.npy")):
        _, sname, *m, e = f.stem.split("_")
        sg, r2 = np.load(f).T
        uni = np.load(OUT / f"fwd_{sname}_uniform_{e}.npy")[:, 1] / 100.0  # R2's own k error
        for label, v in (("raw", r2), ("R2 / R2 uniform", r2 / uni)):
            pct = 100 * (v - sg) / sg
            a = np.abs(pct)
            rows.append({"scheme": sname, "model": "_".join(m), "mesh": e, "R2 values": label,
                         "n": len(sg), "median": np.median(a), "p95": np.percentile(a, 95),
                         "max": a.max(), "mean_signed": pct.mean()})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "forward_summary.csv", index=False)
    print(df.round(3).to_string())
    R["forward"] = df.round(3).to_dict("records")


def compare_synthetic(R, axes_rows, pygimli_csv):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    pg = pd.read_csv(pygimli_csv).set_index("scheme") if pygimli_csv else None
    rows = []
    for (sname, data), axes in zip(SCHEMES.items(), axes_rows):
        name = f"syn_{sname}"
        if not (OUT / f"{name}_summary.json").exists():
            print("missing", name)
            continue
        syn = sge.apparent_resistivity(data, rho_block, noise=NOISE, seed=1)
        rhoa = syn.data["rhoa"].to_numpy()
        assert np.allclose(rhoa, np.load(OUT / f"{name}_rhoa.npy"))
        res = sge.resistivity_tomography(syn, lam=20)
        X, Z = res.mesh.mesh()
        true = rho_block(X, Z)
        r2v = sample_r2(name, X, Z)
        ok = res.covered() & np.isfinite(r2v)
        sgv = res.resistivity
        obs, pred = r2_predicted(name, data)
        assert np.allclose(obs, rhoa, rtol=1e-4), "R2 err file not in protocol order"
        blk = ok & (np.abs(X - 20) < 2.5) & (Z > 1.5) & (Z < 4)
        dl = np.abs(np.log10(sgv[ok] / r2v[ok]))
        r = {"scheme": sname, "n": len(rhoa), "sg_chi2": res.chi2, "sg_rms_pct": res.rms,
             "r2_chi2": chi2_log(rhoa, pred), "r2_rms_pct": rms_pct(rhoa, pred), **r2_info(name),
             "covered_cells": int(ok.sum()),
             "median_abs_dlog10_sg_vs_r2": float(np.median(dl)),
             "p90_abs_dlog10_sg_vs_r2": float(np.percentile(dl, 90)),
             "median_abs_dlog10_sg_vs_true": float(np.median(np.abs(np.log10(sgv[ok] / true[ok])))),
             "median_abs_dlog10_r2_vs_true": float(np.median(np.abs(np.log10(r2v[ok] / true[ok])))),
             "block_median_sg": float(np.median(sgv[blk])),
             "block_median_r2": float(np.median(r2v[blk]))}
        if pg is not None and sname in pg.index:
            r["pg_chi2"] = pg.loc[sname, "pg_chi2_log"]
            r["block_median_pg"] = pg.loc[sname, "block_median_rho_pg"]
        rows.append(r)
        kw = {"norm": LogNorm(15, 500), "cmap": "Spectral_r", "shading": "flat"}
        for ax, v, t in zip(axes, (true, np.where(ok, sgv, np.nan), np.where(ok, r2v, np.nan)),
                            ("true", f"shallowgeo: chi² {r['sg_chi2']:.2f}",
                             f"R2 (ResIPy): chi² {r['r2_chi2']:.2f}")):
            im = ax.pcolormesh(res.mesh.x_edges, res.mesh.z_edges, v, **kw)
            ax.set_ylim(res.mesh.z_edges[-1], 0)
            ax.set_title(f"{sname}: {t}", fontsize=9)
            ax.set_aspect("equal")
        plt.colorbar(im, ax=list(axes), label="ρ (Ω·m)", shrink=0.8)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "synthetic_summary.csv", index=False)
    print(df.round(3).T.to_string())
    R["synthetic"] = df.round(4).to_dict("records")


def compare_field(R):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    rhoa = field.data["rhoa"].to_numpy()
    res = sge.resistivity_tomography(field, lam=20)
    rows = [{"code": "shallowgeo lam=20", "chi2": res.chi2, "rms_pct": res.rms}]
    X, Z = res.mesh.mesh()
    ims = [("shallowgeo λ=20", res.resistivity, res.covered())]
    for name in ("field214", "field214_default"):
        if not (OUT / f"{name}_summary.json").exists():
            print("missing", name)
            continue
        _, pred = r2_predicted(name, field)
        rows.append({"code": f"R2 {name}", "chi2": chi2_log(rhoa, pred),
                     "rms_pct": rms_pct(rhoa, pred), **r2_info(name)})
        if name == "field214":
            r2v = sample_r2(name, X, Z)
            ok = res.covered() & np.isfinite(r2v)
            dl = np.abs(np.log10(res.resistivity[ok] / r2v[ok]))
            rows[-1]["median_abs_dlog10_vs_sg"] = float(np.median(dl))
            ims.append(("R2 (ResIPy), 3 % error", r2v, ok))
    fig, axes = plt.subplots(len(ims), 1, figsize=(10, 3 * len(ims)), squeeze=False,
                             layout="constrained")
    for ax, (t, v, ok) in zip(axes[:, 0], ims):
        im = ax.pcolormesh(res.mesh.x_edges, res.mesh.z_edges, np.where(ok, v, np.nan),
                           norm=LogNorm(10, 1e4), cmap="Spectral_r", shading="flat")
        ax.set_ylim(res.mesh.z_edges[-1], 0)
        ax.set_aspect("equal")
        ax.set_title(t, fontsize=9)
        plt.colorbar(im, ax=ax, label="ρ (Ω·m)", shrink=0.8)
    fig.suptitle("Field line 2610234, 214 readings: no code fits it (not a geological image)")
    fig.savefig(OUT / "field_inversion_r2.png", dpi=110)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "field_summary.csv", index=False)
    print(df.round(3).to_string())
    R["field"] = df.round(4).to_dict("records")


def job_compare(pygimli_csv=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R: dict = {}
    compare_forward(R)
    fig, axes = plt.subplots(2, 3, figsize=(16, 5.5), layout="constrained")
    compare_synthetic(R, axes, pygimli_csv)
    fig.savefig(OUT / "synthetic_inversion_r2.png", dpi=110)
    compare_field(R)
    (OUT / "results_r2.json").write_text(json.dumps(R, indent=1, default=float))


JOBS = {"fwd": job_fwd, "syn_dd824": lambda: job_syn("dd824"),
        "syn_field214": lambda: job_syn("field214"), "field214": job_field,
        "field214_default": lambda: job_field(True)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("job", choices=[*JOBS, "compare"])
    ap.add_argument("--out", type=Path, default=OUT, help="output directory (default: ./output)")
    ap.add_argument("--pygimli-csv", type=Path, default=None,
                    help="synthetic_inversion.csv from the pyGIMLi run, to add its columns")
    args = ap.parse_args()
    OUT = args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.job == "compare":
        job_compare(args.pygimli_csv)
    else:
        JOBS[args.job]()
