"""Cross-check shallowgeo.ert against pyGIMLi.

pyGIMLi is used only here, as an independent reference; it is not a shallowgeo dependency.
See README.md in this directory for the environment.

    python pygimli_comparison.py [part ...] [--out DIR]

Parts (all of them, in this order, if none is given):
    jacobian   finite-difference check of shallowgeo's Jacobian
    forward    forward models (two-layer, block, uniform) on both schemes, and the image series
    synthetic  invert 3 % noisy synthetic data from the block model with both codes
    field      invert the field line with both codes, then the follow-up checks on the fit

Each part writes its CSV and figures to the output directory and merges its numbers into
results.json there.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygimli as pg
import pygimli.meshtools as mt
from matplotlib.colors import LogNorm
from pygimli.physics import ert

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
try:
    import shallowgeo.ert as sge
except ImportError:  # not installed in the pyGIMLi environment: use this checkout
    sys.path.insert(0, str(REPO / "src"))
    import shallowgeo.ert as sge

FIELD = REPO / "examples" / "data" / "2610234-ert-line" / "raw" / "2610234.dat"
SPACING = 0.9144
NOISE = 0.03
DPI = 100
OUT = HERE / "output"  # set from --out
RESULTS: dict = {}
pg.setThreadCount(4)


def stats(pct):
    a = np.abs(np.asarray(pct))
    return {"median_abs_pct": float(np.median(a)), "p95_abs_pct": float(np.percentile(a, 95)),
            "max_abs_pct": float(a.max()), "mean_signed_pct": float(np.mean(pct))}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# Schemes and models
# ---------------------------------------------------------------------------

dd = sge.dipole_dipole(42, SPACING, n_max=8, a_max=4)
field = sge.read_res2dinv(FIELD, n_electrodes=42)
SCHEMES = {"dd824": dd, "field214": field}

# Models are evaluated at cell centres, so on shallowgeo's grid the interface and block edges
# move to the nearest cell edges ("effective" geometry). pyGIMLi and the image series use
# these, so that all three describe the same ground.
mesh0 = sge.make_mesh(dd)
ze, xe = mesh0.z_edges, mesh0.x_edges
zc, xc = mesh0.z, mesh0.x
H_NOM, BLK_NOM = 1.5, (17.5, 22.5, 1.5, 4.0)
H_EFF = float(ze[np.argmax(zc > H_NOM)])  # top edge of the first cell whose centre is below 1.5
in_x = np.abs(xc - 20) < 2.5
in_z = (zc > 1.5) & (zc < 4)
BLK_EFF = (float(xe[np.argmax(in_x)]), float(xe[len(in_x) - np.argmax(in_x[::-1])]),
           float(ze[np.argmax(in_z)]), float(ze[len(in_z) - np.argmax(in_z[::-1])]))


def rho_two_layer(X, Z):
    return np.where(Z < H_NOM, 80.0, 400.0)


def rho_block(X, Z):
    r = rho_two_layer(X, Z)
    return np.where((np.abs(X - 20) < 2.5) & (Z > 1.5) & (Z < 4), 20.0, r)


MODELS = {"two_layer": rho_two_layer, "block": rho_block}


# ---------------------------------------------------------------------------
# Image series, conversion to pyGIMLi, pyGIMLi meshes
# ---------------------------------------------------------------------------

def analytic_two_layer(data, rho1, rho2, h, nterms=400):
    """Two-layer apparent resistivity from the image series."""
    Kr = (rho2 - rho1) / (rho2 + rho1)
    e = data.electrodes[:, 0]
    idx = data.data[list("abmn")].to_numpy()

    def pot(src, rec):
        r = np.abs(e[rec] - e[src])
        s = 1 / r
        for n in range(1, nterms + 1):
            s = s + 2 * Kr ** n / np.sqrt(r ** 2 + (2 * n * h) ** 2)
        return rho1 / (2 * np.pi) * s

    a, b, m, n = idx.T
    v = pot(a, m) - pot(a, n) - pot(b, m) + pot(b, n)
    return data.data["k"].to_numpy() * v


def to_pg(data, rhoa=None):
    """shallowgeo ERTData -> pyGIMLi DataContainerERT, with analytic geometric factors."""
    s = pg.DataContainerERT()
    for x, z in data.electrodes:
        s.createSensor([float(x), -float(z)])
    d = data.data
    s.resize(len(d))
    for c in "abmn":
        s.set(c, pg.Vector(d[c].to_numpy(float)))
    s.markValid(s("a") > -1)
    s.set("k", ert.geometricFactors(s, 3, True))
    if rhoa is not None:
        s.set("rhoa", pg.Vector(np.asarray(rhoa, float)))
    return s


def pg_world(data, *, block=None, h=None, refine_dz=0.1, inner_area=0.05):
    """Flat world padded well beyond the line, electrode nodes plus a refinement node just
    below each, optional layer interface and block (marker 99). Only the inner region around
    the survey carries a cell-size constraint."""
    ex = data.electrodes[:, 0]
    L = float(np.ptp(ex))
    pad = 4 * L
    depth = 5 * L
    X0, X1 = ex.min() - pad, ex.max() + pad
    xi0, xi1, zi = ex.min() - 3, ex.max() + 3, 16.0
    plc = mt.createWorld(start=[X0, 0], end=[X1, -depth], worldMarker=True, marker=1)
    plc += mt.createRectangle(start=[xi0, 0], end=[xi1, -zi], isClosed=True,
                              boundaryMarker=11, marker=2, area=inner_area)
    if h:
        plc += mt.createLine(start=[X0, -h], end=[xi0, -h])
        plc += mt.createLine(start=[xi0, -h], end=[xi1, -h])
        plc += mt.createLine(start=[xi1, -h], end=[X1, -h])
    if block is not None:
        x0, x1, z0, z1 = block
        plc += mt.createRectangle(start=[x0, -z0], end=[x1, -z1], marker=99, area=inner_area,
                                  boundaryMarker=10)
    for x in ex:
        plc.createNode([float(x), 0.0])
        plc.createNode([float(x), -refine_dz * SPACING])
    # explicit region markers and areas for every sub-polygon of the inner box
    xm = 0.5 * (xi0 + xi1) if block is None else xi0 + 0.5
    hh = h or 1.0
    plc.addRegionMarker(pg.Pos(xm, -0.5 * hh), 2, inner_area)
    plc.addRegionMarker(pg.Pos(xm, -zi + 0.5), 2, inner_area)
    for xo in (0.5 * (X0 + xi0), 0.5 * (xi1 + X1)):
        plc.addRegionMarker(pg.Pos(xo, -0.5 * hh), 1, 0.0)
        plc.addRegionMarker(pg.Pos(xo, -depth + 1), 1, 0.0)
    return plc


def pg_forward(data, rho_of, *, block=None, h=None, area=0.05, p2=True):
    """pyGIMLi apparent resistivities (analytic k). ``rho_of(marker, y)`` gives each cell's ρ."""
    plc = pg_world(data, block=block, h=h, inner_area=area)
    mesh = mt.createMesh(plc, quality=34, smooth=[1, 10])
    if p2:
        mesh = mesh.createP2()
    cm = np.array(mesh.cellMarkers())
    cy = np.array([c.center().y() for c in mesh.cells()])
    rho = np.array([rho_of(m, y) for m, y in zip(cm, cy)], float)
    fop = ert.ERTModelling(sr=False)
    fop.setData(to_pg(data))
    fop.setMesh(mesh, ignoreRegionManager=True)
    return np.array(fop.response(pg.Vector(rho))), mesh


def make_rho_of(model, h):
    def f(marker, y):
        if model == "block" and marker == 99:
            return 20.0
        return 80.0 if -y < h else 400.0
    return f


def pg_on_sg_grid(mesh, data, rho, refine=2):
    """pyGIMLi forward (P2) on shallowgeo's own FE grid, each cell split ``refine`` times, with
    the same cell resistivities: a like-for-like forward test without any difference in how
    the model is represented."""
    f = np.linspace(0, 1, refine + 1)[:-1]

    def sub(e):
        return np.append((e[:-1, None] + np.diff(e)[:, None] * f).ravel(), e[-1])

    xn, zn = sub(mesh.xn), sub(mesh.zn)
    g = pg.createGrid(x=xn, y=-zn[::-1])
    for b in g.boundaries():
        if b.outside():
            b.setMarker(-1 if abs(b.center().y()) < 1e-9 else -2)
    cx = np.array([c.center().x() for c in g.cells()])
    cz = -np.array([c.center().y() for c in g.cells()])
    ix = np.clip(np.searchsorted(mesh.x_edges, cx) - 1, 0, mesh.shape[1] - 1)
    iz = np.clip(np.searchsorted(mesh.z_edges, cz) - 1, 0, mesh.shape[0] - 1)
    g = g.createP2()
    fop = ert.ERTModelling(sr=False)
    fop.setData(to_pg(data))
    fop.setMesh(g, ignoreRegionManager=True)
    return np.array(fop.response(pg.Vector(rho[iz, ix])))


# ---------------------------------------------------------------------------
# Inversion helpers
# ---------------------------------------------------------------------------

def chi2_log(obs, pred, err=NOISE):
    r = (np.log(obs) - np.log(pred)) / err
    return float(np.mean(r ** 2))


def rms_pct(obs, pred):
    return float(np.sqrt(np.mean((100 * (obs - pred) / pred) ** 2)))


def pg_invert(data, rhoa, *, lam=20, zWeight=0.5, depth=None, robust=False, maxIter=20):
    s = to_pg(data, rhoa)
    s.set("err", pg.Vector(len(rhoa), NOISE))
    mgr = ert.ERTManager(s, verbose=False)
    t = time.time()
    mgr.invert(lam=lam, zWeight=zWeight, paraDX=0.25, paraMaxCellSize=0.25, paraDepth=depth,
               quality=33.6, maxIter=maxIter, verbose=False, robustData=robust,
               blockyModel=False)
    dt = time.time() - t
    model = np.array(mgr.paraModel())
    resp = np.array(mgr.inv.response)
    out = {"lam": lam, "zWeight": zWeight, "robustData": robust, "time_s": dt,
           "iterations": len(getattr(mgr.inv, "chi2History", [0])) - 1,
           "pg_chi2": float(mgr.inv.chi2()), "pg_relrms_pct": float(mgr.inv.relrms()),
           "chi2_log": chi2_log(rhoa, resp), "rms_pct": rms_pct(rhoa, resp),
           "para_cells": mgr.paraDomain.cellCount()}
    return mgr, model, resp, out


def sample_pg(mgr, model, X, Z):
    """pyGIMLi model at the points (X, Z); NaN outside its parameter domain."""
    dom = mgr.paraDomain
    vals = np.full(X.shape, np.nan)
    for i, (x, z) in enumerate(zip(X.ravel(), Z.ravel())):
        c = dom.findCell(pg.Pos(float(x), -float(z)))
        if c is not None:
            vals.flat[i] = model[c.id()]
    return vals


def draw_pg(ax, mgr, model, mesh, vmin, vmax, title):
    pg.show(mgr.paraDomain, model, ax=ax, cMap="turbo", logScale=True, cMin=vmin, cMax=vmax,
            colorBar=True, label="Ω·m", showMesh=False, coverage=mgr.coverage())
    ax.set_xlim(mesh.x_edges[0], mesh.x_edges[-1])
    ax.set_ylim(-mesh.z_edges[-1], 0)
    ax.set_aspect("equal")
    ax.set_ylabel("elevation (m)")
    ax.set_title(title, fontsize=9)


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------

def part_jacobian():
    log("== Jacobian finite-difference check ==")
    mesh = sge.make_mesh(field)
    rng = np.random.default_rng(3)
    rho = np.exp(np.log(100) + 0.5 * rng.standard_normal(mesh.shape))
    f0 = sge.forward(mesh, rho, field, jacobian=True)
    J = f0["J"]
    cells = rng.choice(mesh.n_cells, 12, replace=False)
    # add a few sensitive near-surface cells
    cells = np.unique(np.concatenate([cells, np.arange(mesh.shape[1] + 20, mesh.shape[1] + 26)]))
    eps = 1e-4
    errs = []
    for c in cells:
        r2 = rho.copy().ravel()
        r2[c] *= np.exp(eps)
        f1 = sge.forward(mesh, r2.reshape(mesh.shape), field)
        fd = (np.log(f1["rhoa"]) - np.log(f0["rhoa"])) / eps
        scale = np.abs(J[:, c]).max()
        if scale > 0:
            errs.append(float(np.abs(fd - J[:, c]).max() / scale))
    RESULTS["jacobian_fd_max_rel_err"] = max(errs)
    log("max relative error", max(errs))


def part_forward():
    log("== forward ==")
    rows, fwd = [], {}
    for sname, data in SCHEMES.items():
        for mname, fn in MODELS.items():
            t = time.time()
            sg = sge.apparent_resistivity(data, fn, noise=0).data["rhoa"].to_numpy()
            t_sg = time.time() - t
            block = BLK_EFF if mname == "block" else None
            t = time.time()
            pgr, pmesh = pg_forward(data, make_rho_of(mname, H_EFF), block=block, h=H_EFF)
            t_pg = time.time() - t
            fwd[(sname, mname)] = (sg, pgr)
            r = {"scheme": sname, "model": mname, "n": len(sg), "pg_cells": pmesh.cellCount(),
                 "t_shallowgeo_s": t_sg, "t_pygimli_s": t_pg,
                 **{f"sg_vs_pg_{k}": v for k, v in stats(100 * (sg - pgr) / pgr).items()}}
            if mname == "two_layer":
                an = analytic_two_layer(data, 80, 400, H_EFF)
                r.update({f"sg_vs_analytic_{k}": v for k, v in stats(100 * (sg - an) / an).items()})
                r.update({f"pg_vs_analytic_{k}": v for k, v in stats(100 * (pgr - an) / an).items()})
                fwd[(sname, "analytic")] = an
            rows.append(r)
            log(sname, mname, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)})
        # uniform ground: pyGIMLi's own accuracy with analytic k (shallowgeo is exact here)
        pgu, _ = pg_forward(data, lambda m, y: 100.0)
        rows.append({"scheme": sname, "model": "uniform100_pg_only", "n": len(pgu),
                     **{f"pg_vs_exact_{k}": v for k, v in stats(pgu - 100.0).items()}})
        log(sname, "uniform, pyGIMLi", stats(pgu - 100))
    # how much pyGIMLi moves with its own mesh (coarse P1 against fine P2), block, dd824
    pg_c, _ = pg_forward(dd, make_rho_of("block", H_EFF), block=BLK_EFF, h=H_EFF,
                         area=0.2, p2=False)
    pg_f = fwd[("dd824", "block")][1]
    rows.append({"scheme": "dd824", "model": "block_pg_coarseP1_vs_fineP2", "n": len(pg_c),
                 **{f"pgcoarse_vs_pgfine_{k}": v
                    for k, v in stats(100 * (pg_c - pg_f) / pg_f).items()}})
    # nominal geometry in pyGIMLi against effective geometry in shallowgeo: what the snapping
    # to cell edges is worth
    pg_nom, _ = pg_forward(dd, make_rho_of("block", H_NOM), block=BLK_NOM, h=H_NOM)
    sg_b = fwd[("dd824", "block")][0]
    rows.append({"scheme": "dd824", "model": "block_sg_vs_pg_nominal_geometry", "n": len(pg_nom),
                 **{f"sg_vs_pgnom_{k}": v for k, v in stats(100 * (sg_b - pg_nom) / pg_nom).items()}})
    pd.DataFrame(rows).to_csv(OUT / "forward_comparison.csv", index=False)
    RESULTS["forward"] = rows

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, key in zip(axes.ravel(), [("dd824", "two_layer"), ("dd824", "block"),
                                      ("field214", "two_layer"), ("field214", "block")]):
        sg, pgr = fwd[key]
        data = SCHEMES[key[0]]
        diff = 100 * (sg - pgr) / pgr
        lim = max(1, np.abs(diff).max())
        sc = ax.scatter(data.data["x"], data.data["depth"], c=diff, cmap="RdBu_r",
                        vmin=-lim, vmax=lim, s=14, marker="s")
        ax.invert_yaxis()
        fig.colorbar(sc, ax=ax, label="100 (sg - pg) / pg  [%]")
        ttl = (f"{key[0]} {key[1]}: median |d| {np.median(np.abs(diff)):.2f} %, "
               f"max {np.abs(diff).max():.2f} %")
        if key[1] == "two_layer":
            an = fwd[(key[0], "analytic")]
            ttl += (f"\nagainst image series: sg median {np.median(np.abs(100 * (sg - an) / an)):.2f} %,"
                    f" pg median {np.median(np.abs(100 * (pgr - an) / an)):.2f} %")
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("pseudo-depth (m)")
    fig.tight_layout()
    fig.savefig(OUT / "forward_pct_diff.png", dpi=DPI)
    plt.close(fig)


def part_synthetic():
    log("== synthetic inversions ==")
    rows = []
    for sname, data in SCHEMES.items():
        syn = sge.apparent_resistivity(data, rho_block, noise=NOISE, seed=1)
        rhoa = syn.data["rhoa"].to_numpy()
        t = time.time()
        res = sge.resistivity_tomography(syn, lam=20)
        t_sg = time.time() - t
        X, Z = res.mesh.mesh()
        true = rho_block(X, Z)
        mgr, model, _, pgo = pg_invert(syn, rhoa, lam=20, zWeight=0.5,
                                       depth=float(res.mesh.z_edges[-1]))
        pgv = sample_pg(mgr, model, X, Z)
        ok = res.covered() & np.isfinite(pgv)
        sgv = res.resistivity
        dlog = np.abs(np.log10(sgv[ok] / pgv[ok]))
        blk = ok & (np.abs(X - 20) < 2.5) & (Z > 1.5) & (Z < 4)
        r = {"scheme": sname, "n": len(rhoa),
             "sg_chi2": res.chi2, "sg_rms_pct": res.rms,
             "sg_iter": int(res.history["iteration"].iloc[-1]), "sg_time_s": t_sg,
             "sg_chi2_check": chi2_log(rhoa, res.data.data["predicted"].to_numpy()),
             **{f"pg_{k}": v for k, v in pgo.items()},
             "covered_cells": int(ok.sum()),
             "median_abs_dlog10_sg_vs_pg": float(np.median(dlog)),
             "p90_abs_dlog10_sg_vs_pg": float(np.percentile(dlog, 90)),
             "median_abs_dlog10_sg_vs_true": float(np.median(np.abs(np.log10(sgv[ok] / true[ok])))),
             "median_abs_dlog10_pg_vs_true": float(np.median(np.abs(np.log10(pgv[ok] / true[ok])))),
             "block_min_rho_sg": float(sgv[blk].min()),
             "block_min_rho_pg": float(np.nanmin(pgv[blk])),
             "block_median_rho_sg": float(np.median(sgv[blk])),
             "block_median_rho_pg": float(np.nanmedian(pgv[blk]))}
        rows.append(r)
        log(sname, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

        fig, axes = plt.subplots(4, 1, figsize=(10, 13))
        res.plot(ax=axes[0], vmin=15, vmax=500, colorbar=True)
        axes[0].set_title(f"shallowgeo, λ = 20: chi² {res.chi2:.2f}, RMS {res.rms:.1f} %",
                          fontsize=9)
        draw_pg(axes[1], mgr, model, res.mesh, 15, 500,
                f"pyGIMLi, λ = 20, zWeight 0.5: chi² {pgo['chi2_log']:.2f}, "
                f"RMS {pgo['rms_pct']:.1f} %")
        m = res.mesh
        im = axes[2].pcolormesh(m.x_edges, m.z_edges, true, cmap="turbo",
                                norm=LogNorm(vmin=15, vmax=500))
        axes[2].set_title("true model (shallowgeo grid)", fontsize=9)
        plt.colorbar(im, ax=axes[2], shrink=0.8)
        dl = np.where(ok, np.log10(sgv / pgv), np.nan)
        im = axes[3].pcolormesh(m.x_edges, m.z_edges, dl, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        axes[3].set_title(f"log10(ρ shallowgeo / ρ pyGIMLi) on covered cells: "
                          f"median |Δ| {r['median_abs_dlog10_sg_vs_pg']:.3f}", fontsize=9)
        plt.colorbar(im, ax=axes[3], shrink=0.8)
        for ax in axes[2:]:
            ax.set_ylim(m.z_edges[-1], 0)
            ax.set_aspect("equal")
            ax.set_ylabel("depth (m)")
        fig.suptitle(f"Synthetic block model, {sname} ({len(rhoa)} readings, 3 % noise)")
        fig.tight_layout()
        fig.savefig(OUT / f"synthetic_inversion_{sname}.png", dpi=DPI)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "synthetic_inversion.csv", index=False)
    RESULTS["synthetic_inversion"] = rows


def part_field():
    log("== field inversion ==")
    rhoa = field.data["rhoa"].to_numpy()
    rows = []
    t = time.time()
    res = sge.resistivity_tomography(field, lam=20)
    rows.append({"code": "shallowgeo", "lam": 20, "chi2_log": res.chi2, "rms_pct": res.rms,
                 "iterations": int(res.history["iteration"].iloc[-1]),
                 "time_s": time.time() - t})
    log(rows[-1])
    res_lo = sge.resistivity_tomography(field, lam=1)
    rows.append({"code": "shallowgeo", "lam": 1, "chi2_log": res_lo.chi2, "rms_pct": res_lo.rms,
                 "iterations": int(res_lo.history["iteration"].iloc[-1])})
    log(rows[-1])
    depth = float(res.mesh.z_edges[-1])
    pgs = {}
    for lam, robust in ((20, False), (1, False), (0.1, False), (20, True)):
        pgs[lam, robust] = pg_invert(field, rhoa, lam=lam, zWeight=0.5, depth=depth,
                                     robust=robust)
        rows.append({"code": "pygimli", **pgs[lam, robust][3]})
        log(rows[-1])
    pd.DataFrame(rows).to_csv(OUT / "field_inversion.csv", index=False)
    RESULTS["field_inversion"] = rows

    # follow-up: is the poor fit the data or the inverter?
    mgr, model, resp, o = pgs[20, False]
    X, Z = res.mesh.mesh()
    pgv = sample_pg(mgr, model, X, Z)
    pgv = np.where(np.isnan(pgv), np.nanmedian(pgv), pgv)
    f = sge.forward(res.mesh, pgv, field)["rhoa"]
    pr = pg_on_sg_grid(res.mesh, field, pgv)
    r2 = sge.resistivity_tomography(field, lam=20, start=pgv, max_iter=10)
    follow = {
        # pyGIMLi's model sampled onto shallowgeo's grid, forward modelled by shallowgeo
        "sg_forward_of_sampled_pg_model": {
            "chi2": chi2_log(rhoa, f), "rms_pct": rms_pct(rhoa, f),
            **{f"vs_pg_response_{k}": v for k, v in stats(100 * (f - resp) / resp).items()}},
        # both forward solvers on the identical grid with that extreme model
        "same_grid_forward_extreme_model": {
            "rho_min": float(pgv.min()), "rho_max": float(pgv.max()),
            **stats(100 * (f - pr) / pr)},
        # shallowgeo started from pyGIMLi's model
        "sg_lam20_started_from_pg_model": {
            "chi2": r2.chi2, "rms_pct": r2.rms,
            "iterations": int(r2.history["iteration"].iloc[-1])},
    }
    for k, v in follow.items():
        log(k, v)
    RESULTS["field_followup"] = follow

    fig, axes = plt.subplots(4, 1, figsize=(10, 13))
    vmin, vmax = np.percentile(rhoa, 2), np.percentile(rhoa, 98)
    res.plot(ax=axes[0], vmin=vmin, vmax=vmax, colorbar=True)
    axes[0].set_title(f"shallowgeo, λ = 20: chi² {res.chi2:.1f}, RMS {res.rms:.1f} %", fontsize=9)
    draw_pg(axes[1], mgr, model, res.mesh, vmin, vmax,
            f"pyGIMLi, λ = 20: chi² {o['chi2_log']:.1f}, RMS {o['rms_pct']:.1f} %")
    m01, mod01, resp01, o01 = pgs[0.1, False]
    draw_pg(axes[2], m01, mod01, res.mesh, vmin, vmax,
            f"pyGIMLi, λ = 0.1: chi² {o01['chi2_log']:.1f}, RMS {o01['rms_pct']:.1f} %")
    ax = axes[3]
    ax.loglog(rhoa, res.data.data["predicted"], ".", label="shallowgeo, λ = 20")
    ax.loglog(rhoa, resp, "x", ms=4, label="pyGIMLi, λ = 20")
    ax.loglog(rhoa, resp01, "+", ms=4, label="pyGIMLi, λ = 0.1")
    lim = [rhoa.min() / 2, rhoa.max() * 2]
    ax.plot(lim, lim, "k-", lw=0.7)
    ax.set_xlabel("observed ρa (Ω·m)")
    ax.set_ylabel("predicted ρa (Ω·m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "field_inversion.png", dpi=DPI)
    plt.close(fig)

    # the data diagnostic in dataset.md: transfer resistance by dipole length and n
    d = field.data.copy()
    d["dip"] = d.a - d.b
    d["sep"] = (d.m - d.a) // d.dip
    tab = d.groupby(["dip", "sep"]).agg(count=("r", "size"), r_median=("r", "median"),
                                        r_min=("r", "min"), r_max=("r", "max"),
                                        k=("k", "median"), rhoa_median=("rhoa", "median"))
    tab.to_csv(OUT / "field_resistance_by_dipole_sep.csv")


PARTS = {"jacobian": part_jacobian, "forward": part_forward, "synthetic": part_synthetic,
         "field": part_field}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parts", nargs="*", metavar="part",
                    help=f"any of {', '.join(PARTS)} (default: all)")
    ap.add_argument("--out", type=Path, default=OUT, help="output directory (default: %(default)s)")
    args = ap.parse_args()
    if bad := set(args.parts) - set(PARTS):
        ap.error(f"unknown part(s): {', '.join(sorted(bad))}")
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    if path.exists():
        RESULTS.update(json.loads(path.read_text()))
    RESULTS.update(pygimli_version=pg.__version__,
                   geometry={"interface_nominal": H_NOM, "interface_effective": H_EFF,
                             "block_nominal_x0x1z0z1": BLK_NOM,
                             "block_effective_x0x1z0z1": BLK_EFF})
    log("effective interface", H_EFF, "block", BLK_EFF)
    t0 = time.time()
    for name in args.parts or list(PARTS):
        PARTS[name]()
        path.write_text(json.dumps(RESULTS, indent=2, default=float))
    log(f"done in {time.time() - t0:.0f} s")
