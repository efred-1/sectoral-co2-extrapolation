"""Load predictions from all model families and compute errors."""
import glob
import pickle
import numpy as np
import pandas as pd

from features import C_OFF, E_OFF, level_from_growth
from assemble import physics_variants

MODEL_ORDER = ["naive", "trend", "damped", "damped-H", "damped-P",
               "XGB-E", "XGB-D", "XGB-P", "XGB-H",
               "NN-U", "NN-S", "NN-P", "NN-H",
               "COMB", "COMB-H", "COMB-P"]
SENSITIVITY = ["XGB-H-band", "XGB-H-drift", "NN-H-band"]


def _variants_on(gc, ge, T, n, keys=("hard", "posthoc")):
    """Apply stoichiometric variants to rows where both CO2 and energy forecasts exist."""
    rows = np.where(np.isfinite(gc[:, 0]) & np.isfinite(ge[:, 0]))[0]
    pv = physics_variants(gc[rows], ge[rows], T["c0"][rows], T["e0"][rows])
    out = {}
    for k in keys:
        a, b = np.full((n, 5), np.nan), np.full((n, 3), np.nan)
        a[rows], b[rows] = pv[k]
        out[k] = (a, b)
    return out


def load_samples():
    return pickle.load(open("results/samples.pkl", "rb"))


def _stack(n, results, key_gc, key_ge=None):
    """Average over seeds; also return per-seed arrays. results: {(T0,seed): dict}."""
    seeds = sorted({s for (_, s) in results})
    per_seed = {}
    for s in seeds:
        gc = np.full((n, 5), np.nan)
        ge = np.full((n, 3), np.nan)
        for (T0, ss), r in results.items():
            if ss != s:
                continue
            gc[r["idx"], :r[key_gc].shape[1]] = r[key_gc]
            if key_ge is not None:
                ge[r["idx"]] = r[key_ge]
        per_seed[s] = (gc, ge)
    return per_seed


def mean_over_seeds(per_seed):
    gcs = np.stack([v[0] for v in per_seed.values()])
    ges = np.stack([v[1] for v in per_seed.values()])
    with np.errstate(all="ignore"):
        return np.nanmean(gcs, axis=0), np.nanmean(ges, axis=0)


def build_predictions(meta, X, T, xgb_files, nn_files, baseline_file="results/pred_baselines.pkl"):
    """Return {model: {seed: (gc, ge)}} growth predictions aligned to meta rows."""
    n = len(meta)
    P = {}
    base = pickle.load(open(baseline_file, "rb"))
    for m, (gc, ge) in base.items():
        P[m] = {0: (gc, ge)}
    v = _variants_on(*base["damped"], T, n)
    P["damped-H"] = {0: v["hard"]}
    P["damped-P"] = {0: v["posthoc"]}

    xres = {}
    for f in xgb_files:
        xres.update(pickle.load(open(f, "rb"))["res"])
    if xres:
        e = _stack(n, xres, "E")
        d = _stack(n, xres, "D", "EN")
        dr = _stack(n, xres, "DR")
        q_by_origin = {T0: r["q"] for (T0, s), r in xres.items()}
        P["XGB-E"] = {s: (v[0], np.full((n, 3), np.nan)) for s, v in e.items()}
        P["XGB-D"] = d
        for name in ["XGB-P", "XGB-H", "XGB-H-band", "XGB-H-drift"]:
            P[name] = {}
        for s, (gc, ge) in d.items():
            rows = np.where(np.isfinite(gc[:, 0]))[0]
            out = {k: (np.full((n, 5), np.nan), np.full((n, 3), np.nan)) for k in
                   ["posthoc", "hard", "hard_band", "hard_drift"]}
            pv = physics_variants(gc[rows], ge[rows], T["c0"][rows], T["e0"][rows])
            for k in ["posthoc", "hard", "hard_band"]:
                out[k][0][rows], out[k][1][rows] = pv[k]
            # learned emission-factor drift, clipped to the training-period 90% quantile for that horizon
            qh = np.stack([q_by_origin[tau][h - 1] for tau, h in
                           zip(meta.tau.values[rows], meta.h.values[rows])])
            d_clip = np.clip(dr[s][0][rows, :3], -qh, qh)
            out["hard_drift"][0][rows], out["hard_drift"][1][rows] = _drift_struct(
                gc[rows], ge[rows], T["c0"][rows], T["e0"][rows], d_clip)
            P["XGB-P"][s] = out["posthoc"]
            P["XGB-H"][s] = out["hard"]
            P["XGB-H-band"][s] = out["hard_band"]
            P["XGB-H-drift"][s] = out["hard_drift"]
        # equal-weight combination of the local damped trend and the global XGB-D model (log-growth space)
        gd, ed = base["damped"]
        P["COMB"], P["COMB-H"], P["COMB-P"] = {}, {}, {}
        for s, (gc, ge) in d.items():
            gcc, gec = (gd + gc) / 2, (ed + ge) / 2
            P["COMB"][s] = (gcc, gec)
            v = _variants_on(gcc, gec, T, n)
            P["COMB-H"][s] = v["hard"]
            P["COMB-P"][s] = v["posthoc"]

    loaded = [pickle.load(open(f, "rb")) for f in nn_files]
    for variant, name in [("U", "NN-U"), ("S", "NN-S"), ("H", "NN-H")]:
        res = {}
        for obj in loaded:
            if obj["cfg"]["variant"] == variant:
                res.update(obj["res"])
        if not res:
            continue
        P[name] = _stack(n, res, "gc", "ge")
        if variant == "U":
            P["NN-P"] = {}
            for s, (gc, ge) in P[name].items():
                rows = np.where(np.isfinite(gc[:, 0]))[0]
                pv = physics_variants(gc[rows], ge[rows], T["c0"][rows], T["e0"][rows])
                a, b = np.full((n, 5), np.nan), np.full((n, 3), np.nan)
                a[rows], b[rows] = pv["posthoc"]
                P["NN-P"][s] = (a, b)
        if variant == "H":
            P["NN-H-band"] = {}
            for s, (gc, ge) in P[name].items():
                rows = np.where(np.isfinite(gc[:, 0]))[0]
                pv = physics_variants(gc[rows], ge[rows], T["c0"][rows], T["e0"][rows])
                a, b = np.full((n, 5), np.nan), np.full((n, 3), np.nan)
                a[rows], b[rows] = pv["hard_band"]
                P["NN-H-band"][s] = (a, b)
    return P


def _drift_struct(gc_direct, ge, c0, e0, d):
    from features import coupled_mask
    cm = coupled_mask(c0, e0)
    ef = np.where(cm, c0[:, :3] / np.maximum(e0, 1e-9), np.nan) * np.exp(d)
    e_hat = level_from_growth(ge, e0, E_OFF)
    gc = gc_direct.copy()
    cs = np.nan_to_num(ef * e_hat)
    gc[:, :3] = np.where(cm, np.log(np.maximum(cs, 0) + C_OFF) - np.log(c0[:, :3] + C_OFF), gc_direct[:, :3])
    return gc, ge.copy()


def errors(gc, T, scales):
    """Absolute errors (Mt) and scaled errors for 5 sectors and total."""
    sc, se, st = scales
    c_hat = level_from_growth(gc, T["c0"], C_OFF)
    act = T["c_act"]
    ae = np.abs(c_hat - act)
    tot_hat = c_hat.sum(1)
    tot_act = act.sum(1)                       # nan if any sector missing
    ae_tot = np.abs(tot_hat - tot_act)
    return dict(ae=ae, se=ae / sc, ae_tot=ae_tot, se_tot=ae_tot / st, c_hat=c_hat)


def energy_errors(ge, T, scales):
    sc, se, st = scales
    e_hat = level_from_growth(ge, T["e0"], E_OFF)
    ae = np.abs(e_hat - T["e_act"])
    return dict(ae=ae, se=ae / se)
