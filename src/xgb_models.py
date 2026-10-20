"""Pooled (global) XGBoost models fitted at each forecast origin.

Raw components saved per (origin, seed), for test rows tau == origin:
  E   : CO2 log-growth, emissions-history features only           (5)
  D   : CO2 log-growth, emissions + energy features (direct)       (5)
  EN  : fuel-energy log-growth, all features                      (3)
  DR  : log emission-factor drift log(EF[t+h]/EF[tau]), coupled rows only (3)
  q   : 90% quantile of |EF drift| by horizon from training rows   (10,3)
Physics-structured and post-hoc variants are assembled from these in assemble.py.
"""
import argparse
import pickle
import sys
import time
import numpy as np
from xgboost import XGBRegressor

sys.path.insert(0, "src")
from features import coupled_mask, E_MIN, C_MIN  # noqa: E402
from common import XGB_PARAMS  # noqa: E402

EMIS_ONLY_COLS = list(range(0, 30)) + [47, 48]   # sector levels/dynamics, population growth, horizon


def fit_one(X, y, seed, depth, n_est):
    ok = np.isfinite(y)
    m = XGBRegressor(n_estimators=n_est, max_depth=depth, random_state=seed, **XGB_PARAMS)
    m.fit(X[ok], y[ok])
    return m


def run_origin(meta, X, T, T0, seed, depth, n_est):
    tr = np.where(meta.target_year.values <= T0)[0]
    te = np.where(meta.tau.values == T0)[0]
    out = dict(idx=te)
    Xtr, Xte = X[tr], X[te]
    out["E"] = np.column_stack([fit_one(Xtr[:, EMIS_ONLY_COLS], T["g_c"][tr, j], seed, depth, n_est)
                                .predict(Xte[:, EMIS_ONLY_COLS]) for j in range(5)])
    out["D"] = np.column_stack([fit_one(Xtr, T["g_c"][tr, j], seed, depth, n_est).predict(Xte) for j in range(5)])
    out["EN"] = np.column_stack([fit_one(Xtr, T["g_e"][tr, j], seed, depth, n_est).predict(Xte) for j in range(3)])
    # emission-factor drift model (coupled at origin and at target)
    m0 = coupled_mask(T["c0"][tr], T["e0"][tr])
    m1 = (T["e_act"][tr] >= E_MIN) & (T["c_act"][tr][:, :3] >= C_MIN)
    ef0 = T["c0"][tr][:, :3] / np.maximum(T["e0"][tr], 1e-9)
    ef1 = T["c_act"][tr][:, :3] / np.maximum(T["e_act"][tr], 1e-9)
    drift = np.where(m0 & m1, np.log(np.maximum(ef1, 1e-9)) - np.log(np.maximum(ef0, 1e-9)), np.nan)
    out["DR"] = np.column_stack([fit_one(Xtr, drift[:, j], seed, depth, n_est).predict(Xte) for j in range(3)])
    htr = meta.h.values[tr]
    q = np.full((10, 3), np.nan)
    for h in range(1, 11):
        for j in range(3):
            d = np.abs(drift[htr == h, j])
            d = d[np.isfinite(d)]
            if len(d):
                q[h - 1, j] = np.quantile(d, 0.9)
    out["q"] = q
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", required=True)          # e.g. 1998-2002
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--n_est", type=int, default=400)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = map(int, a.origins.split("-"))
    seeds = [int(s) for s in a.seeds.split(",")]
    meta, X, T = pickle.load(open("results/samples.pkl", "rb"))
    res = {}
    t0 = time.time()
    for T0 in range(lo, hi + 1):
        for s in seeds:
            res[(T0, s)] = run_origin(meta, X, T, T0, s, a.depth, a.n_est)
        print(f"origin {T0} done ({time.time() - t0:.0f}s)", flush=True)
    pickle.dump(dict(cfg=vars(a), res=res), open(a.out, "wb"))


if __name__ == "__main__":
    main()
