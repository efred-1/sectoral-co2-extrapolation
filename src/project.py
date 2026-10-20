"""Projections 2025-2030 from origin 2024 with split-conformal 90% intervals.

MODEL is one of XGB-D, XGB-P, XGB-H, damped, damped-P, COMB, COMB-P (proposed system).
XGBoost (direct CO2 model + fuel-energy model, 3 seeds) is trained on every sample whose target
year is <= 2024; the damped trend is fitted to each series up to 2024.
Intervals use log residuals of the same model, source and horizon from the test-origin forecasts (2008-2019).
"""
import glob
import sys
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, "src")
from features import future_rows, C_OFF, E_OFF, SECTORS, ECOLS, level_from_growth  # noqa: E402
from common import XGB_PARAMS, TEST_ORIGINS  # noqa: E402
from assemble import physics_variants  # noqa: E402
from evallib import load_samples, build_predictions, mean_over_seeds  # noqa: E402
from baselines import damped_path  # noqa: E402

ORIGIN = 2024
HS = list(range(1, 7))
SRC = ["coal", "oil", "gas", "flaring", "cement"]


def xgb_forecast(X, T, fX):
    gcs, ges = [], []
    for seed in range(3):
        gc = []
        for j in range(5):
            ok = np.isfinite(T["g_c"][:, j])
            m = XGBRegressor(n_estimators=300, max_depth=5, random_state=seed, **XGB_PARAMS)
            gc.append(m.fit(X[ok], T["g_c"][ok, j]).predict(fX))
        ge = [XGBRegressor(n_estimators=300, max_depth=5, random_state=seed, **XGB_PARAMS)
              .fit(X, T["g_e"][:, j]).predict(fX) for j in range(3)]
        gcs.append(np.column_stack(gc)); ges.append(np.column_stack(ge))
        print(f"  xgb seed {seed} done", flush=True)
    return np.mean(gcs, axis=0), np.mean(ges, axis=0)


def damped_forecast(P, fm):
    gc = np.full((len(fm), 5), np.nan)
    ge = np.full((len(fm), 3), np.nan)
    for country, rows in fm.groupby("country"):
        g = P[P.country == country].set_index("year").sort_index()
        idx, hs = rows.index.values, rows.h.values
        for cols, off, out in [(SECTORS, C_OFF, gc), (ECOLS, E_OFF, ge)]:
            for j, s in enumerate(cols):
                vals = g.loc[:ORIGIN, s].values.astype(float)
                k = len(vals)
                while k > 0 and np.isfinite(vals[k - 1]):
                    k -= 1
                ly = np.log(vals[k:] + off)
                out[idx, j] = damped_path(ly, max(hs))[hs - 1] - ly[-1]
    return gc, ge


def main():
    models = sys.argv[1:] or ["COMB-P", "XGB-P", "damped-P"]
    meta, X, T = load_samples()
    P = pd.read_csv("data/panel.csv")
    fm, fX, c0, e0 = future_rows(P, ORIGIN, HS)
    fm = fm.reset_index(drop=True)
    print(f"projection rows: {len(fm)} ({fm.country.nunique()} countries)")
    gx, ex = xgb_forecast(X, T, fX)
    gd, ed = damped_forecast(P, fm)
    Pr = build_predictions(meta, X, T, sorted(glob.glob("results/full/xgb_*.pkl")), [])
    test = np.where(meta.tau.isin(TEST_ORIGINS).values)[0]
    htest = meta.h.values[test]
    for MODEL in models:
        if MODEL.startswith("XGB"):
            gc, ge = gx, ex
        elif MODEL.startswith("damped"):
            gc, ge = gd, ed
        else:
            gc, ge = (gx + gd) / 2, (ex + ed) / 2
        if MODEL.endswith("-P") or MODEL.endswith("-H"):
            gc, ge = physics_variants(gc, ge, c0, e0)["posthoc" if MODEL.endswith("-P") else "hard"]
        c_hat = level_from_growth(gc, c0, C_OFF)
        e_hat = level_from_growth(ge, e0, E_OFF)
        # conformal residuals from the test-period forecasts of the same model
        tgc, _ = mean_over_seeds(Pr[MODEL])
        ch = level_from_growth(tgc[test], T["c0"][test], C_OFF)
        res = np.log(T["c_act"][test] + C_OFF) - np.log(ch + C_OFF)
        res_tot = np.log(T["c_act"][test].sum(1) + C_OFF) - np.log(ch.sum(1) + C_OFF)
        rows = []
        for i in range(len(fm)):
            h = fm.h.values[i]
            for j, s in enumerate(SRC + ["total"]):
                if s == "total":
                    r, mid, base = res_tot[htest == h], c_hat[i].sum(), c0[i].sum()
                else:
                    r, mid, base = res[htest == h, j], c_hat[i, j], c0[i, j]
                r = r[np.isfinite(r)]
                q05, q95 = np.quantile(r, [0.05, 0.95])
                rows.append(dict(country=fm.country.values[i], year=int(fm.target_year.values[i]), h=int(h), source=s,
                                 proj=mid, lo90=max((mid + C_OFF) * np.exp(q05) - C_OFF, 0.0),
                                 hi90=(mid + C_OFF) * np.exp(q95) - C_OFF, base_2024=base))
        out = pd.DataFrame(rows)
        out.to_csv(f"results/projections_{MODEL}.csv", index=False)
        en = pd.DataFrame(e_hat, columns=["coal_twh", "oil_twh", "gas_twh"])
        en.insert(0, "year", fm.target_year.values)
        en.insert(0, "country", fm.country.values)
        en.to_csv(f"results/projections_energy_{MODEL}.csv", index=False)
        panel = out[out.source != "total"].groupby(["year", "source"])[["proj", "base_2024"]].sum().reset_index()
        panel.to_csv(f"results/projections_panel_{MODEL}.csv", index=False)
        print(MODEL)
        print(panel.pivot(index="year", columns="source", values="proj").round(0))


if __name__ == "__main__":
    main()
