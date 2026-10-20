"""Test-period evaluation: accuracy by horizon, robust relative error, significance tests,
seed variability, COVID sensitivity, energy accuracy, and split-conformal prediction intervals."""
import glob
import json
import pickle
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, "src")
from evallib import load_samples, build_predictions, mean_over_seeds, errors, energy_errors, MODEL_ORDER  # noqa: E402
from common import TEST_ORIGINS, ALL_ORIGINS  # noqa: E402
from features import C_OFF, level_from_growth  # noqa: E402

SECT = ["coal", "oil", "gas", "flaring", "cement"]
rng = np.random.default_rng(2026)


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    meta, X, T = load_samples()
    scales = pickle.load(open("results/scales.pkl", "rb"))
    P = build_predictions(meta, X, T, sorted(glob.glob("results/full/xgb_*.pkl")),
                          sorted(glob.glob("results/full/nn*.pkl")))
    models = [m for m in MODEL_ORDER if m in P] + [m for m in ["XGB-H-band", "XGB-H-drift", "NN-H-band"] if m in P]
    test = meta.tau.isin(TEST_ORIGINS).values
    out = {}
    E = {}
    for m in models:
        gc, ge = mean_over_seeds(P[m])
        E[m] = errors(gc, T, scales)
        E[m]["energy"] = energy_errors(ge, T, scales)
    # keep only models with forecasts for (almost) all test rows (NN chunks may still be running)
    base_ok = test & np.isfinite(E["naive"]["se"]).any(axis=1)
    keep = [m for m in models if np.isfinite(E[m]["se"][base_ok]).any(axis=1).mean() > 0.99]
    dropped = [m for m in models if m not in keep]
    if dropped:
        print("models without full test coverage (skipped):", dropped)
    models = keep
    # rows where every model has a forecast (common evaluation set)
    ok = test.copy()
    for m in models:
        ok &= np.isfinite(E[m]["se"]).any(axis=1)
    R = np.where(ok)[0]
    H = meta.h.values
    out["n_rows"] = int(len(R))
    out["n_countries"] = int(meta.country.values[R].astype(str).size and meta.iloc[R].country.nunique())

    # ---- accuracy by horizon -------------------------------------------------
    rows = []
    for m in models:
        for h in range(1, 11):
            r = R[H[R] == h]
            se = E[m]["se"][r]
            rows.append(dict(model=m, h=h, n=len(r),
                             mase_sectors=np.nanmean(se), mase_total=np.nanmean(E[m]["se_tot"][r]),
                             mae_total_mt=np.nanmean(E[m]["ae_tot"][r]),
                             **{f"mase_{s}": np.nanmean(se[:, j]) for j, s in enumerate(SECT)}))
    acc = pd.DataFrame(rows)
    acc.to_csv("results/acc_by_horizon.csv", index=False)

    # ---- robust relative error vs naive (AvgRelMAE, Davydenko & Fildes 2013) -----
    key = meta.country.values
    rel = []
    for m in models:
        for h in range(1, 11):
            r = R[H[R] == h]
            vals = []
            for j in range(5):
                df = pd.DataFrame({"c": key[r], "m": E[m]["ae"][r, j], "n": E["naive"]["ae"][r, j]}).dropna()
                g = df.groupby("c")[["m", "n"]].mean()
                g = g[(g.n > 1e-6) & (g.m > 1e-9)]
                vals.extend(np.log(g.m / g.n).values)
            df = pd.DataFrame({"c": key[r], "m": E[m]["ae_tot"][r], "n": E["naive"]["ae_tot"][r]}).dropna()
            g = df.groupby("c")[["m", "n"]].mean()
            g = g[(g.n > 1e-6) & (g.m > 1e-9)]
            rel.append(dict(model=m, h=h, avgrelmae_sectors=float(np.exp(np.mean(vals))),
                            avgrelmae_total=float(np.exp(np.mean(np.log(g.m / g.n))))))
    rel = pd.DataFrame(rel)
    rel.to_csv("results/avgrelmae_by_horizon.csv", index=False)
    # per-source AvgRelMAE by horizon group
    gmap = {"1": [1], "2-3": [2, 3], "4-6": [4, 5, 6], "7-10": [7, 8, 9, 10], "all": list(range(1, 11))}
    rs = []
    for m in models:
        for gname, hs in gmap.items():
            r = R[np.isin(H[R], hs)]
            for j, sname in enumerate(SECT):
                df = pd.DataFrame({"c": key[r], "h": H[r], "m": E[m]["ae"][r, j], "n": E["naive"]["ae"][r, j]}).dropna()
                g = df.groupby(["c", "h"])[["m", "n"]].mean()
                g = g[(g.n > 1e-6) & (g.m > 1e-9)]
                rs.append(dict(model=m, group=gname, source=sname, avgrelmae=float(np.exp(np.mean(np.log(g.m / g.n))))))
    pd.DataFrame(rs).to_csv("results/avgrelmae_by_source.csv", index=False)

    # ---- horizon-group summary table ---------------------------------------
    groups = {"1": [1], "2-3": [2, 3], "4-6": [4, 5, 6], "7-10": [7, 8, 9, 10], "all": list(range(1, 11))}
    summ = []
    for m in models:
        d = dict(model=m)
        for gname, hs in groups.items():
            r = R[np.isin(H[R], hs)]
            d[f"mase_{gname}"] = np.nanmean(E[m]["se"][r])
            d[f"masetot_{gname}"] = np.nanmean(E[m]["se_tot"][r])
            sub = rel[(rel.model == m) & rel.h.isin(hs)]
            d[f"rel_{gname}"] = float(np.exp(np.log(sub.avgrelmae_sectors).mean()))
            d[f"reltot_{gname}"] = float(np.exp(np.log(sub.avgrelmae_total).mean()))
        summ.append(d)
    pd.DataFrame(summ).to_csv("results/summary_table.csv", index=False)

    # ---- seed variability (ML models) ----------------------------------------
    sv = []
    for m in models:
        if len(P[m]) < 2:
            continue
        per = []
        for s, (gc, ge) in P[m].items():
            e = errors(gc, T, scales)
            if np.isfinite(e["se"][R]).any():
                per.append([np.nanmean(e["se"][R]), np.nanmean(e["se_tot"][R])])
        per = np.array(per)
        sv.append(dict(model=m, seeds=len(per), per_seed=";".join(f"{x:.4f}" for x in per[:, 0]),
                       per_seed_tot=";".join(f"{x:.4f}" for x in per[:, 1]),
                       mase_mean=per[:, 0].mean(), mase_sd=per[:, 0].std(ddof=1),
                       masetot_mean=per[:, 1].mean(), masetot_sd=per[:, 1].std(ddof=1)))
    pd.DataFrame(sv).to_csv("results/seed_variability.csv", index=False)

    # ---- COVID sensitivity -----------------------------------------------------
    nc = R[~np.isin(meta.target_year.values[R], [2020, 2021])]
    cov = []
    for m in models:
        cov.append(dict(model=m, mdase_all=np.nanmedian(E[m]["se"][R]), mdase_tot=np.nanmedian(E[m]["se_tot"][R]),
                        n_se_over_50=int(np.nansum(E[m]["se"][R] > 50)),
                        share_err_over_50=float(np.nansum(np.where(E[m]["se"][R] > 50, E[m]["se"][R], 0)) / np.nansum(E[m]["se"][R])),
                        n_se_finite=int(np.isfinite(E[m]["se"][R]).sum()),
                        mase_all=np.nanmean(E[m]["se"][R]), mase_excl_2020_21=np.nanmean(E[m]["se"][nc]),
                        masetot_all=np.nanmean(E[m]["se_tot"][R]), masetot_excl=np.nanmean(E[m]["se_tot"][nc])))
    pd.DataFrame(cov).to_csv("results/covid_sensitivity.csv", index=False)

    # ---- energy accuracy ---------------------------------------------------------
    en = []
    for m in models:
        se = E[m]["energy"]["se"]
        if not np.isfinite(se[R]).any():
            continue
        for gname, hs in groups.items():
            r = R[np.isin(H[R], hs)]
            en.append(dict(model=m, group=gname, mase_energy=np.nanmean(se[r]),
                           **{f"mase_{f}": np.nanmean(se[r, j]) for j, f in enumerate(["coal", "oil", "gas"])}))
    pd.DataFrame(en).to_csv("results/energy_accuracy.csv", index=False)

    # ---- significance: Wilcoxon across countries, Holm-adjusted ---------------
    pairs = [("damped", "naive"), ("XGB-D", "naive"), ("NN-U", "naive"),
             ("XGB-D", "damped"), ("XGB-D", "XGB-E"),
             ("damped-H", "damped"), ("damped-P", "damped"),
             ("XGB-H", "XGB-D"), ("XGB-P", "XGB-D"),
             ("NN-S", "NN-U"), ("NN-P", "NN-U"), ("NN-H", "NN-U"),
             ("COMB", "damped"), ("COMB", "XGB-D"), ("COMB-H", "COMB"), ("COMB-P", "COMB"),
             ("COMB-P", "damped")]
    pairs = [p for p in pairs if p[0] in models and p[1] in models]
    tests = []
    for a, b in pairs:
        for gname, hs in groups.items():
            if gname == "all":
                continue
            r = R[np.isin(H[R], hs)]
            for target in ["sectors", "total"]:
                # (i) robust: country-level mean log ratio of MAE (per source, then averaged over sources)
                if target == "sectors":
                    lr = []
                    for j in range(5):
                        df = pd.DataFrame({"c": key[r], "a": E[a]["ae"][r, j], "b": E[b]["ae"][r, j]}).dropna()
                        g = df.groupby("c")[["a", "b"]].mean()
                        g = g[(g.a > 1e-9) & (g.b > 1e-9)]
                        lr.append(np.log(g.a / g.b).rename(j))
                    dc = pd.concat(lr, axis=1).mean(axis=1)
                else:
                    df = pd.DataFrame({"c": key[r], "a": E[a]["ae_tot"][r], "b": E[b]["ae_tot"][r]}).dropna()
                    g = df.groupby("c")[["a", "b"]].mean()
                    g = g[(g.a > 1e-9) & (g.b > 1e-9)]
                    dc = np.log(g.a / g.b)
                d = dc.values
                nz = np.abs(d) > 1e-12
                p = wilcoxon(d[nz]).pvalue if nz.sum() > 5 else 1.0
                boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                # (ii) pre-specified: country-level mean scaled error
                va = E[a]["se"][r].mean(axis=1) if target == "sectors" else E[a]["se_tot"][r]
                vb = E[b]["se"][r].mean(axis=1) if target == "sectors" else E[b]["se_tot"][r]
                df = pd.DataFrame({"c": key[r], "a": va, "b": vb}).dropna()
                g2 = df.groupby("c")[["a", "b"]].mean()
                d2 = (g2.a - g2.b).values
                p2 = wilcoxon(d2[np.abs(d2) > 1e-12]).pvalue if (np.abs(d2) > 1e-12).sum() > 5 else 1.0
                tests.append(dict(a=a, b=b, horizons=gname, target=target, n_countries=len(d),
                                  rel_change_pct=100 * (np.exp(d.mean()) - 1),
                                  ci_lo=100 * (np.exp(np.quantile(boots, 0.025)) - 1),
                                  ci_hi=100 * (np.exp(np.quantile(boots, 0.975)) - 1),
                                  share_countries_a_better=float((d < 0).mean()), p=p,
                                  mase_diff=d2.mean(), share_mase_a_better=float((d2 < 0).mean()), p_mase=p2))
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests.p.values)
    tests["p_mase_holm"] = holm(tests.p_mase.values)
    tests.to_csv("results/significance_tests.csv", index=False)

    # ---- per-source robust effect of each constrained variant vs its twin (all horizons) ----
    ps = []
    for a, b in [p for p in pairs if p[0].endswith(("-P", "-H", "-S"))]:
        for j, sname in enumerate(SECT):
            df = pd.DataFrame({"c": key[R], "h": H[R], "a": E[a]["ae"][R, j], "b": E[b]["ae"][R, j]}).dropna()
            g = df.groupby(["c", "h"])[["a", "b"]].mean()
            g = g[(g.a > 1e-9) & (g.b > 1e-9)]
            lr = np.log(g.a / g.b).groupby(level=0).mean()
            ps.append(dict(a=a, b=b, source=sname, rel_change_pct=100 * (np.exp(lr.mean()) - 1), share_better=float((lr < 0).mean())))
    pd.DataFrame(ps).to_csv("results/constraint_by_source.csv", index=False)

    # ---- split-conformal intervals (residuals from origins strictly before T0, targets <= T0) ----
    allrows = np.where(meta.tau.isin(ALL_ORIGINS).values)[0]
    tau = meta.tau.values
    ty = meta.target_year.values
    cov_rows = []
    for m in [x for x in ["naive", "damped", "damped-P", "XGB-D", "XGB-P", "NN-U", "NN-P", "COMB", "COMB-P"] if x in models]:
        gc, _ = mean_over_seeds(P[m])
        c_hat = level_from_growth(gc, T["c0"], C_OFF)
        res = np.log(T["c_act"] + C_OFF) - np.log(c_hat + C_OFF)            # log residuals
        hit90, hit50, width90 = [], [], []
        for i in R:
            for j in range(5):
                if not np.isfinite(res[i, j]):
                    continue
                cal = allrows[(tau[allrows] < tau[i]) & (ty[allrows] <= tau[i]) & (H[allrows] == H[i])]
                rr = res[cal, j]
                rr = rr[np.isfinite(rr)]
                if len(rr) < 30:
                    continue
                q = np.quantile(rr, [0.05, 0.25, 0.75, 0.95])
                lc = np.log(c_hat[i, j] + C_OFF)
                y = np.log(T["c_act"][i, j] + C_OFF)
                hit90.append(q[0] <= y - lc <= q[3])
                hit50.append(q[1] <= y - lc <= q[2])
                width90.append(np.exp(q[3]) - np.exp(q[0]))
                lo, hi, z = lc + q[0], lc + q[3], y
                iscore = (hi - lo) + (2 / 0.1) * max(lo - z, 0) + (2 / 0.1) * max(z - hi, 0)   # log scale
                cov_rows.append(dict(model=m, h=H[i], sector=SECT[j], hit90=hit90[-1], hit50=hit50[-1],
                                     width90=width90[-1], is90=iscore))
    cv = pd.DataFrame(cov_rows)
    cols = ["hit90", "hit50", "width90", "is90"]
    cv.groupby(["model", "h"])[cols].mean().reset_index().to_csv("results/conformal_by_horizon.csv", index=False)
    cv.groupby(["model"])[cols].mean().reset_index().to_csv("results/conformal_overall.csv", index=False)
    cv.groupby(["model", "sector"])[cols].mean().reset_index().to_csv("results/conformal_by_sector.csv", index=False)

    # ---- stoichiometric violation of unconstrained forecasts vs observed emission-factor drift ----
    from features import coupled_mask, E_OFF
    cm0 = coupled_mask(T["c0"], T["e0"])
    ef0 = T["c0"][:, :3] / np.maximum(T["e0"], 1e-9)
    cm1 = (T["e_act"] >= 1.0) & (T["c_act"][:, :3] >= 0.1)
    act_drift = np.abs(np.log(np.maximum(T["c_act"][:, :3], 1e-9) / np.maximum(T["e_act"], 1e-9)) - np.log(np.maximum(ef0, 1e-9)))
    vrows = []
    for m in [x for x in ["damped", "XGB-D", "NN-U", "COMB"] if x in models]:
        gc, ge = mean_over_seeds(P[m])
        c_hat = level_from_growth(gc, T["c0"], C_OFF)[:, :3]
        e_hat = level_from_growth(ge, T["e0"], E_OFF)
        imp = np.abs(np.log(np.maximum(c_hat, 1e-9) / np.maximum(e_hat, 1e-9)) - np.log(np.maximum(ef0, 1e-9)))
        for h in range(1, 11):
            r = R[H[R] == h]
            for j, f in enumerate(["coal", "oil", "gas"]):
                ok2 = cm0[r, j] & cm1[r, j] & np.isfinite(imp[r, j])
                vrows.append(dict(model=m, h=h, fuel=f, n=int(ok2.sum()),
                                  implied_med=float(np.median(imp[r, j][ok2])), implied_p90=float(np.quantile(imp[r, j][ok2], 0.9)),
                                  observed_med=float(np.median(act_drift[r, j][ok2])), observed_p90=float(np.quantile(act_drift[r, j][ok2], 0.9))))
    pd.DataFrame(vrows).to_csv("results/violation_by_horizon.csv", index=False)

    # ---- error correlation between the direct and the energy-implied (structural) CO2 forecast ----
    crow = []
    for m in [x for x in ["damped", "XGB-D", "NN-U", "COMB"] if x in models]:
        gc, ge = mean_over_seeds(P[m])
        c_dir = level_from_growth(gc, T["c0"], C_OFF)[:, :3]
        c_str = ef0 * level_from_growth(ge, T["e0"], E_OFF)
        for j, f in enumerate(["coal", "oil", "gas"]):
            r = R[cm0[R, j]]
            ed = np.log(c_dir[r, j] + C_OFF) - np.log(T["c_act"][r, j] + C_OFF)
            es = np.log(c_str[r, j] + C_OFF) - np.log(T["c_act"][r, j] + C_OFF)
            ok = np.isfinite(ed) & np.isfinite(es) & (np.abs(ed) < 5) & (np.abs(es) < 5)   # drop exploding forecasts
            ed, es = ed[ok], es[ok]
            crow.append(dict(model=m, fuel=f, n=int(ok.sum()), n_dropped=int((~ok).sum()), rho=float(np.corrcoef(ed, es)[0, 1]),
                             mdae_direct=float(np.median(np.abs(ed))), mdae_struct=float(np.median(np.abs(es))),
                             mdae_avg=float(np.median(np.abs((ed + es) / 2)))))
    pd.DataFrame(crow).to_csv("results/route_correlation.csv", index=False)

    json.dump(out, open("results/eval_meta.json", "w"), indent=1)
    print("evaluation done:", out)


if __name__ == "__main__":
    main()
