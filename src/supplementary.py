"""Supplementary analyses added after the test results were seen (reported as such in the paper).

S1  Weighted post-forecast correction. The equal-weight projection (Eq. 6) is generalised to
        x* = x - (1 - w) v,   z* = z + w v,     v = x - z - a,
    so the corrected CO2 forecast is w * direct + (1 - w) * structural (in logs) and the energy forecast
    moves by the complementary amount; the constraint x* - z* = a still holds exactly. The weight w on the
    direct forecast is the error-variance-optimal combination weight (Bates and Granger), estimated at every
    origin tau from log errors of earlier forecasts whose targets were observed by tau (same horizon and fuel,
    pooled over countries), clipped to [0, 1]. No test outcome after tau is used.

S2  Perfect-energy-foresight conversion test. Observed fuel energy in the target year is converted to CO2 with
    (a) the country's own emission factor at the origin, (b) that factor clipped into the IPCC band, and
    (c) a single IPCC default factor per fuel. The error against inventory CO2 isolates the conversion step that
    planners apply to energy outlooks.
"""
import glob
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from evallib import load_samples, build_predictions, mean_over_seeds, errors, energy_errors  # noqa: E402
from features import C_OFF, E_OFF, level_from_growth, coupled_mask  # noqa: E402
from common import TEST_ORIGINS, ALL_ORIGINS  # noqa: E402
from assemble import IPCC_BAND  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
import pickle  # noqa: E402

FUELS = ["coal", "oil", "gas"]
SECT = ["coal", "oil", "gas", "flaring", "cement"]
# IPCC 2006 default CO2 factors (kg/TJ -> t/MWh): other bituminous coal 94,600; crude oil 73,300; natural gas 56,100
IPCC_DEFAULT = np.array([94600, 73300, 56100]) * 3.6e-6
rng = np.random.default_rng(2026)


def weighted_variant(gc, ge, T, meta, w_table=None):
    """Apply the rolling-weight correction. Returns gc_w, ge_w and the weight used per row and fuel."""
    n = len(meta)
    c0, e0 = T["c0"], T["e0"]
    cm = coupled_mask(c0, e0)
    ef0 = np.where(cm, c0[:, :3] / np.maximum(e0, 1e-9), np.nan)
    c_dir = level_from_growth(gc, c0, C_OFF)[:, :3]
    e_hat = level_from_growth(ge, e0, E_OFF)
    x = np.log(np.maximum(c_dir, 1e-6))
    z = np.log(np.maximum(e_hat, 1e-6))
    a = np.log(np.where(cm, ef0, 1.0))
    v = x - z - a
    y = np.log(np.maximum(T["c_act"][:, :3], 1e-6))
    ed, es = x - y, (z + a) - y                                   # log errors of direct and structural forecasts
    tau, ty, H = meta.tau.values, meta.target_year.values, meta.h.values
    W = np.full((n, 3), np.nan)
    rows = np.where(np.isin(tau, ALL_ORIGINS) & np.isfinite(gc[:, 0]) & np.isfinite(ge[:, 0]))[0]
    cache = {}
    for i in rows:
        for j in range(3):
            if not cm[i, j]:
                continue
            key = (tau[i], H[i], j)
            if key not in cache:
                cal = rows[(tau[rows] < tau[i]) & (ty[rows] <= tau[i]) & (H[rows] == H[i]) & cm[rows, j]]
                d, s = ed[cal, j], es[cal, j]
                ok = np.isfinite(d) & np.isfinite(s) & (np.abs(d) < 5) & (np.abs(s) < 5)
                d, s = d[ok], s[ok]
                if len(d) < 30:
                    cache[key] = 0.5
                else:
                    num = np.mean(s * s) - np.mean(d * s)
                    den = np.mean((d - s) ** 2)
                    cache[key] = float(np.clip(num / den, 0, 1)) if den > 1e-12 else 0.5
            W[i, j] = cache[key]
    gc_w, ge_w = gc.copy(), ge.copy()
    w = np.nan_to_num(W, nan=1.0)
    xs = x - (1 - w) * v
    zs = z + w * v
    gc_w[:, :3] = np.where(cm, np.log(np.exp(xs) + C_OFF) - np.log(c0[:, :3] + C_OFF), gc[:, :3])
    ge_w = np.where(cm, np.log(np.exp(zs) + E_OFF) - np.log(e0 + E_OFF), ge)
    return gc_w, ge_w, W


def rel_change(Ea, Eb, R, key, hs_rows):
    """Country-level mean log MAE ratio over sources (as in the main tests)."""
    lr = []
    for j in range(5):
        df = pd.DataFrame({"c": key[hs_rows], "a": Ea["ae"][hs_rows, j], "b": Eb["ae"][hs_rows, j]}).dropna()
        g = df.groupby("c")[["a", "b"]].mean()
        g = g[(g.a > 1e-9) & (g.b > 1e-9)]
        lr.append(np.log(g.a / g.b).rename(j))
    dc = pd.concat(lr, axis=1).mean(axis=1).values
    boots = [dc[rng.integers(0, len(dc), len(dc))].mean() for _ in range(2000)]
    nz = np.abs(dc) > 1e-12
    p = wilcoxon(dc[nz]).pvalue if nz.sum() > 5 else 1.0
    return dict(rel_change_pct=100 * (np.exp(dc.mean()) - 1), ci_lo=100 * (np.exp(np.quantile(boots, 0.025)) - 1),
                ci_hi=100 * (np.exp(np.quantile(boots, 0.975)) - 1), n_better=int((dc < 0).sum()), n=len(dc), p=p)


def main():
    meta, X, T = load_samples()
    scales = pickle.load(open("results/scales.pkl", "rb"))
    P = build_predictions(meta, X, T, sorted(glob.glob("results/full/xgb_*.pkl")), sorted(glob.glob("results/full/nn*.pkl")))
    test = meta.tau.isin(TEST_ORIGINS).values
    H = meta.h.values
    key = meta.country.values
    groups = {"1": [1], "2-3": [2, 3], "4-6": [4, 5, 6], "7-10": [7, 8, 9, 10], "all": list(range(1, 11))}
    out, wrows, summ = [], [], []
    for base, eq in [("damped", "damped-P"), ("XGB-D", "XGB-P"), ("NN-U", "NN-P"), ("COMB", "COMB-P")]:
        gc, ge = mean_over_seeds(P[base])
        gcw, gew, W = weighted_variant(gc, ge, T, meta)
        Eu = errors(gc, T, scales)
        Ep = errors(mean_over_seeds(P[eq])[0], T, scales)
        Ew = errors(gcw, T, scales)
        R = np.where(test & np.isfinite(Eu["se"]).any(axis=1) & np.isfinite(Ew["se"]).any(axis=1))[0]
        name = base.split("-")[0] + "-W" if base != "damped" else "damped-W"
        for gname, hs in groups.items():
            r = R[np.isin(H[R], hs)]
            for comp, Eb in [("unconstrained", Eu), ("equal-weight", Ep)]:
                d = rel_change(Ew, Eb, R, key, r)
                out.append(dict(model=name, versus=comp, horizons=gname, **d))
        en_u = energy_errors(ge, T, scales)["se"][R]
        en_w = energy_errors(gew, T, scales)["se"][R]
        summ.append(dict(model=name, base=base, mase_w=np.nanmean(Ew["se"][R]), mase_p=np.nanmean(Ep["se"][R]),
                         mase_u=np.nanmean(Eu["se"][R]), mdase_w=np.nanmedian(Ew["se"][R]), mdase_p=np.nanmedian(Ep["se"][R]),
                         mdase_u=np.nanmedian(Eu["se"][R]), energy_u=np.nanmean(en_u), energy_w=np.nanmean(en_w),
                         masetot_w=np.nanmean(Ew["se_tot"][R]), masetot_p=np.nanmean(Ep["se_tot"][R])))
        for j, f in enumerate(FUELS):
            for h in [1, 5, 10]:
                sel = R[(H[R] == h) & np.isfinite(W[R, j])]
                wrows.append(dict(model=name, fuel=f, h=h, w_median=float(np.median(W[sel, j])),
                                  w_p10=float(np.quantile(W[sel, j], 0.1)), w_p90=float(np.quantile(W[sel, j], 0.9))))
    o = pd.DataFrame(out)
    o["p_bonf"] = np.minimum(1, o.p * len(o))         # conservative Bonferroni over this family
    o.to_csv("results/weighted_tests.csv", index=False)
    pd.DataFrame(wrows).to_csv("results/weighted_weights.csv", index=False)
    pd.DataFrame(summ).to_csv("results/weighted_summary.csv", index=False)

    # ---- S2: perfect-energy-foresight conversion ----
    Pn = pd.read_csv("data/panel.csv")
    En = pd.read_csv("data/owid-energy-data.csv", usecols=["country", "year", "oil_production", "gas_production",
                                                            "oil_consumption", "gas_consumption"])
    prod = En[En.year.between(2009, 2019)].groupby("country")[["oil_production", "gas_production",
                                                               "oil_consumption", "gas_consumption"]].mean()
    producer = ((prod.oil_production > prod.oil_consumption) | (prod.gas_production > prod.gas_consumption))
    producers = sorted(set(producer[producer].index) & set(Pn.country))
    cm0 = coupled_mask(T["c0"], T["e0"])
    ef0 = T["c0"][:, :3] / np.maximum(T["e0"], 1e-9)
    ok1 = (T["e_act"] >= 1.0) & (T["c_act"][:, :3] >= 0.1)
    rows = np.where(test)[0]
    is_prod = np.isin(key, producers)
    variants = {"Country factor at origin": ef0,
                "Country factor clipped to IPCC band": np.clip(ef0, IPCC_BAND[:, 0], IPCC_BAND[:, 1]),
                "IPCC default factor": np.broadcast_to(IPCC_DEFAULT, ef0.shape)}
    srows = []
    for vname, ef in variants.items():
        chat = ef * T["e_act"]
        ape = 100 * np.abs(chat - T["c_act"][:, :3]) / T["c_act"][:, :3]
        sgn = 100 * (chat - T["c_act"][:, :3]) / T["c_act"][:, :3]
        for j, f in enumerate(FUELS):
            for grp, gmask in [("all", np.ones(len(meta), bool)), ("producers", is_prod), ("importers", ~is_prod)]:
                for h in ["1", "5", "10", "all"]:
                    hm = np.ones(len(meta), bool) if h == "all" else (H == int(h))
                    r = rows[cm0[rows, j] & ok1[rows, j] & gmask[rows] & hm[rows]]
                    srows.append(dict(variant=vname, fuel=f, group=grp, h=h, n=len(r),
                                      mdape=float(np.median(ape[r, j])), p90ape=float(np.quantile(ape[r, j], 0.9)),
                                      median_bias=float(np.median(sgn[r, j]))))
        # three-fuel combustion total per row
        tot_hat = np.nansum(np.where(cm0 & ok1, chat, np.nan), axis=1)
        tot_act = np.nansum(np.where(cm0 & ok1, T["c_act"][:, :3], np.nan), axis=1)
        for grp, gmask in [("all", np.ones(len(meta), bool)), ("producers", is_prod), ("importers", ~is_prod)]:
            r = rows[gmask[rows] & (tot_act[rows] > 0.1)]
            ape_t = 100 * np.abs(tot_hat[r] - tot_act[r]) / tot_act[r]
            srows.append(dict(variant=vname, fuel="combustion total", group=grp, h="all", n=len(r),
                              mdape=float(np.median(ape_t)), p90ape=float(np.quantile(ape_t, 0.9)),
                              median_bias=float(np.median(100 * (tot_hat[r] - tot_act[r]) / tot_act[r]))))
    pd.DataFrame(srows).to_csv("results/foresight_conversion.csv", index=False)
    json.dump(dict(producers=producers, ipcc_default_t_per_mwh=dict(zip(FUELS, IPCC_DEFAULT.round(4).tolist()))),
              open("results/foresight_meta.json", "w"), indent=1)
    print("supplementary analyses written; producers:", len(producers))


if __name__ == "__main__":
    main()
