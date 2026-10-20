"""Local statistical baselines, fitted per country-series at each forecast origin.

  naive  : g = 0 (random walk / persistence)
  trend  : g = h * mean annual log growth over the 5-year window
  damped : additive damped-trend exponential smoothing on log(y + c), full history to origin
Outputs predicted log growth for 5 CO2 sectors and 3 fuel energies, aligned to sample rows.
"""
import pickle
import sys
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

sys.path.insert(0, "src")
from features import SECTORS, ECOLS, C_OFF, E_OFF  # noqa: E402

warnings.filterwarnings("ignore")
ORIGINS = range(1995, 2020)


def damped_path(y, hmax):
    """Return damped-trend forecast path (length hmax) in the same (log) units as y."""
    y = np.asarray(y, dtype=float)
    if len(y) < 10 or np.ptp(y) < 1e-9:
        return np.repeat(y[-1], hmax)
    try:
        fit = ExponentialSmoothing(y, trend="add", damped_trend=True,
                                   initialization_method="estimated").fit()
        return np.asarray(fit.forecast(hmax))
    except Exception:
        return np.repeat(y[-1], hmax)


def main():
    meta, X, T = pickle.load(open("results/samples.pkl", "rb"))
    P = pd.read_csv("data/panel.csv").sort_values(["country", "year"])
    series = {}
    for country, g in P.groupby("country"):
        g = g.set_index("year")
        for j, s in enumerate(SECTORS):
            series[(country, "c", j)] = g[s]
        for j, e in enumerate(ECOLS):
            series[(country, "e", j)] = g[e]

    n = len(meta)
    out = {m: (np.full((n, 5), np.nan), np.full((n, 3), np.nan)) for m in ["naive", "trend", "damped"]}
    mean_dlog_c = X[:, 25:30]          # mean_dlog of sectors in window
    mean_dlog_e = X[:, 33:36]
    hcol = X[:, -1]
    for T0 in ORIGINS:
        idx = np.where(meta.tau.values == T0)[0]
        out["naive"][0][idx] = 0.0
        out["naive"][1][idx] = 0.0
        out["trend"][0][idx] = mean_dlog_c[idx] * hcol[idx, None]
        out["trend"][1][idx] = mean_dlog_e[idx] * hcol[idx, None]
        sub = meta.iloc[idx]
        for country, rows in sub.groupby("country"):
            ridx = rows.index.values
            hs = rows.h.values
            hmax = hs.max()
            for kind, ncol, off in [("c", 5, C_OFF), ("e", 3, E_OFF)]:
                for j in range(ncol):
                    s = series[(country, kind, j)]
                    hist = s[s.index <= T0]
                    # contiguous non-missing tail ending at T0
                    vals = hist.values
                    k = len(vals)
                    while k > 0 and np.isfinite(vals[k - 1]):
                        k -= 1
                    tail = vals[k:]
                    if len(tail) == 0:
                        continue
                    ly = np.log(tail + off)
                    path = damped_path(ly, hmax)
                    g = path[hs - 1] - ly[-1]
                    target = out["damped"][0] if kind == "c" else out["damped"][1]
                    target[ridx, j] = g
        print(f"origin {T0} done", flush=True)
    pickle.dump(out, open("results/pred_baselines.pkl", "wb"))


if __name__ == "__main__":
    main()
