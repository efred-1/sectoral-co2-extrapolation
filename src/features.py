"""Sample construction for direct multi-horizon forecasting.

A sample is (country, origin year tau, horizon h). Features use only data in the
five-year window tau-4..tau. Targets are log growth from tau to tau+h:
    g = log(y[tau+h] + c) - log(y[tau] + c)
For a model fitted at forecast origin T0, training samples need tau + h <= T0.
"""
import numpy as np
import pandas as pd

SECTORS = ["coal_co2", "oil_co2", "gas_co2", "flaring_co2", "cement_co2"]
FUELS = ["coal", "oil", "gas"]
ECOLS = [f"{f}_consumption" for f in FUELS]
C_OFF = 0.1     # Mt CO2 offset for log of small series
E_OFF = 1.0     # TWh offset for fuel energy
E_MIN = 1.0     # TWh: fuel considered present for stoichiometric coupling
C_MIN = 0.1     # Mt
WIN = 5
H_MAX = 10
FIRST_TAU = 1974          # window 1970-1974


def _lg(x, off):
    return np.log(np.asarray(x, dtype=float) + off)


def window_features(g, tau):
    """Feature vector (without horizon) for one country at origin tau, or None if the window is incomplete.
    g: country frame indexed by year. Returns (base, c0, e0)."""
    win = list(range(tau - WIN + 1, tau + 1))
    if not all(y in g.index for y in win):
        return None
    Cw = g.loc[win, SECTORS].values.astype(float)
    Ew = g.loc[win, ECOLS].values.astype(float)
    if np.isnan(Cw).any() or np.isnan(Ew).any():
        return None
    lc = _lg(Cw, C_OFF)
    le = _lg(Ew, E_OFF)
    dlc = np.diff(lc, axis=0)
    dle = np.diff(le, axis=0)
    c0, e0 = Cw[-1], Ew[-1]
    ef0 = np.where((e0 >= E_MIN) & (c0[:3] >= C_MIN), c0[:3] / np.maximum(e0, 1e-9), np.nan)
    lcs = g.loc[win, "low_carbon_share"].values.astype(float)
    pop = g.loc[win, "population"].values.astype(float)
    lcs0 = lcs[-1] if np.isfinite(lcs[-1]) else 0.0
    lcs_d = (lcs[-1] - lcs[0]) if np.isfinite(lcs).all() else 0.0
    popg = float(np.mean(np.diff(np.log(pop)))) if np.isfinite(pop).all() and (pop > 0).all() else 0.0
    base = np.concatenate([
        lc[-1], dlc.ravel(), dlc.mean(axis=0),          # 5 + 20 + 5
        le[-1], dle.mean(axis=0), dle[-1],              # 3 + 3 + 3
        np.nan_to_num(np.log(ef0), nan=0.0),            # 3
        (np.isfinite(ef0)).astype(float),               # 3 coupling indicators
        [lcs0, lcs_d, popg],                            # 3
    ])
    return base, c0, e0


def future_rows(panel, tau, horizons):
    """Feature rows at origin tau for future horizons (no targets). Returns meta, X, c0, e0."""
    rows, feats, c0s, e0s = [], [], [], []
    for country, g in panel.sort_values(["country", "year"]).groupby("country"):
        r = window_features(g.set_index("year"), tau)
        if r is None:
            continue
        base, c0, e0 = r
        for h in horizons:
            rows.append(dict(country=country, tau=tau, h=h, target_year=tau + h))
            feats.append(np.concatenate([base, [h]]))
            c0s.append(c0); e0s.append(e0)
    return pd.DataFrame(rows), np.asarray(feats, dtype=np.float32), np.asarray(c0s), np.asarray(e0s)


def build_samples(panel, h_max=H_MAX, first_tau=FIRST_TAU, last_target=2024):
    """Return (meta DataFrame, feature matrix, target dict, origin-level dict)."""
    P = panel.sort_values(["country", "year"])
    rows, feats = [], []
    tgt_c, tgt_e, lev_c, lev_e, act_c, act_e = [], [], [], [], [], []
    for country, g in P.groupby("country"):
        g = g.set_index("year")
        years = set(g.index)
        C = g[SECTORS]
        E = g[ECOLS]
        for tau in range(first_tau, last_target):
            r = window_features(g, tau)
            if r is None:
                continue
            base, c0, e0 = r
            for h in range(1, h_max + 1):
                t = tau + h
                if t > last_target or t not in years:
                    continue
                yc = C.loc[t].values.astype(float)
                ye = E.loc[t].values.astype(float)
                rows.append(dict(country=country, tau=tau, h=h, target_year=t))
                feats.append(np.concatenate([base, [h]]))
                tgt_c.append(_lg(yc, C_OFF) - _lg(c0, C_OFF))
                tgt_e.append(_lg(ye, E_OFF) - _lg(e0, E_OFF))
                lev_c.append(c0); lev_e.append(e0)
                act_c.append(yc); act_e.append(ye)
    meta = pd.DataFrame(rows)
    X = np.asarray(feats, dtype=np.float32)
    out = dict(
        g_c=np.asarray(tgt_c), g_e=np.asarray(tgt_e),
        c0=np.asarray(lev_c), e0=np.asarray(lev_e),
        c_act=np.asarray(act_c), e_act=np.asarray(act_e),
    )
    return meta, X, out


FEATURE_NAMES = (
    [f"log_{s}" for s in SECTORS]
    + [f"dlog_{s}_lag{k}" for k in range(4, 0, -1) for s in SECTORS]
    + [f"mean_dlog_{s}" for s in SECTORS]
    + [f"log_{f}_energy" for f in FUELS]
    + [f"mean_dlog_{f}_energy" for f in FUELS]
    + [f"last_dlog_{f}_energy" for f in FUELS]
    + [f"log_ef_{f}" for f in FUELS]
    + [f"coupled_{f}" for f in FUELS]
    + ["low_carbon_share", "d_low_carbon_share", "pop_growth", "horizon"]
)


def coupled_mask(c0, e0):
    """(n,3) boolean: stoichiometric coupling applies for coal, oil, gas."""
    return (e0 >= E_MIN) & (c0[:, :3] >= C_MIN)


def level_from_growth(g, level0, off):
    return np.maximum((level0 + off) * np.exp(g) - off, 0.0)
