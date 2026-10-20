"""Shared settings, scaling, and metrics."""
import numpy as np
import pandas as pd

from features import SECTORS, ECOLS

ALL_ORIGINS = list(range(1995, 2020))       # models are fitted at every origin (needed for conformal calibration)
DEV_ORIGINS = list(range(1998, 2003))       # hyperparameter tuning, scored only on targets <= DEV_LAST_TARGET
DEV_LAST_TARGET = 2007
TEST_ORIGINS = list(range(2008, 2020))      # reported test evaluation (all targets >= 2009)
HORIZONS = list(range(1, 11))
SCALE_FLOOR = {"c": 0.05, "e": 0.5}         # Mt, TWh floors for near-constant series


def in_sample_scales(panel, meta):
    """MASE denominators: mean absolute one-step change of each series up to the origin (tau).

    Returns arrays (n,5) for CO2 sectors and (n,3) for fuel energy, aligned to meta rows.
    """
    P = panel.sort_values(["country", "year"])
    cache = {}
    for country, g in P.groupby("country"):
        g = g.set_index("year")
        cache[country] = (g[SECTORS], g[ECOLS])
    sc = np.full((len(meta), 5), np.nan)
    se = np.full((len(meta), 3), np.nan)
    st = np.full(len(meta), np.nan)
    keyed = {}
    for i, (country, tau) in enumerate(zip(meta.country.values, meta.tau.values)):
        k = (country, tau)
        if k not in keyed:
            C, E = cache[country]
            Ch = C[C.index <= tau]
            dc = Ch.diff().abs()
            de = E[E.index <= tau].diff().abs()
            dt = Ch.sum(axis=1, min_count=5).diff().abs()
            keyed[k] = (np.maximum(dc.mean().values, SCALE_FLOOR["c"]),
                        np.maximum(de.mean().values, SCALE_FLOOR["e"]),
                        max(dt.mean(), SCALE_FLOOR["c"]))
        sc[i], se[i], st[i] = keyed[k]
    return sc, se, st


def summarise(err_abs, scale, mask=None):
    """Mean absolute scaled error over valid entries."""
    q = err_abs / scale
    if mask is not None:
        q = np.where(mask, q, np.nan)
    return np.nanmean(q)


XGB_PARAMS = dict(learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=3, tree_method="hist", n_jobs=2)
