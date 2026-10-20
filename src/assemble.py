"""Assemble model forecasts (log growth) from raw components.

Physics-structured CO2 forecast for coupled fuels (stoichiometric coupling, CO2 = EF x E):
    C_hat = EF0 * E_hat                       (hard, emission factor held at origin value)
Post-hoc projection onto the same constraint (Euclidean in log space):
    log C' = (log C_direct + log C_struct)/2,  log E' = (log E_hat + log C_direct - log EF0)/2
Sensitivity variants:
    hard-band  : EF0 clipped into the IPCC 2006 band before multiplying
    hard-drift : EF0 * exp(clip(d_hat, -q_h, q_h)), d_hat from a learned drift model
"""
import numpy as np

from features import C_OFF, E_OFF, coupled_mask, level_from_growth

IPCC_BAND = np.array([[0.3143, 0.414], [0.2034, 0.2837], [0.1955, 0.2099]])  # coal, oil, gas (t/MWh)


def _to_growth_c(level, c0):
    return np.log(np.maximum(level, 0) + C_OFF) - np.log(c0 + C_OFF)


def _to_growth_e(level, e0):
    return np.log(np.maximum(level, 0) + E_OFF) - np.log(e0 + E_OFF)


def physics_variants(gc_direct, ge, c0, e0, h=None, drift=None, q=None):
    """Return dict of (gc (n,5), ge (n,3)) for hard, posthoc, hard_band, hard_drift."""
    cm = coupled_mask(c0, e0)                                  # (n,3)
    ef0 = np.where(cm, c0[:, :3] / np.maximum(e0, 1e-9), np.nan)
    e_hat = level_from_growth(ge, e0, E_OFF)                   # (n,3) TWh
    c_dir = level_from_growth(gc_direct, c0, C_OFF)            # (n,5) Mt

    def struct(ef):
        gc = gc_direct.copy()
        cs = ef * e_hat
        gc[:, :3] = np.where(cm, _to_growth_c(np.nan_to_num(cs), c0[:, :3]), gc_direct[:, :3])
        return gc

    out = {"hard": (struct(ef0), ge.copy())}
    # post-hoc Euclidean projection in log space
    lc_d = np.log(np.maximum(c_dir[:, :3], 1e-6))
    le = np.log(np.maximum(e_hat, 1e-6))
    lef = np.log(np.where(cm, ef0, 1.0))
    viol = lc_d - le - lef
    lc_p = lc_d - viol / 2
    le_p = le + viol / 2
    gc_p, ge_p = gc_direct.copy(), ge.copy()
    gc_p[:, :3] = np.where(cm, _to_growth_c(np.exp(lc_p), c0[:, :3]), gc_direct[:, :3])
    ge_p = np.where(cm, _to_growth_e(np.exp(le_p), e0), ge)
    out["posthoc"] = (gc_p, ge_p)
    out["hard_band"] = (struct(np.clip(ef0, IPCC_BAND[:, 0], IPCC_BAND[:, 1])), ge.copy())
    if drift is not None and q is not None and h is not None:
        qh = q[h - 1]                                          # (n,3)
        d = np.clip(drift, -qh, qh)
        out["hard_drift"] = (struct(ef0 * np.exp(d)), ge.copy())
    return out
