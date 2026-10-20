"""Score development-period runs on targets <= DEV_LAST_TARGET only."""
import glob
import pickle
import sys
import numpy as np

sys.path.insert(0, "src")
from evallib import load_samples, build_predictions, mean_over_seeds, errors  # noqa: E402
from common import DEV_ORIGINS, DEV_LAST_TARGET  # noqa: E402

meta, X, T = load_samples()
scales = pickle.load(open("results/scales.pkl", "rb"))
rows = np.where(meta.tau.isin(DEV_ORIGINS).values & (meta.target_year.values <= DEV_LAST_TARGET))[0]
Tsub = {k: v[rows] for k, v in T.items()}
ssub = tuple(s[rows] for s in scales)
print(f"dev rows: {len(rows)} (origins {DEV_ORIGINS[0]}-{DEV_ORIGINS[-1]}, targets <= {DEV_LAST_TARGET})")


def score(models, xgb_files=(), nn_files=()):
    P = build_predictions(meta, X, T, list(xgb_files), list(nn_files))
    out = {}
    for m in models:
        gc, _ = mean_over_seeds(P[m])
        e = errors(gc[rows], Tsub, ssub)
        out[m] = (np.nanmean(e["se"]), np.nanmean(e["se_tot"]))
    return out


if __name__ == "__main__":
    base = score(["naive", "trend", "damped"])
    for m, v in base.items():
        print(f"  {m:8s} MASE sectors {v[0]:.4f}  total {v[1]:.4f}")
    for f in sorted(glob.glob("results/dev/xgb_*.pkl")):
        v = score(["XGB-E", "XGB-D", "XGB-H", "XGB-P"], xgb_files=[f])
        print(f, " ".join(f"{m}:{a:.4f}/{b:.4f}" for m, (a, b) in v.items()))
    for f in sorted(glob.glob("results/dev/nn*.pkl")):
        cfg = pickle.load(open(f, "rb"))["cfg"]
        names = {"U": ["NN-U", "NN-P"], "S": ["NN-S"], "H": ["NN-H"]}[cfg["variant"]]
        v = score(names, nn_files=[f])
        print(f, " ".join(f"{m}:{a:.4f}/{b:.4f}" for m, (a, b) in v.items()))
