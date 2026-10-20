"""Select neural-network settings from development scores only (targets <= 2007)."""
import glob
import json
import pickle
import sys

sys.path.insert(0, "src")
from score_dev import score  # noqa: E402

mode = sys.argv[1]
path = "results/dev/nn_choice.json"
if mode == "nn":
    best = None
    rows = []
    for f in sorted(glob.glob("results/dev/nnU_*.pkl")):
        cfg = pickle.load(open(f, "rb"))["cfg"]
        v = score(["NN-U"], nn_files=[f])["NN-U"]
        rows.append(dict(file=f, width=cfg["width"], dropout=cfg["dropout"], mase=v[0], mase_total=v[1]))
        if best is None or v[0] < best[0]:
            best = (v[0], cfg["width"], cfg["dropout"])
    json.dump(dict(width=best[1], dropout=best[2], grid=rows), open(path, "w"), indent=1)
    print("NN choice:", best)
elif mode == "lam":
    choice = json.load(open(path))
    rows = []
    best = None
    for f in sorted(glob.glob("results/dev/nnS_*.pkl")):
        cfg = pickle.load(open(f, "rb"))["cfg"]
        v = score(["NN-S"], nn_files=[f])["NN-S"]
        rows.append(dict(file=f, lam=cfg["lam"], mase=v[0], mase_total=v[1]))
        if best is None or v[0] < best[0]:
            best = (v[0], cfg["lam"])
    choice["lam"] = best[1]
    choice["lam_grid"] = rows
    json.dump(choice, open(path, "w"), indent=1)
    print("lambda choice:", best)
