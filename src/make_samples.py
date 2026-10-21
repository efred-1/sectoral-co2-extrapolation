"""Build forecasting samples (country x origin x horizon) and MASE scales from cleaned_data/energy_matched_panel_42countries_1965-2024.csv."""
import pickle
import sys
import pandas as pd

sys.path.insert(0, "src")
from features import build_samples  # noqa: E402
from common import in_sample_scales  # noqa: E402

out_s = sys.argv[1] if len(sys.argv) > 1 else "results/samples.pkl"
out_c = sys.argv[2] if len(sys.argv) > 2 else "results/scales.pkl"
P = pd.read_csv("cleaned_data/energy_matched_panel_42countries_1965-2024.csv")
meta, X, T = build_samples(P)
pickle.dump((meta, X, T), open(out_s, "wb"))
pickle.dump(in_sample_scales(P, meta), open(out_c, "wb"))
print(f"samples: {len(meta):,}  features: {X.shape[1]}")
