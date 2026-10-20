"""Results figures from evaluation outputs."""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

sys.path.insert(0, "src")
from style import model_style, SLOTS, SEQ_BLUE, INK, INK2, MUTED, GRID, DIVERGE, FUEL_COLORS, FUEL_LABELS, panel_label  # noqa: E402

rel = pd.read_csv("results/avgrelmae_by_horizon.csv")
acc = pd.read_csv("results/acc_by_horizon.csv")
tests = pd.read_csv("results/significance_tests.csv")

MAIN = [m for m in ["damped", "XGB-D", "NN-U", "COMB"] if m in set(rel.model)]
LABEL = {"damped": "Damped trend", "damped-P": "Damped-P (post-hoc)", "damped-H": "Damped-H (hard)",
         "COMB": "COMB (damped + XGB-D)", "COMB-P": "COMB-P (proposed)", "COMB-H": "COMB-H (hard)", "trend": "Short-run trend", "naive": "Naive",
         "XGB-E": "XGB-E (emissions only)", "XGB-D": "XGB-D (direct)", "XGB-P": "XGB-P (post-hoc)",
         "XGB-H": "XGB-H (hard)", "NN-U": "NN-U (multi-task)", "NN-S": "NN-S (soft)", "NN-P": "NN-P (post-hoc)",
         "NN-H": "NN-H (hard)", "XGB-H-band": "XGB-H, IPCC band", "XGB-H-drift": "XGB-H, learned drift",
         "NN-H-band": "NN-H, IPCC band"}

# ---- Fig 4: accuracy relative to naive by horizon ----
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
for ax, col, title in [(axes[0], "avgrelmae_sectors", "Individual sources"), (axes[1], "avgrelmae_total", "National total")]:
    ax.axhline(1.0, color=INK2, lw=0.9, ls="--")
    for m in MAIN:
        d = rel[rel.model == m].sort_values("h")
        st = model_style(m)
        ax.plot(d.h, d[col], color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.8, ms=4, label=LABEL[m])
    ax.set_title(title)
    ax.set_xlabel("Forecast horizon h (years)")
    ax.set_xticks(range(1, 11))
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("AvgRelMAE (naive = 1)")
axes[0].text(10.2, 1.0, "naive", fontsize=7, color=INK2, va="bottom", ha="right")
for ax, l in zip(axes, "ab"):
    panel_label(ax, l)
axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5)
fig.tight_layout()
fig.savefig("figs/fig4_accuracy_horizon.png")
plt.close(fig)

# ---- Fig 5: effect of stoichiometric constraints (difference in mean scaled error, 95% CI) ----
comps = [("damped-H", "damped"), ("damped-P", "damped"), ("XGB-H", "XGB-D"), ("XGB-P", "XGB-D"),
         ("NN-S", "NN-U"), ("NN-H", "NN-U"), ("NN-P", "NN-U"), ("COMB-H", "COMB"), ("COMB-P", "COMB")]
comps = [c for c in comps if ((tests.a == c[0]) & (tests.b == c[1])).any()]
groups = ["1", "2-3", "4-6", "7-10"]
HCOL = [SEQ_BLUE[2], SEQ_BLUE[3], SEQ_BLUE[5], SEQ_BLUE[6]]   # ordinal horizon groups: one hue, light to dark
fig, axes = plt.subplots(1, 2, figsize=(7.4, 0.62 * len(comps) + 0.9), sharey=True)
for ax, target, title in [(axes[0], "sectors", "Individual sources"), (axes[1], "total", "National total")]:
    ax.axvline(0, color=INK2, lw=0.9)
    for k, (a, b) in enumerate(comps):
        for gi, g in enumerate(groups):
            r = tests[(tests.a == a) & (tests.b == b) & (tests.horizons == g) & (tests.target == target)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = k * (len(groups) + 1.2) + gi
            sig = r.p_holm < 0.05
            col = HCOL[gi]
            ax.plot([r.ci_lo, r.ci_hi], [y, y], color=col, lw=1.6)
            ax.plot(r.rel_change_pct, y, "o", color=col, ms=5 if sig else 4.5,
                    mfc=col if sig else "white", mec=col, mew=1.3)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
yt = [k * (len(groups) + 1.2) + 1.5 for k in range(len(comps))]
axes[0].set_yticks(yt)
axes[0].set_yticklabels([f"{a} vs {b}" for a, b in comps])
axes[0].invert_yaxis()
from matplotlib.lines import Line2D  # noqa: E402
hand = [Line2D([], [], color=HCOL[i], marker="o", lw=1.6, label=f"h = {g}") for i, g in enumerate(groups)]
hand += [Line2D([], [], color=INK2, marker="o", lw=0, mfc=INK2, label="Holm p < 0.05"),
         Line2D([], [], color=INK2, marker="o", lw=0, mfc="white", label="not significant")]
axes[1].legend(handles=hand, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5)
for ax in axes:
    lo, hi = ax.get_xlim()
    ax.text(0.02, -0.06, "← constraint helps", transform=ax.transAxes, fontsize=7, color=INK2, ha="left", va="top")
    ax.text(0.98, -0.06, "constraint hurts →", transform=ax.transAxes, fontsize=7, color=INK2, ha="right", va="top")
for ax, l in zip(axes, "ab"):
    panel_label(ax, l)
fig.supxlabel("Change in mean absolute error relative to the unconstrained twin (%)", fontsize=9, color=INK2, y=0.0)
fig.tight_layout()
fig.savefig("figs/fig5_physics_effect.png")
plt.close(fig)

# ---- Fig 6: per-source AvgRelMAE against naive, by horizon group ----
rs = pd.read_csv("results/avgrelmae_by_source.csv")
models = [m for m in ["damped", "damped-H", "damped-P", "XGB-E", "XGB-D", "XGB-H", "XGB-P", "NN-U", "NN-S", "NN-H", "NN-P",
                      "COMB", "COMB-H", "COMB-P"] if m in set(rs.model)]
gmap = {"1": [1], "2-3": [2, 3], "4-6": [4, 5, 6], "7-10": [7, 8, 9, 10]}
srcs = ["coal", "oil", "gas", "flaring", "cement"]
mat = np.zeros((len(models), len(srcs) * len(gmap)))
for i, m in enumerate(models):
    for gi, g in enumerate(gmap):
        for si, s in enumerate(srcs):
            mat[i, gi * len(srcs) + si] = rs[(rs.model == m) & (rs.group == g) & (rs.source == s)].avgrelmae.iloc[0]
cmap = LinearSegmentedColormap.from_list("div", [DIVERGE[0], DIVERGE[1], DIVERGE[2]])
fig, ax = plt.subplots(figsize=(7.4, 0.27 * len(models) + 1.3))
im = ax.imshow(mat, cmap=cmap, norm=TwoSlopeNorm(vmin=0.7, vcenter=1.0, vmax=1.3), aspect="auto")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.2,
                color="white" if abs(mat[i, j] - 1) > 0.22 else INK)
ax.set_yticks(range(len(models)))
ax.set_yticklabels([LABEL[m] for m in models], fontsize=7.5)
ax.set_xticks(range(mat.shape[1]))
ax.set_xticklabels([FUEL_LABELS[s] for s in srcs] * len(gmap), rotation=90, fontsize=7)
for gi, g in enumerate(gmap):
    ax.text(gi * len(srcs) + 2, -0.9, f"h = {g}", ha="center", fontsize=8, color=INK2)
    if gi:
        ax.axvline(gi * len(srcs) - 0.5, color="white", lw=2.5)
ax.grid(False)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cb.set_label("AvgRelMAE (naive = 1)", fontsize=7.5)
cb.outline.set_visible(False)
fig.tight_layout()
fig.savefig("figs/fig6_source_heatmap.png")
plt.close(fig)
print("results figures written")
