"""Implied emission-factor drift (Fig. 7), interval calibration (Fig. 8) and 2025-2030 projections (Fig. 9)."""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from style import model_style, SLOTS, INK, INK2, MUTED, FUEL_COLORS, FUEL_LABELS, panel_label  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "COMB-P"
cv = pd.read_csv("results/conformal_by_horizon.csv")
show = [m for m in ["naive", "damped-P", "XGB-P", "NN-P", "COMB-P"] if m in set(cv.model)]
LABEL = {"naive": "Naive", "damped": "Damped trend", "damped-P": "Damped-P", "XGB-D": "XGB-D", "XGB-P": "XGB-P",
         "NN-U": "NN-U", "NN-P": "NN-P", "COMB-P": "COMB-P", "COMB": "COMB"}

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
for m in show:
    d = cv[cv.model == m].sort_values("h")
    st = model_style(m)
    kw = dict(color=st["color"], marker=st["marker"], ls=st["ls"], ms=3.5, lw=1.5)
    axes[0].plot(d.h, 100 * d.hit90, label=LABEL[m], **kw)
    axes[1].plot(d.h, d.width90, **kw)
    axes[2].plot(d.h, d.is90, **kw)
axes[0].axhline(90, color=INK2, ls="--", lw=0.9)
axes[0].set_ylabel("Empirical coverage (%)")
axes[0].set_title("Coverage of 90% interval")
axes[1].set_ylabel("Width (fraction of forecast)")
axes[1].set_title("Mean relative width")
axes[2].set_ylabel("Interval score (log units)")
axes[2].set_title("Interval score")
for ax, l in zip(axes, "abc"):
    ax.set_xlabel("Horizon h (years)")
    ax.set_xticks([1, 4, 7, 10])
    ax.grid(axis="x", visible=False)
    panel_label(ax, l)
axes[0].legend(fontsize=6.8, loc="lower left")
fig.tight_layout()
fig.savefig("figs/fig8_intervals.png")
plt.close(fig)

# ---- implied emission-factor drift of unconstrained forecasts vs observed drift ----
vi = pd.read_csv("results/violation_by_horizon.csv")
vm = [m for m in ["damped", "XGB-D", "NN-U", "COMB"] if m in set(vi.model)]
LABEL.update({"damped": "Damped trend", "XGB-D": "XGB-D", "NN-U": "NN-U", "COMB": "COMB"})
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True)
for ax, f, l in zip(axes, ["coal", "oil", "gas"], "abc"):
    ob = vi[(vi.model == vm[0]) & (vi.fuel == f)].sort_values("h")
    ax.fill_between(ob.h, 0, 100 * ob.observed_med, color=MUTED, alpha=0.25, lw=0, label="Observed drift (median)")
    for k, m in enumerate(vm):
        d = vi[(vi.model == m) & (vi.fuel == f)].sort_values("h")
        st = model_style(m)
        ax.plot(d.h, 100 * d.implied_med, color=st["color"], marker=st["marker"], lw=1.6, ms=3.5, label=LABEL.get(m, m))
    ax.set_title(FUEL_LABELS[f])
    ax.set_xlabel("Horizon h (years)")
    ax.set_xticks([1, 4, 7, 10])
    ax.grid(axis="x", visible=False)
    panel_label(ax, l)
axes[0].set_ylabel("|log EF change| (median, %)")
hd, lb = axes[0].get_legend_handles_labels()
fig.legend(hd, lb, loc="lower center", ncol=len(lb), fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("figs/fig7_violation.png")
plt.close(fig)

# ---- projections ----
pr = pd.read_csv(f"results/projections_{MODEL}.csv")
P = pd.read_csv("data/panel.csv")
src = ["coal", "oil", "gas", "flaring", "cement"]
countries = ["China", "United States", "India", "Russia", "Saudi Arabia", "Iran"]
fig = plt.figure(figsize=(7.4, 6.4))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.15], hspace=0.55, wspace=0.32)
for k, c in enumerate(countries):
    ax = fig.add_subplot(gs[k // 3, k % 3])
    hist = P[(P.country == c) & (P.year.between(2005, 2024))]
    tot = hist[[f"{s}_co2" for s in src]].sum(axis=1)
    ax.plot(hist.year, tot, color=INK, lw=1.4)
    d = pr[(pr.country == c) & (pr.source == "total")].sort_values("year")
    yrs = np.r_[2024, d.year.values]
    mid = np.r_[tot.iloc[-1], d.proj.values]
    lo = np.r_[tot.iloc[-1], d.lo90.values]
    hi = np.r_[tot.iloc[-1], d.hi90.values]
    ax.fill_between(yrs, lo, hi, color=SLOTS[0], alpha=0.18, lw=0)
    ax.plot(yrs, mid, color=SLOTS[0], lw=1.6, ls="--")
    ax.set_title(c, fontsize=8.5)
    ax.tick_params(labelsize=7)
    ax.set_xticks([2005, 2015, 2030])
    ax.grid(axis="x", visible=False)
    if k % 3 == 0:
        ax.set_ylabel("Mt CO$_2$", fontsize=7.5)
    if k == 0:
        panel_label(ax, "a")
ax = fig.add_subplot(gs[2, :])
pp = pd.read_csv(f"results/projections_panel_{MODEL}.csv")
hp = P[P.year.between(2005, 2024)].groupby("year")[[f"{s}_co2" for s in src]].sum()
base = pp[pp.year == pp.year.min()].set_index("source")
for s in src:
    proj = pp[pp.source == s].sort_values("year")
    yrs = np.r_[hp.index.values, proj.year.values]
    vals = np.r_[hp[f"{s}_co2"].values, proj.proj.values] / 1000
    ax.plot(hp.index, hp[f"{s}_co2"] / 1000, color=FUEL_COLORS[s], lw=1.6)
    ax.plot(np.r_[2024, proj.year], np.r_[hp[f"{s}_co2"].iloc[-1], proj.proj.values] / 1000,
            color=FUEL_COLORS[s], lw=1.6, ls="--")
    off = {"cement": 0.45, "flaring": -0.45}.get(s, 0.0)
    ax.text(2030.3, proj.proj.values[-1] / 1000 + off, FUEL_LABELS[s], fontsize=7.2, color=INK2, va="center")
ax.axvline(2024.5, color=MUTED, lw=0.8)
ax.text(2024.7, 2.4, "projection", fontsize=7, color=INK2, va="bottom")
ax.set_ylabel("Gt CO$_2$ (panel)")
ax.set_xlim(2005, 2032.5)
ax.grid(axis="x", visible=False)
panel_label(ax, "b")
fig.savefig("figs/fig9_projections.png")
plt.close(fig)
print("extra figures written")
