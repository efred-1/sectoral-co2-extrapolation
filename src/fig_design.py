"""Fig. 1: study design. (a) modelling framework, (b) rolling-origin evaluation timeline."""
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, "src")
from style import SLOTS, INK, INK2, MUTED, AXIS, panel_label  # noqa: E402

fig = plt.figure(figsize=(7.4, 7.6))
ax = fig.add_axes([0.0, 0.40, 1.0, 0.58])
ax.set_xlim(0, 100)
ax.set_ylim(0, 65)
ax.axis("off")


def box(x, y, w, h, title, body, edge, fill="#ffffff"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                linewidth=1.1, edgecolor=edge, facecolor=fill))
    ax.text(x + 1.2, y + h - 1.6, title, fontsize=8.2, fontweight="bold", color=INK, va="top")
    ax.text(x + 1.2, y + h - 5.2, body, fontsize=7.1, color=INK2, va="top", linespacing=1.35)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0, shrinkA=0, shrinkB=0))


box(1, 44, 30, 17, "Data (42 countries, 1965–2024)",
    "Fossil CO$_2$ by source: coal, oil, gas,\nflaring, cement (Global Carbon\nBudget 2025)\nFuel energy: coal, oil, gas; low-carbon\nshare (Energy Institute 2025)", AXIS)
box(35, 44, 30, 17, "Features at origin τ",
    "5-year window of CO$_2$ by source\nand fuel energy (levels, growth),\nEF$_f$ at τ, low-carbon share,\npopulation growth, horizon h\nTarget: log growth τ → τ+h", AXIS)
box(69, 44, 30, 17, "Combustion stoichiometry",
    "CO$_{2,f}$ = EF$_f$ × E$_f$ for f = coal, oil, gas\nEF$_f$: carbon released per MWh,\nset by fuel chemistry\nIPCC 2006 bands used to audit\ninventory–energy consistency", SLOTS[1])
arrow(31, 52.5, 35, 52.5)

box(1, 19, 30, 21, "Local statistical forecasters",
    "Naive (random walk)\nShort-run log trend\nDamped-trend exponential\nsmoothing, fitted to CO$_2$ and\nto fuel energy\n(one model per country-series)", MUTED)
box(35, 19, 30, 21, "Global ML and combination",
    "XGB-E: emissions history only\nXGB-D: + energy features\nNN-U: multi-task network,\nCO$_2$ and fuel energy jointly\n(pooled across 42 countries)\nCOMB: mean of damped and XGB-D", SLOTS[0])
box(69, 19, 30, 21, "Stoichiometric layer",
    "Any forecaster of CO$_2$ and energy\nHard (H): CO$_2$ = EF$_τ$ × Ê\nSoft (S): penalty on implied\nEF drift in network training\nPost-hoc (P): projection onto\nln CO$_2$ − ln E = ln EF$_τ$", SLOTS[2])
arrow(16, 44, 16, 40.6)
arrow(50, 44, 50, 40.6)
arrow(84, 44, 84, 40.6)
arrow(31, 29.5, 35, 29.5)
arrow(65, 29.5, 69, 29.5)

box(1, 1, 98, 14, "Evaluation and outputs",
    "Rolling forecast origins 2008–2019, horizons h = 1–10 years (targets 2009–2024); 3 seeds per ML model\n"
    "MASE by source and for the national total; Wilcoxon signed-rank tests across countries with Holm correction;\n"
    "country-cluster bootstrap intervals; split-conformal prediction intervals; projections 2025–2030", AXIS, fill="#fafaf8")
for x in (16, 50, 84):
    arrow(x, 19, x, 15.6)
ax.text(0.2, 64.8, "(a)", fontsize=10, fontweight="bold", va="top")

# (b) rolling-origin timeline
bx = fig.add_axes([0.10, 0.05, 0.86, 0.29])
rows = [(1998, "dev"), (2002, "dev"), (2008, "test"), (2012, "test"), (2016, "test"), (2019, "test")]
for i, (T0, kind) in enumerate(rows):
    y = len(rows) - i
    bx.barh(y, T0 - 1970 + 1, left=1969.5, height=0.5, color="#d9e8fb", edgecolor="none")
    end = min(T0 + 10, 2024 if kind == "test" else 2007)
    col = SLOTS[1] if kind == "test" else SLOTS[3]
    bx.barh(y, end - T0, left=T0 + 0.5, height=0.5, color=col, edgecolor="none")
    bx.text(1968.5, y, f"origin {T0}", ha="right", va="center", fontsize=7.5, color=INK2)
bx.axvline(2007.5, color=INK2, lw=0.8, ls="--")
bx.text(2006.9, len(rows) + 0.75, "development", ha="right", fontsize=7.2, color=INK2)
bx.text(2008.1, len(rows) + 0.75, "test", ha="left", fontsize=7.2, color=INK2)
bx.set_xlim(1969, 2025)
bx.set_ylim(0.3, len(rows) + 1.1)
bx.set_yticks([])
bx.set_xlabel("Year")
bx.grid(axis="y", visible=False)
bx.spines["left"].set_visible(False)
from matplotlib.patches import Patch  # noqa: E402
bx.legend(handles=[Patch(color="#d9e8fb", label="training targets (≤ origin)"),
                   Patch(color=SLOTS[3], label="tuning forecasts (targets ≤ 2007)"),
                   Patch(color=SLOTS[1], label="test forecasts (h = 1–10, targets ≤ 2024)")],
          loc="lower left", fontsize=7, ncol=3, bbox_to_anchor=(0.0, 1.0), handlelength=1.4, columnspacing=1.2)
bx.text(-0.11, 1.17, "(b)", transform=bx.transAxes, fontsize=10, fontweight="bold", va="top")
fig.savefig("figs/fig1_design.png")
print("design figure written")
