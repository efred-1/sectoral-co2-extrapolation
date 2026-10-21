"""Data figures: panel coverage and fuel composition (Fig. 2), stoichiometric diagnostic (Fig. 3)."""
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

sys.path.insert(0, "src")
from style import FUEL_COLORS, FUEL_LABELS, INK2, MUTED, SEQ_BLUE, GRID, panel_label  # noqa: E402

P = pd.read_csv("cleaned_data/energy_matched_panel_42countries_1965-2024.csv")
S = pd.read_csv("cleaned_data/panel_countries.csv")
D = pd.read_csv("cleaned_data/emission_factor_audit.csv")
world = gpd.read_file(sys.argv[1])

# ---------------- Fig 2: coverage map + fuel composition ----------------
fig = plt.figure(figsize=(7.2, 6.6))
ax1 = fig.add_axes([0.0, 0.50, 1.0, 0.48])
ax2 = fig.add_axes([0.08, 0.06, 0.80, 0.31])
iso = "ADM0_A3" if "ADM0_A3" in world.columns else "ISO_A3"
w = world.merge(S[["iso_code", "share_world_2022_pct"]], left_on=iso, right_on="iso_code", how="left")
w = w[w["NAME"] != "Antarctica"] if "NAME" in w.columns else w
bins = [0, 0.25, 0.5, 1, 2, 5, 40]
cmap = LinearSegmentedColormap.from_list("b", SEQ_BLUE[1:])
norm = BoundaryNorm(bins, cmap.N)
w[w.share_world_2022_pct.isna()].plot(ax=ax1, color="#f0efec", edgecolor="white", linewidth=0.3)
w[w.share_world_2022_pct.notna()].plot(ax=ax1, column="share_world_2022_pct", cmap=cmap, norm=norm,
                                        edgecolor="white", linewidth=0.3)
ax1.set_axis_off()
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cax = fig.add_axes([0.30, 0.505, 0.40, 0.016])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=bins)
cb.ax.set_xticklabels(["0", "0.25", "0.5", "1", "2", "5", ">5"])
cb.set_label("Share of world fossil CO$_2$ in 2022 (%)", color=INK2, fontsize=8)
cb.outline.set_visible(False)
ax1.text(0.0, 1.0, "(a)", transform=ax1.transAxes, fontsize=10, fontweight="bold", va="top")
ax1.text(0.04, 1.0, f"42 panel countries, {S.share_world_2022_pct.sum():.1f}% of world fossil CO$_2$ (grey: not in panel)",
         transform=ax1.transAxes, fontsize=8.5, va="top", color=INK2)

tot = P[P.year.between(1970, 2024)].groupby("year")[["coal_co2", "oil_co2", "gas_co2", "flaring_co2", "cement_co2"]].sum() / 1000
fuels = ["coal", "oil", "gas", "flaring", "cement"]
ax2.stackplot(tot.index, [tot[f"{f}_co2"] for f in fuels], colors=[FUEL_COLORS[f] for f in fuels],
              edgecolor="white", linewidth=0.6)
cum = tot.cumsum(axis=1)
label_y = {}
for f in fuels:
    label_y[f] = cum[f"{f}_co2"].iloc[-1] - tot[f"{f}_co2"].iloc[-1] / 2
label_y["flaring"] = label_y["flaring"] - 1.4   # spread the two thin top layers
label_y["cement"] = label_y["cement"] + 0.9
for f in fuels:
    y_mid = cum[f"{f}_co2"].iloc[-1] - tot[f"{f}_co2"].iloc[-1] / 2
    ax2.annotate(f"{FUEL_LABELS[f]} {tot[f'{f}_co2'].iloc[-1]:.1f}", xy=(2024, y_mid), xytext=(2026.5, label_y[f]),
                 fontsize=7.5, color=INK2, va="center", annotation_clip=False,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
# separate overlapping small labels
ax2.set_xlim(1970, 2024)
ax2.set_ylabel("Gt CO$_2$ per year (panel total)")
ax2.grid(axis="x", visible=False)
panel_label(ax2, "b")
ax2.set_title("Fossil CO$_2$ of the 42-country panel by source, 1970–2024", loc="left", fontsize=9, pad=8, x=0.02)
fig.savefig("figs/fig2_panel.png")
plt.close(fig)

# ---------------- Fig 3: stoichiometric diagnostic ----------------
fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), sharey=False)
for a in axes: a.set_xlim(0.05, 0.75)
for ax, f in zip(axes, ["coal", "oil", "gas"]):
    s = D[D.fuel == f]
    lo, hi = s.band_lo.iloc[0], s.band_hi.iloc[0]
    bins_ = np.linspace(0.05, 0.6, 56)
    vals = s.ef.clip(0.05, 0.6)  # values beyond 0.6 shown in last bin
    ax.hist(vals, bins=bins_, color=FUEL_COLORS[f], alpha=0.9, edgecolor="white", linewidth=0.4)
    ax.axvspan(lo, hi, color="#cfcdc4", alpha=0.45, zorder=0, lw=0)
    ax.axvline(hi, color=INK2, lw=0.9, ls="--")
    within = (s.band_class == "within").mean() * 100
    above = (s.band_class == "above_upper").mean() * 100
    below = (s.band_class == "below_lower").mean() * 100
    ax.set_title(f"{FUEL_LABELS[f]}", loc="left")
    ax.text(0.98, 0.97, f"within band {within:.0f}%\nabove max. {above:.0f}%\nbelow band {below:.0f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.0, color=INK2,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5))
    ax.set_xlabel("Implied emission factor (t CO$_2$/MWh)")
    ax.grid(axis="x", visible=False)
    ax.set_xlim(0.05, 0.75)
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
axes[0].set_ylabel("Country-years")
for ax, letter in zip(axes, "abc"):
    panel_label(ax, letter)
fig.tight_layout()
fig.savefig("figs/fig3_ef_diagnostic.png")
plt.close(fig)

# gas: share of years above physical maximum, by country (top 12) -> supplementary panel
g = D[D.fuel == "gas"]
share = (g.band_class == "above_upper").groupby(g.country).mean().sort_values(ascending=False).head(12) * 100
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.barh(share.index[::-1], share.values[::-1], color=FUEL_COLORS["gas"], height=0.6)
for i, v in enumerate(share.values[::-1]):
    ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=7.5, color=INK2)
ax.set_xlabel("Share of years with gas emission factor above IPCC maximum (%)")
ax.set_xlim(0, 110)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig("figs/figS1_gas_above_max.png")
plt.close(fig)
print("data figures written")
