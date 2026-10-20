"""Shared figure style (print, light surface). Palette: validated categorical slots in fixed order."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FUEL_COLORS = dict(zip(["coal", "oil", "gas", "flaring", "cement"], SLOTS[:5]))
FUEL_LABELS = {"coal": "Coal", "oil": "Oil", "gas": "Gas", "flaring": "Flaring", "cement": "Cement"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGE = ("#2a78d6", "#f0efec", "#e34948")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300,
    "savefig.bbox": "tight", "legend.frameon": False, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def panel_label(ax, letter):
    ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="right", color=INK)

# Model families: one colour each (validated all-pairs for CVD and normal vision); constrained variants share the family colour.
FAMILY_COLORS = {"damped": "#2a78d6", "XGB": "#eda100", "NN": "#008300", "COMB": "#e87ba4"}
FAMILY_MARKERS = {"damped": "o", "XGB": "s", "NN": "^", "COMB": "D"}


def family(m):
    return "damped" if m.startswith("damped") else m.split("-")[0]


def model_style(m):
    if m == "naive":
        return dict(color=MUTED, marker="", ls="--")
    f = family(m)
    return dict(color=FAMILY_COLORS[f], marker=FAMILY_MARKERS[f], ls="-")
