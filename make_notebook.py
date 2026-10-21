"""Build the companion notebook (reproduce_and_explore.ipynb) that walks through every result in the paper."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s))


def code(s):
    C.append(nbf.v4.new_code_cell(s))


md("""# Physics-Informed Extrapolation of Sectoral Fossil CO₂ Emissions in the Energy Transition: Stoichiometry as a Post-Forecast Correction
Companion notebook. It loads the saved outputs of the pipeline and reproduces every table and figure in the paper.
Heavy steps (model training, about 4–5 hours on two cores) are run by `run_all.sh`; the last section shows how to
re-run the light steps (evaluation, projections, figures) from here.""")
code("""import json, os
import pandas as pd
from IPython.display import Image, display
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
def fig(name, width=900):
    display(Image(filename=f"figs/{name}", width=width))""")

md("## 1. Data and the stoichiometric audit")
code("""panel = pd.read_csv("cleaned_data/energy_matched_panel_42countries_1965-2024.csv")
countries = pd.read_csv("cleaned_data/panel_countries.csv")
print(f"{countries.shape[0]} countries, {len(panel):,} country-years; share of world fossil CO2 in 2022: "
      f"{countries.share_world_2022_pct.sum():.1f}%")
countries.sort_values("co2_2022_mt", ascending=False).head(10)""")
code("""D = pd.read_csv("cleaned_data/emission_factor_audit.csv")
(D.groupby("fuel").band_class.value_counts(normalize=True).unstack() * 100).round(1)""")
code("""fig("fig2_panel.png"); fig("fig3_ef_diagnostic.png"); fig("figS1_gas_above_max.png", 600)""")

md("## 2. Development period and the recorded model-selection decision")
code("""print(open("results/dev/dev_scores.txt").read())
pre = json.load(open("results/dev/preregistration.json"))
print(json.dumps({k: pre[k] for k in ["recorded_at", "basis", "decision"]}, indent=1))""")
code("""json.load(open("results/dev/nn_choice.json"))""")

md("## 3. Test-period accuracy (Table 1, Fig. 4)")
code("""S = pd.read_csv("results/summary_table.csv").set_index("model")
S[["mase_1", "mase_2-3", "mase_4-6", "mase_7-10", "mase_all", "rel_all", "masetot_all", "reltot_all"]].round(3)""")
code("""R = pd.read_csv("results/covid_sensitivity.csv").set_index("model")
R[["mdase_all", "n_se_over_50", "share_err_over_50"]].round(3)   # median scaled error and exploding forecasts""")
code("""fig("fig4_accuracy_horizon.png")""")

md("## 4. Effect of the stoichiometric constraint (Table 2, Figs. 5–6)")
code("""T = pd.read_csv("results/significance_tests.csv")
T["cell"] = T.apply(lambda r: f"{r.rel_change_pct:+.1f}% [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}] p_Holm={r.p_holm:.3f}", axis=1)
T[T.target == "sectors"].pivot_table(index=["a", "b"], columns="horizons", values="cell", aggfunc="first", sort=False)""")
code("""pd.read_csv("results/constraint_by_source.csv").pivot(index=["a", "b"], columns="source", values="rel_change_pct").round(1)""")
code("""fig("fig5_physics_effect.png"); fig("fig6_source_heatmap.png")""")

md("## 5. Why the projection helps (Fig. 7) and fuel-energy accuracy")
code("""V = pd.read_csv("results/violation_by_horizon.csv")
t = V.pivot_table(index=["fuel", "h"], columns="model", values="implied_med") * 100
t["observed"] = V.groupby(["fuel", "h"]).observed_med.first() * 100
t.round(1)""")
code("""E = pd.read_csv("results/energy_accuracy.csv")
E.pivot(index="model", columns="group", values="mase_energy").round(2)""")
code("""fig("fig7_violation.png")""")

md("## 5b. Supplementary: weighted correction (Table A.4) and energy-to-CO₂ conversion (Table 6)")
code("""pd.read_csv("results/weighted_tests.csv").query("horizons == 'all'")[["model", "versus", "rel_change_pct", "ci_lo", "ci_hi", "n_better"]].round(2)""")
code("""fc = pd.read_csv("results/foresight_conversion.csv")
fc[fc.h.astype(str) == "all"].pivot_table(index=["fuel", "group"], columns="variant", values=["mdape", "median_bias"], sort=False).round(1)""")

md("## 6. Robustness, seeds and prediction intervals (Tables 3–4, Fig. 8)")
code("""pd.read_csv("results/seed_variability.csv").round(3)""")
code("""pd.read_csv("results/conformal_overall.csv").round(3)""")
code("""fig("fig8_intervals.png")""")

md("## 7. Projections to 2030 (Table 5, Fig. 9)")
code("""pp = pd.read_csv("results/projections_panel_COMB-P.csv")
pp.pivot(index="year", columns="source", values="proj").round(0)""")
code("""pr = pd.read_csv("results/projections_COMB-P.csv")
chk = pr[(pr.source == "total") & (pr.year == 2025) & pr.country.isin(["China", "India", "United States"])].copy()
chk["proj_change_%"] = 100 * (chk.proj / chk.base_2024 - 1)
chk["lo90_%"] = 100 * (chk.lo90 / chk.base_2024 - 1); chk["hi90_%"] = 100 * (chk.hi90 / chk.base_2024 - 1)
chk["GCB_2025_preliminary_%"] = chk.country.map({"China": 0.4, "India": 1.4, "United States": 1.9})
chk[["country", "proj_change_%", "lo90_%", "hi90_%", "GCB_2025_preliminary_%"]].round(1)""")
code("""fig("fig9_projections.png")""")

md("""## 8. Re-running the light steps
Uncomment to recompute the evaluation, projections and figures from the saved model outputs
(about 2 minutes; projections retrain XGBoost on all data, about 1 minute).""")
code("""# !python3 -I src/evaluate.py
# !python3 -I src/project.py COMB-P XGB-P damped-P
# !python3 -I src/fig_results.py && python3 -I src/fig_extra.py COMB-P""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "reproduce_and_explore.ipynb")
print("notebook written")
