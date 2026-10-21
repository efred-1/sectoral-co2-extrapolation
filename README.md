# Physics-Informed Extrapolation of Sectoral Fossil CO₂ Emissions in the Energy Transition

**Stoichiometry as a post-forecast correction, tested on 42 economies, 1965–2024**

This repository holds the cleaned data and the complete analysis code for the paper of the same title. The paper is not yet published; its full reference will be added here once it is.

The study pairs Global Carbon Budget 2025 emissions by source (coal, oil, gas, flaring, cement) with Energy Institute fuel consumption for 42 economies that emit 81.2% of world fossil CO₂. It compares statistical, machine-learning and combined forecasters one to ten years ahead over 12 rolling forecast origins (2008–2019), and tests three ways of imposing combustion stoichiometry, CO₂ = EF × E: built into the model, as a training penalty, and as a correction applied after forecasting.

## Files

| File / folder | Contents |
|---|---|
| `cleaned_data/` | **The cleaned data.** |
| `cleaned_data/energy_matched_panel_42countries_1965-2024.csv` | Energy-matched panel: 42 economies × 60 years (1965–2024), 2,520 rows. CO₂ by source (Mt CO₂/yr), fuel consumption (TWh/yr), low-carbon share and implied emission factors (t CO₂/MWh). |
| `cleaned_data/panel_countries.csv` | The 42 countries, their complete years and their share of world fossil CO₂ in 2022. |
| `cleaned_data/emission_factor_audit.csv` | Every implied emission factor, 1970–2024, classed against the IPCC 2006 range (within, above, below). |
| `DATA_DICTIONARY.md` | Every column of the cleaned data and the projection files: meaning, unit, source, observed range and missing values. Also as `results/data_dictionary.csv`. |
| `reproduce_and_explore.ipynb` | Executed notebook that walks through the data, the emission-factor audit, the development-period choices, every main result and the projections, from the saved results. Built by `make_notebook.py`. |
| `raw_data/` | Raw inputs the cleaned data are built from: the Our World in Data CO₂ and energy datasets with their codebooks, and Natural Earth country outlines for the map. |
| `src/` | All analysis code (pipeline below). |
| `run_all.sh` | Reruns the whole study from the raw inputs. |
| `results/*.csv` | Every result reported in the paper: accuracy by horizon and source, paired tests, robustness checks, prediction intervals, supplementary analyses and projections. |
| `results/projections_*.csv` | Extrapolations to 2030 from three methods. They are extrapolations of history, not policy scenarios; see `results/README_Scenarios.txt`. |
| `results/dev/` | Development-period runs and scores, and the time-stamped model-selection record (`preregistration.json`). |
| `results/full/` | Rolling-origin forecasts of the machine-learning models at every origin. |
| `figs/` | The figures of the paper. |
| `requirements.txt` | The Python packages the code needs. |

## Running the code

Tested with Python 3.13.

```
pip install -r requirements.txt
jupyter notebook reproduce_and_explore.ipynb
```

The notebook reads the saved results and runs in a few seconds. Run it from the repository root so that it finds `cleaned_data/`, `results/` and `figs/`.

To rerun the whole study from the raw inputs:

```
./run_all.sh
```

This takes about 4–5 hours on two CPU cores, almost all of it for the neural networks; no GPU is needed. The steps are:

1. `src/build_data.py` builds the cleaned data in `cleaned_data/` from the two Our World in Data files in `raw_data/`.
2. `src/make_samples.py` builds the country × origin × horizon samples (49 features, log-growth targets) and the MASE scales.
3. `src/baselines.py` fits the naive, short-run trend and damped-trend forecasts of CO₂ and fuel energy.
4. `src/xgb_models.py`, `src/nn_models.py`, `src/score_dev.py` and `src/select_dev.py` tune the models on development forecasts only (origins 1998–2002, target years up to 2007).
5. `src/run_full.sh` refits XGBoost and the multi-task networks at every origin from 1995 to 2019, with three seeds at the test origins.
6. `src/evaluate.py` runs the test-period evaluation (origins 2008–2019): accuracy, paired tests with a country bootstrap, robustness checks and split-conformal intervals. `src/supplementary.py` runs the two supplementary analyses, `src/project.py` the projections to 2030, and `src/fig_*.py` draws the figures.
7. `src/make_data_dictionary.py` writes the data dictionary.

`src/assemble.py` holds the stoichiometric variants (hard, post-forecast correction, IPCC band, drift) and `src/evallib.py` the combination and the error measures.

Tested with numpy 2.5, pandas 3.0, scipy 1.18, statsmodels 0.15, xgboost 3.4, torch 2.14, matplotlib 3.11 and geopandas 1.2. Neural-network results can differ slightly on other hardware or library versions.

## How the evaluation was kept honest

- No test-period outcome influenced any modelling choice. Hyperparameters were chosen on development forecasts (origins 1998–2002, targets up to 2007).
- The combination (COMB) and the stoichiometric variants of the damped trend were added after the development results and before the test evaluation. The decision and the development scores are recorded in `results/dev/preregistration.json`.
- The paired test in the main text (log error ratios) replaced the planned test (mean scaled errors) after the results showed that a few exploding damped-trend forecasts dominate mean scaled errors. Both are reported.
- The weighted correction and the energy-to-CO₂ conversion test (`src/supplementary.py`) were added after the test results had been seen and did not change any model in the main comparison.

## Data sources

- Fossil CO₂ by source: Friedlingstein P, O'Sullivan M, Jones MW, Andrew RM, Bakker DCE, Hauck J, et al. Global Carbon Budget 2025. *Earth System Science Data* 2026;18:3211–88. https://doi.org/10.5194/essd-18-3211-2026
- Fuel consumption: Energy Institute. *Statistical Review of World Energy 2025*. London: Energy Institute; 2025.
- Both were taken from the Our World in Data compilations, https://github.com/owid/co2-data and https://github.com/owid/energy-data (accessed October 2026). Total primary energy in that compilation combines Energy Institute and U.S. Energy Information Administration data.
- Emission-factor ranges: 2006 IPCC Guidelines for National Greenhouse Gas Inventories, Volume 2 (Energy), Chapter 1, Table 1.4.
- Country outlines: Natural Earth 1:110m admin-0 countries (public domain).

Our World in Data publishes its compilations under a CC BY 4.0 licence; the underlying data remain subject to the terms of their original providers. The cleaned files here are derived from these sources. If you reuse them, please cite the original data sets and the paper.
