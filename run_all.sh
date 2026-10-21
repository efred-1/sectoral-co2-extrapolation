#!/bin/bash
# Full reproduction pipeline. Run from the project root. Total run time on 2 CPU cores: about 4-5 hours,
# almost all of it in the neural-network runs. Every step skips work whose output already exists only
# where noted (run_full.sh chunks); delete results/ to start from scratch.
set -e
cd "$(dirname "$0")"
mkdir -p results/dev results/full figs logs

# 1. Data: energy-matched panel, country list, emission-factor diagnostic
python3 -I src/build_data.py raw_data/owid-co2-data.csv raw_data/owid-energy-data.csv

# 2. Samples (country x origin x horizon) and MASE scales
python3 -I src/make_samples.py

# 3. Local statistical baselines (naive, short-run trend, damped trend) for CO2 and fuel energy, origins 1995-2019
python3 -I src/baselines.py

# 4. Development-period tuning (origins 1998-2002, scored on targets <= 2007 only)
for cfg in "3 300" "3 600" "5 300" "5 600"; do set -- $cfg
  python3 -I src/xgb_models.py --origins 1998-2002 --seeds 0 --depth $1 --n_est $2 --out results/dev/xgb_d$1_n$2.pkl
done
for w in 64 128; do for p in 0.1 0.3; do
  python3 -I src/nn_models.py --variant U --origins 1998-2002 --seeds 0 --width $w --dropout $p --out results/dev/nnU_w${w}_p${p}.pkl
done; done
python3 -I src/select_dev.py nn
W=$(python3 -c "import json;print(json.load(open('results/dev/nn_choice.json'))['width'])")
P=$(python3 -c "import json;print(json.load(open('results/dev/nn_choice.json'))['dropout'])")
for lam in 0.1 1.0 10.0; do
  python3 -I src/nn_models.py --variant S --origins 1998-2002 --seeds 0 --width $W --dropout $P --lam $lam --out results/dev/nnS_lam${lam}.pkl
done
python3 -I src/select_dev.py lam
LAM=$(python3 -c "import json;print(json.load(open('results/dev/nn_choice.json'))['lam'])")
python3 -I src/score_dev.py | tee results/dev/dev_scores.txt
# (The model-selection decision recorded in results/dev/preregistration.json was taken at this point,
#  from development scores only, before step 6.)

# 5. Rolling-origin runs at every origin 1995-2019 (3 seeds at test origins 2008-2019)
for c in xgb_cal xgb_t1 xgb_t2 xgb_t3; do ./src/run_full.sh $c; done
for v in U S H; do for part in cal t1 t2; do NN_W=$W NN_P=$P NN_LAM=$LAM ./src/run_full.sh nn_${v}_${part}; done; done

# 6. Test-period evaluation, projections and figures
python3 -I src/evaluate.py
python3 -I src/supplementary.py   # weighted correction and energy-to-CO2 conversion test (added after test evaluation)
python3 -I src/project.py COMB-P XGB-P damped-P
python3 -I src/fig_design.py
python3 -I src/fig_data.py raw_data/ne_110m_countries.geojson
python3 -I src/fig_results.py
python3 -I src/fig_extra.py COMB-P

# 7. Data dictionary (DATA_DICTIONARY.md, results/data_dictionary.csv)
python3 -I src/make_data_dictionary.py
echo "ALL DONE"
