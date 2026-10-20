#!/bin/bash
# Full rolling-origin run, in chunks that each finish well inside 30 minutes.
# Usage: src/run_full.sh <chunk-name>   (skips a chunk whose output already exists)
# Settings fixed from development tuning: XGB depth 5 / 300 trees; NN width/dropout/lambda from NN_* below.
set -e
cd "$(dirname "$0")/.."
mkdir -p results/full
NN_W=${NN_W:-64}; NN_P=${NN_P:-0.1}; NN_LAM=${NN_LAM:-1.0}
run() { out=$1; shift; if [ -f "$out" ]; then echo "skip $out"; else "$@" --out "$out"; fi; }
case "$1" in
  xgb_cal)   run results/full/xgb_1995_2007.pkl python3 -I src/xgb_models.py --origins 1995-2007 --seeds 0 --depth 5 --n_est 300 ;;
  xgb_t1)    run results/full/xgb_2008_2011.pkl python3 -I src/xgb_models.py --origins 2008-2011 --seeds 0,1,2 --depth 5 --n_est 300 ;;
  xgb_t2)    run results/full/xgb_2012_2015.pkl python3 -I src/xgb_models.py --origins 2012-2015 --seeds 0,1,2 --depth 5 --n_est 300 ;;
  xgb_t3)    run results/full/xgb_2016_2019.pkl python3 -I src/xgb_models.py --origins 2016-2019 --seeds 0,1,2 --depth 5 --n_est 300 ;;
  nn_*)      v=${1#nn_}; v=${v%%_*}; part=${1##*_}
             lam=0; [ "$v" = "S" ] && lam=$NN_LAM
             case $part in
               cal) run results/full/nn_${v}_1995_2007.pkl python3 -I src/nn_models.py --variant $v --origins 1995-2007 --seeds 0 --width $NN_W --dropout $NN_P --lam $lam ;;
               t1)  run results/full/nn_${v}_2008_2013.pkl python3 -I src/nn_models.py --variant $v --origins 2008-2013 --seeds 0,1,2 --width $NN_W --dropout $NN_P --lam $lam ;;
               t2)  run results/full/nn_${v}_2014_2019.pkl python3 -I src/nn_models.py --variant $v --origins 2014-2019 --seeds 0,1,2 --width $NN_W --dropout $NN_P --lam $lam ;;
             esac ;;
esac
echo "CHUNK_DONE $1"
