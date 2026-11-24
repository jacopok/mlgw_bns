#!/usr/bin/env bash
# Prediction times of models with kernel ridge on N points (truncations of a
# shipped model, metric_kernel_fits.py truncate), against the shipped one, on
# one whole node.   PYTHON=... sbatch slurm/modes/metric_timing.sh OUT_DIR
#SBATCH --partition=short
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=2:55:00
#SBATCH --job-name=metric-timing
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT="$1"
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 MPLBACKEND=Agg TQDM_MININTERVAL=60
"$PYTHON" visualization/metric_kernel_fits.py truncate "$OUT"
MODELS=() PATHS=()
for n in 1024 2048 4096 8192 16384; do MODELS+=("N$n=$OUT/n$n/model"); PATHS+=("$OUT/n$n/model"); done
for points in 1000 4000; do
    "$PYTHON" visualization/predict_time_by_model.py "$points" default_hom_lambda2000 "${PATHS[@]}" | tee "$OUT/predict_$points.txt"
done
"$PYTHON" visualization/benchmark_evaluation_time.py --models shipped=default_hom_lambda2000 "${MODELS[@]}" \
    --variants numpy,jax,jax-batch --n-grid 7 --log-min 2 --log-max 5 --seeds 5 --epochs 4 \
    --jax-batch 64 --max-batch-points 1e5 --out "$OUT/timing"
