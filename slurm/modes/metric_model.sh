#!/usr/bin/env bash
# A metric-placed kernel ridge model (metric_kernel_fits.py model, linear q) and
# its stored validation.
#   PYTHON=... sbatch slurm/modes/metric_model.sh LAMBDA_MAX DATASET N GAMMA
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=2:55:00
#SBATCH --job-name=metric-model
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
L="$1" DATASET="$2" N="$3" GAMMA="$4"
S=visualization/metric_study
BASE="$S/release/lambda$L/runs/metric${N}_g${GAMMA}"
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
"$PYTHON" visualization/metric_kernel_fits.py model --dataset "$DATASET" --metric "$S/metric_teob_$L.npz" \
    --spec constant:cbrt:residual_shifted:sum:0.25 --n "$N" --gamma "$GAMMA" --out "$BASE/model"
OMP_NUM_THREADS=1 "$PYTHON" visualization/modes_scale.py validate "$DATASET/validation" --model "$BASE/model" --jobs "$SLURM_CPUS_PER_TASK" \
    --n 8192 --out "$S/release/lambda$L/runs/validate_metric${N}_g${GAMMA}.npz"
