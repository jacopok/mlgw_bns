#!/usr/bin/env bash
# One shard of the full waveform mismatches of the saved predictions of
# metric_kernel_fits.py (what is done is skipped).
#   PYTHON=... sbatch --array=0-K slurm/modes/metric_full.sh FITS_DIR DATASET REFERENCE...
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=2:55:00
#SBATCH --job-name=metric-full
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export SCRIPT=visualization/metric_kernel_fits.py
out="$1" dataset="$2"; shift 2
exec slurm/modes/metric_study.sh full "$out" --dataset "$dataset" --jobs 16 \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --shards "${SLURM_ARRAY_TASK_COUNT:-1}" --reference "$@"
