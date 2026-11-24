#!/usr/bin/env bash
# One line of a task file (the arguments of metric_kernel_fits.py) per array
# task; finished fits are skipped, so that resubmitting continues.
#   PYTHON=... sbatch --array=0-K%6 slurm/modes/metric_fits.sh TASKS
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=2:55:00
#SBATCH --job-name=metric-fits
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mapfile -t TASKS < "$1"
read -r -a ARGUMENTS <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
export SCRIPT=visualization/metric_kernel_fits.py
exec slurm/modes/metric_study.sh "${ARGUMENTS[@]}"
