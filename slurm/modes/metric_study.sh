#!/usr/bin/env bash
# One command of visualization/metric_kernel_study.py on a Draco node, with the
# repository this is submitted from (PYTHONPATH, so that a worktree can share
# the environment of the main clone: PYTHON=~/mlgw_bns/.venv/bin/python).
#
#   PYTHON=... sbatch -o visualization/metric_study/%x-%j.log slurm/modes/metric_study.sh COMMAND [OPTIONS]
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --job-name=metric-study
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
PYTHON="${PYTHON:-.venv/bin/python}"
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 MPLBACKEND=Agg JAX_PLATFORMS=cpu OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "$(hostname) $*"
"$PYTHON" -c "import mlgw_bns; print(mlgw_bns.__file__)"
exec "$PYTHON" visualization/metric_kernel_study.py "$@"
