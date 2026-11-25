#!/usr/bin/env bash
# The release benchmark of one model (validate_model.py against TEOBResumS), some of its steps.
#   PYTHON=... sbatch slurm/modes/metric_bench.sh MODEL_BASE PREFIX N STEP...
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --job-name=metric-bench
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
MODEL="$1" PREFIX="$2" N="$3"; shift 3
mkdir -p "$(dirname "$PREFIX")"
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 MPLBACKEND=Agg JAX_PLATFORMS=cpu OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
"$PYTHON" -c "import mlgw_bns; print(mlgw_bns.__file__)"
echo "$(hostname) $MODEL $N $*"
exec "$PYTHON" visualization/validate_model.py --model "$MODEL" --prefix "$PREFIX" --n-mismatches "$N" --steps "$@"
