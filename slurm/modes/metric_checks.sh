#!/usr/bin/env bash
# Corner boxes and merger times against TEOBResumS of several models with one
# Lambda range: the corners where earlier candidates regressed (q ~ 2-3 with a
# large Lambda_1 and small Lambda_2; small Lambda_1) and the whole box.
#   PYTHON=... sbatch slurm/modes/metric_checks.sh LAMBDA_MAX MODEL_BASE...   (absolute bases)
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --job-name=metric-checks
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
ROOT="$PWD" L="$1"; shift
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 MPLBACKEND=Agg JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 TQDM_DISABLE=1
OUT="visualization/metric_study/release/lambda$L/checks"
mkdir -p "$OUT"; cd "$OUT"
echo "box: large Lambda_1, small Lambda_2"
"$PYTHON" "$ROOT/visualization/corner_scan.py" 400 1.8 3.0 $((L * 6 / 10)) "$L" 0 $((L / 20)) "$@"
echo "box: small Lambda_1"
"$PYTHON" "$ROOT/visualization/corner_scan.py" 400 1.0 3.0 0 50 0 "$L" "$@"
echo "box: near equal mass"
"$PYTHON" "$ROOT/visualization/corner_scan.py" 400 1.0 1.15 0 "$L" 0 "$L" "$@"
echo "merger times"
"$PYTHON" "$ROOT/visualization/merger_time_release.py" 4000 "$@"
