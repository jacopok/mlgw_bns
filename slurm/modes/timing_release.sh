#!/usr/bin/env bash
# Prediction wall-clock times of several models, on one whole node so that
# every timing shares the same hardware and nothing else runs beside them:
# visualization/benchmark_evaluation_time.py --models, one grid size at a time
# in a random order, every trial of every
# (model, method, draw, repeat) at it shuffled, on 50
# grid sizes from 100 to 1e5 points, with a c1 + c2 N fit per curve:
# numpy and JAX Model.predict (one waveform a call, and a jax.vmap batch of
# 64), and the precessing model's JAX predictor (one binary, a batch of 64).
# TEOBResumS-SPA is timed in the first sweep and read back from
# $OUT_DIR/teob.npz afterwards.
#
#   sbatch -o visualization/modes_scale/timing/%j.log slurm/modes/timing_release.sh \
#       NAME LABEL=MODEL [LABEL=MODEL ...]
#
# MODEL: a shipped model's name or a trained model's base filename
# (.../runs/<run>/model). Writes $OUT_DIR/NAME.npz (every timing) and
# $OUT_DIR/NAME_<method>.png. Run from the repository.
#SBATCH --partition=standard
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --job-name=timing-release
set -euo pipefail
[[ $# -ge 2 ]] || { sed -n '2,20p' "$0"; exit 1; }
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
NAME="$1"; shift
OUT_DIR="${OUT_DIR:-visualization/modes_scale/timing}"
mkdir -p "$OUT_DIR"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 TQDM_MININTERVAL=60
echo "node $(hostname), $(nproc) cores, $(lscpu | sed -n 's/^Model name: *//p')"
"${PYTHON:-.venv/bin/python}" visualization/benchmark_evaluation_time.py --models "$@" \
    --n-grid "${N_GRID:-50}" --log-min 2 --log-max 5 --seeds 10 --epochs 5 \
    --jax-batch 64 --max-batch-points 1e5 \
    --teob-cache "$OUT_DIR/teob.npz" --out "$OUT_DIR/$NAME"
