#!/usr/bin/env bash
# TEOBResumS builds before and after the sub-sample SPA attachment: jitter,
# merger-time errors of the shipped models, and a (2,2) model on each
# (visualization/clean_training_data.py), in clean_training/.
#   PYTHON=... TEOB=<fork checkout> sbatch slurm/modes/clean_training.sh [N]
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=2:55:00
#SBATCH --job-name=clean-teob
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
N="${1:-17408}"
OUT=clean_training
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 N_JOBS=32
declare -A COMMITS=([jittery]=85ead73a [clean]=fd-merger-subsample)
for name in jittery clean; do
    if [ ! -d "$OUT/build-$name" ]; then
        rm -rf "$OUT/src-$name"; mkdir -p "$OUT/src-$name"
        git -C "$TEOB" archive "${COMMITS[$name]}" | tar -x -C "$OUT/src-$name"
        (cd "$OUT/src-$name/Python" && uv run --no-project --with setuptools --with numpy \
            --python "$PYTHON" python setup.py -q build_ext --build-lib "$PWD/../../build-$name" > ../build.log 2>&1)
    fi
done
for name in jittery clean; do
    export PYTHONPATH=".:$PWD/$OUT/build-$name"
    "$PYTHON" -W ignore visualization/teob_reference_jitter.py 64
    "$PYTHON" -W ignore visualization/merger_time_release.py 800
    [ -f "$OUT/$name.npz" ] || "$PYTHON" -W ignore visualization/clean_training_data.py generate "$OUT/$name.npz" "$N"
done
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS
"$PYTHON" -W ignore visualization/clean_training_data.py compare "$OUT/jittery.npz" "$OUT/clean.npz"
