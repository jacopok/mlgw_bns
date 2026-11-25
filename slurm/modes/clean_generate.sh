#!/usr/bin/env bash
# The training and validation waveforms of a dataset again (same draws, same
# downsampling), with the EOBRun_module on PYTHONPATH; then the principal
# components. Resumable.
#   PYTHON=... PYTHONPATH=<build> sbatch --array=0-1 slurm/modes/clean_generate.sh OLD NEW generate
#   PYTHON=... PYTHONPATH=<build> sbatch slurm/modes/clean_generate.sh OLD NEW pca
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=2:55:00
#SBATCH --job-name=clean-generate
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
OLD="$1" NEW="$2"
export PYTHONPATH="$PWD:$PYTHONPATH" PYTHONUNBUFFERED=1
"$PYTHON" -c "import mlgw_bns, EOBRun_module; print(mlgw_bns.__file__, EOBRun_module.__file__)"
M=visualization/modes_scale.py
if [ "$3" = pca ]; then
    OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" exec "$PYTHON" "$M" pca "$NEW/train" --n 32768 --components 60
fi
for part in train validation; do
    mkdir -p "$NEW/$part"
    for file in config.json downsampling.npz; do cp -n "$OLD/$part/$file" "$NEW/$part/" || true; done
done
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 exec "$PYTHON" "$M" generate "$NEW/validation" "$NEW/train" --jobs "$SLURM_CPUS_PER_TASK"
