#!/usr/bin/env bash
# The likelihood of one model (of metric_kernel_fits.py model) at the nested
# samples of a teob_injections run: chunk $SLURM_ARRAY_TASK_ID of $NCHUNK.
#   MODEL=DIR (with model_l2_m2.yaml ...; "shipped" for the one of $BASE) NCHUNK=8 sbatch --array=0-7 slurm/modes/metric_lnl.sh
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --time=2:55:00
#SBATCH --job-name=metric-lnl
set -euo pipefail
HERE="${SLURM_SUBMIT_DIR}"
BASE="${BASE:-$HOME/calib_study/overlay_iau}"
RUN="${RUN:-teob_AS_HFLF_2L_l2000}"
NAME=$(basename "$MODEL")
OVERLAY="$TMPDIR/overlay-$NAME-$SLURM_JOB_ID"
[ -d "${TMPDIR:-}" ] || OVERLAY="/tmp/overlay-$NAME-$SLURM_JOB_ID"
trap 'rm -rf "$OVERLAY"' EXIT
mkdir -p "$OVERLAY/mlgw_bns/data"
ln -s "$BASE/bilby_xG" "$OVERLAY/bilby_xG"
cp "$HERE"/mlgw_bns/*.py "$OVERLAY/mlgw_bns/"
ln -s "$BASE"/mlgw_bns/data/* "$OVERLAY/mlgw_bns/data/"
if [ "$NAME" != shipped ]; then
  for f in "$MODEL"/model_l*; do
    ln -sf "$(realpath "$f")" "$OVERLAY/mlgw_bns/data/default_hom_lambda2000_$(basename "$f" | sed s/model_//)"
  done
fi
export PYTHONPATH="$OVERLAY" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu
OUT="$HERE/visualization/metric_study/lnl"
mkdir -p "$OUT"
cd "$HOME/calib_study"
"$HOME/xg_pe/.venv/bin/python" "$HERE/visualization/metric_lnl_nested.py" run "$HOME/xg_pe/runs/$RUN" \
  "rb_correction/$RUN/ratio_zero_fine.pickle" "lacc/$RUN/surrogate_fine_nested_part*of32.npz" \
  "$OUT/${NAME}_${SLURM_ARRAY_TASK_ID}of$NCHUNK.npz" --chunk "$SLURM_ARRAY_TASK_ID/$NCHUNK" --npool 24 ${EXTRA:-}
