#!/usr/bin/env bash
# One step of the scaling pipeline of the co-precessing modes model, inside a
# SLURM job (submit.sh submits these) or here:
#
#   slurm/modes/job.sh downsampling           # the downsampling of the training set, shared by the validation set
#   slurm/modes/job.sh generate               # claim and make shards until none is left
#   slurm/modes/job.sh pca                    # the principal components of the first PCA_SIZE waveforms
#   slurm/modes/job.sh train NAME SIZE [modes_scale.py train options]
#                                             # the regressor of mode SLURM_ARRAY_TASK_ID (all, here)
#   slurm/modes/job.sh validate NAME SIZE [modes_scale.py train options]
#                                             # assemble the model, validate it on the validation set
#
# Settings from cluster.env (see cluster.env.example). Every step skips what
# is already done. Inside SLURM, USR1 (--signal=B:USR1@SIGNAL_LEAD, before the
# walltime) makes the step checkpoint or abandon its shard and exit with 75,
# and the job is requeued, at most MAX_REQUEUES times; TERM (scancel) stops it
# without requeueing.
set -euo pipefail
# under sbatch the script runs from a spool copy: submit.sh submits from the repository
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
[[ -f slurm/modes/cluster.env ]] && source slurm/modes/cluster.env
source slurm/lib.sh
DATA="${DATA:-visualization/modes_scale/run}"
M=visualization/modes_scale.py
CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
[[ $# -ge 1 ]] || { sed -n '2,18p' "$0"; exit 1; }
STEP="$1"; shift

# the modes, in the order of the array tasks of a training
MODES=(l2_m2 l2_m1 l3_m1 l3_m2 l3_m3 l4_m3 l4_m4)

case "$STEP" in
    downsampling)
        single_threaded
        "$PYTHON" "$M" downsampling "$DATA/train" --size "${DOWNSAMPLING_SIZE:-64}" --jobs "$CPUS"
        CMD=("$PYTHON" "$M" init "$DATA/validation" --n "$N_VALIDATION" --seed 2 --shard-size "$SHARD_SIZE"
             --lambda-max "$LAMBDA_MAX" --downsampling-from "$DATA/train") ;;
    generate)
        single_threaded
        CMD=("$PYTHON" "$M" generate "$DATA/validation" "$DATA/train" --jobs "$CPUS") ;;
    pca)
        export OMP_NUM_THREADS="$CPUS"
        CMD=("$PYTHON" "$M" pca "$DATA/train" --n "$PCA_SIZE") ;;
    train|validate)
        export OMP_NUM_THREADS="$CPUS"
        NAME="$1" SIZE="$2"; shift 2
        RUN="$DATA/runs/${NAME}_$SIZE"
        mkdir -p "$RUN"
        CMD=("$PYTHON" "$M" train "$DATA/train" --n "$SIZE" --pca-size "$PCA_SIZE" --out "$RUN/model"
             --checkpoint "$RUN/checkpoint" "$@")
        if [[ $STEP == train ]]; then
            [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] && CMD+=(--modes "${MODES[$SLURM_ARRAY_TASK_ID]}")
        else
            CMD+=(--validate "$DATA/validation" --jobs "$CPUS")
        fi ;;
    *)
        echo "unknown step $STEP" >&2; exit 1 ;;
esac
run_step "${CMD[@]}"
