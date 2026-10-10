#!/usr/bin/env bash
# One step of the precession-regression scaling pipeline, inside a SLURM job
# (submit.sh submits these) or here:
#
#   slurm/precession/job.sh generate            # claim and make shards until none is left
#   slurm/precession/job.sh exact               # validation waveforms with integrated angles
#   slurm/precession/job.sh prior ITERATION     # the prior of fold SLURM_ARRAY_TASK_ID (all, here)
#   slurm/precession/job.sh refit ITERATION     # refit the envelopes with the priors
#   slurm/precession/job.sh train NAME SIZE [precession_scale.py train options]
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
[[ -f slurm/precession/cluster.env ]] && source slurm/precession/cluster.env
source slurm/lib.sh
DATA="${DATA:-visualization/precession_scale/run}"
S=visualization/precession_scale.py
CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
[[ $# -ge 1 ]] || { sed -n '2,17p' "$0"; exit 1; }
STEP="$1"; shift

# processes of their own, single-threaded, for the integrations and fits;
# threaded BLAS and XLA for kernel ridge and the perceptrons
case "$STEP" in
    generate)
        single_threaded
        CMD=("$PYTHON" "$S" generate "$DATA/validation" "$DATA/train" --jobs "$CPUS") ;;
    exact)
        export OMP_NUM_THREADS="$CPUS"
        CMD=("$PYTHON" "$S" exact "$DATA/validation") ;;
    prior)
        export OMP_NUM_THREADS="$CPUS"
        CMD=("$PYTHON" "$S" refine-prior "$DATA/train" --iteration "$1" --folds "${FOLDS:-4}"
             --prior-size "${PRIOR_SIZE:-16384}" ${SLURM_ARRAY_TASK_ID:+--fold "$SLURM_ARRAY_TASK_ID"}) ;;
    refit)
        single_threaded
        CMD=("$PYTHON" "$S" refine-fit "$DATA/train" --iteration "$1" --jobs "$CPUS") ;;
    train)
        export OMP_NUM_THREADS="$CPUS"
        NAME="$1" SIZE="$2"; shift 2
        mkdir -p "$DATA/runs"
        CMD=("$PYTHON" "$S" train "$DATA/train" --n "$SIZE" --out "$DATA/runs/${NAME}_$SIZE.joblib"
             --checkpoint "$DATA/runs/${NAME}_$SIZE.checkpoint" --validate "$DATA/validation" "$@") ;;
    *)
        echo "unknown step $STEP" >&2; exit 1 ;;
esac
run_step "${CMD[@]}"
