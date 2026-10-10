# Shared by the pipelines of slurm/*/: sourced by their submit.sh and job.sh,
# after they cd to the repository and source their cluster.env.

PYTHON="${PYTHON:-.venv/bin/python}"

# DATA made absolute (the jobs start elsewhere than where they were submitted)
# and exported, with its logs directory made: SLURM silently drops the output
# of a job whose --output directory does not exist
resolve_data() {
    [[ -n "${DATA:-}" ]] || { echo "DATA is not set: set it in cluster.env" >&2; exit 1; }
    mkdir -p "$DATA/logs"
    DATA="$(cd "$DATA" && pwd)"
    export DATA
}

# after ID...: the dependency on those of the jobs submitted (empty ids ignored)
after() {
    local ids=()
    for id in "$@"; do [[ -n "$id" ]] && ids+=("$id"); done
    [[ ${#ids[@]} -gt 0 ]] && echo "--dependency=afterok:$(IFS=:; echo "${ids[*]}") --kill-on-invalid-dep=yes"
    return 0
}

# the options of every job: run from the repository, requeueable (appending
# to its log, which keeps the output of every attempt), warned SIGNAL_LEAD
# seconds before the walltime, and the account, partition and extras of
# cluster.env
sbatch_common() {
    COMMON=(--parsable --nodes=1 --ntasks=1 --chdir="$PWD" --requeue --open-mode=append
            --signal="B:USR1@${SIGNAL_LEAD:-300}")
    [[ -n "${SLURM_ACCOUNT:-}" ]] && COMMON+=(--account="$SLURM_ACCOUNT")
    [[ -n "${SLURM_PARTITION:-}" ]] && COMMON+=(--partition="$SLURM_PARTITION")
    # shellcheck disable=SC2206
    COMMON+=(${SLURM_EXTRA:-})
}

# submit PIPELINE NAME CPUS MEMORY TIME [sbatch options] -- [job.sh arguments]:
# a job running slurm/PIPELINE/job.sh, logging to $DATA/logs; its id
submit() {
    local pipeline="$1" name="$2" cpus="$3" mem="$4" time="$5"; shift 5
    local options=()
    while [[ "$1" != -- ]]; do options+=("$1"); shift; done
    shift
    local output="$DATA/logs/%x-%j.log"
    [[ " ${options[*]} " == *" --array="* ]] && output="$DATA/logs/%x-%A_%a.log"
    local id
    id=$(sbatch "${COMMON[@]}" --job-name="$pipeline-$name" --cpus-per-task="$cpus" --mem="$mem" \
        --time="$time" --output="$output" "${options[@]}" "slurm/$pipeline/job.sh" "$@")
    echo "${id%%;*}"
}

# processes of their own, single-threaded, for sweeps of many small
# computations; the multithreaded BLAS and XLA of the others otherwise
single_threaded() {
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu
    export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
}

# run_step COMMAND...: run it; inside SLURM, USR1 (the walltime near) is
# forwarded to it, and if it exits with 75 (stopped, its work saved) the job
# is requeued, at most MAX_REQUEUES times, provided it was warned of the
# walltime or ran for MIN_RUN_SECONDS (not to requeue in a loop something
# stopping as it starts); TERM (scancel) stops it without requeueing. Exits
# with its status.
run_step() {
    echo "=== [$(date -Is)] $(hostname) ${SLURM_JOB_ID:+job $SLURM_JOB_ID${SLURM_ARRAY_TASK_ID:+ task $SLURM_ARRAY_TASK_ID} (requeue ${SLURM_RESTART_COUNT:-0}) }cpus=${CPUS:-?}"
    echo "$*"
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        exec "$@"
    fi
    "$@" &
    local child=$!
    walltime=0 terminated=0
    trap 'echo "=== [$(date -Is)] walltime near: stopping"; walltime=1; kill -USR1 "$child" 2>/dev/null || true' USR1
    trap 'echo "=== [$(date -Is)] SIGTERM: stopping"; terminated=1; kill -TERM "$child" 2>/dev/null || true' TERM
    local rc=0 s
    while kill -0 "$child" 2>/dev/null; do
        wait "$child" && rc=0 || rc=$?  # returns early when a trap runs
    done
    wait "$child" 2>/dev/null && rc=0 || { s=$?; (( s == 127 )) || rc=$s; }
    echo "=== [$(date -Is)] exited with $rc"
    # 75: stopped before finishing (the walltime, or --hours), with its work saved
    if [[ $rc == 75 && $terminated == 0 ]]; then
        if [[ $walltime == 0 ]] && (( SECONDS < ${MIN_RUN_SECONDS:-600} )); then
            echo "=== stopped after ${SECONDS} s without the walltime signal: not requeued" >&2
        elif (( ${SLURM_RESTART_COUNT:-0} < ${MAX_REQUEUES:-20} )); then
            local job="${SLURM_ARRAY_JOB_ID:+${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
            job="${job:-$SLURM_JOB_ID}"
            echo "=== requeueing $job"
            scontrol requeue "$job"
        else
            echo "=== MAX_REQUEUES (${MAX_REQUEUES:-20}) reached, not requeueing" >&2
        fi
    elif [[ $walltime == 1 ]]; then
        echo "=== stopped by the walltime signal with status $rc: not requeued" >&2
    fi
    exit "$rc"
}

# the batch of a perceptron's training on N samples: 128, growing with the
# training set to 1024
mlp_batch() {
    local n=$1 batch=128
    while (( batch < 1024 && batch * 128 < n )); do batch=$((batch * 2)); done
    echo "$batch"
}
