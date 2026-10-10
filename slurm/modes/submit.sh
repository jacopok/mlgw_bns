#!/usr/bin/env bash
# The scaling experiment of the co-precessing modes model (the seven modes of
# hom7_big) on SLURM: a training set of N_TRAIN waveforms and an independent
# validation set made in parallel, then models trained on the first N
# waveforms for each N in SIZES, with kernel ridge and with perceptrons, each
# validated on the whole validation set, timed, and sized.
# Settings in cluster.env (cp slurm/modes/cluster.env.example
# slurm/modes/cluster.env, and edit it); run from anywhere.
#
#   slurm/modes/submit.sh                          # all the stages, each after the one before
#   slurm/modes/submit.sh downsampling generate    # only these
#   slurm/modes/submit.sh train                    # e.g. after adding SERIES or SIZES
#
# The stages:
#   downsampling  the downsampling indices of the training set, shared by the
#                 validation set
#   generate      GENERATE_TASKS jobs (an array) claiming shards of both
#                 datasets until none is left
#   pca           the principal components of the first PCA_SIZE waveforms
#   train         per series and size, a job per mode (an array) training its
#                 regressor, then one assembling the model and validating it
# Each starts once the stages it needs, submitted with it, have succeeded.
# Everything is resumable and skips what is done: submitting a stage again
# (after a failure, or with more sizes) redoes nothing that is there. Jobs
# reaching their walltime checkpoint and are requeued (job.sh).
#
# Follow with (from the repository: source slurm/modes/cluster.env sets DATA)
#   squeue --me
#   tail -f $DATA/logs/*.log
#   .venv/bin/python visualization/modes_scale.py status $DATA/train $DATA/validation
# and, as the validations end,
#   .venv/bin/python visualization/modes_scale.py summary $DATA/runs
#   .venv/bin/python visualization/precession_learning_curve.py --directory $DATA/runs --threshold 1e-4
# What to copy back: see docs/usage_guides/cluster.md.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
[[ -f slurm/modes/cluster.env ]] || {
    echo "no slurm/modes/cluster.env: cp slurm/modes/cluster.env.example slurm/modes/cluster.env and edit it" >&2
    exit 1
}
source slurm/modes/cluster.env
source slurm/lib.sh
[[ -x "$PYTHON" ]] || { echo "no $PYTHON: uv sync first" >&2; exit 1; }
resolve_data
M=visualization/modes_scale.py
STAGES=("$@")
[[ ${#STAGES[@]} -gt 0 ]] || STAGES=(downsampling generate pca train)

# the datasets: made, or checked against the configuration already there (the
# validation set gets the downsampling of the training set in its job)
for dataset in "train $N_TRAIN 1" "validation $N_VALIDATION 2"; do
    read -r name n seed <<< "$dataset"
    OMP_NUM_THREADS=1 "$PYTHON" "$M" init "$DATA/$name" --n "$n" --seed "$seed" --shard-size "$SHARD_SIZE" \
        --lambda-max "$LAMBDA_MAX" 2>/dev/null | head -1
done
sbatch_common

DOWNSAMPLED="" GENERATED="" PCA=""
for stage in "${STAGES[@]}"; do
    case "$stage" in
    downsampling)
        DOWNSAMPLED=$(submit modes downsampling "$DOWNSAMPLING_CPUS" "$DOWNSAMPLING_MEM" "$DOWNSAMPLING_TIME" \
            -- downsampling)
        echo "downsampling: $DOWNSAMPLED" ;;
    generate)
        # shellcheck disable=SC2046
        GENERATED=$(submit modes generate "$GENERATE_CPUS" "$GENERATE_MEM" "$GENERATE_TIME" \
            --array="0-$((GENERATE_TASKS - 1))" $(after "$DOWNSAMPLED") -- generate)
        echo "generate: $GENERATED (array of $GENERATE_TASKS)" ;;
    pca)
        # shellcheck disable=SC2046
        PCA=$(submit modes pca "$PCA_CPUS" "$PCA_MEM" "$PCA_TIME" $(after "$GENERATED") -- pca)
        echo "pca: $PCA" ;;
    train)
        for series in "${SERIES[@]}"; do
            name="${series%%|*}" options="${series#*|}"
            for n in $SIZES; do
                (( n <= N_TRAIN )) || continue
                cpus="$TRAIN_CPUS"
                if [[ " $options " == *" --mlp "* ]]; then
                    cpus="${MLP_CPUS:-$TRAIN_CPUS}"
                    mem="$((8 + 4 * n / 1048576))G"
                    extra=(${MLP_GRES:+--gres="$MLP_GRES"})
                    # shellcheck disable=SC2206
                    arguments=($options --mlp-steps "$MLP_STEPS" --mlp-batch "$(mlp_batch "$n")")
                else
                    (( n <= KRR_MAX )) || continue
                    # the kernel matrix and its eigenvectors, and some room
                    mem="$((8 + 21 * n / 1000 * n / 1000 * 8 / 10000))G"
                    extra=()
                    # shellcheck disable=SC2206
                    arguments=($options --max-kernel-ridge "$KRR_MAX")
                fi
                # shellcheck disable=SC2046
                trained=$(submit modes "${name}_$n" "$cpus" "$mem" "$TRAIN_TIME" --array=0-6 "${extra[@]}" \
                    $(after "$PCA") -- train "$name" "$n" "${arguments[@]}")
                # shellcheck disable=SC2046
                validated=$(submit modes "validate_${name}_$n" "$VALIDATE_CPUS" "$VALIDATE_MEM" "$VALIDATE_TIME" \
                    $(after "$trained" "$GENERATED") -- validate "$name" "$n" "${arguments[@]}")
                echo "train $name, $n waveforms: $trained (array of 7 modes), then validate: $validated"
            done
        done ;;
    *)
        echo "unknown stage $stage: downsampling, generate, pca or train" >&2; exit 1 ;;
    esac
done

echo
echo "logs in $DATA/logs; to follow, from $ROOT:"
echo "  source slurm/modes/cluster.env   # sets DATA=$DATA"
echo "  squeue --me; tail -f \$DATA/logs/*.log"
echo "  .venv/bin/python visualization/modes_scale.py status $DATA/train $DATA/validation"
