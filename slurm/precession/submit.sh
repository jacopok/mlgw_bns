#!/usr/bin/env bash
# The scaling experiment of the regressed precession angles on SLURM: a
# training set of N_TRAIN binaries and an independent validation set made in
# parallel, the envelopes refined, then regressors trained on the first N
# binaries for each N in SIZES, each validated on the whole validation set.
# Settings in cluster.env (cp slurm/precession/cluster.env.example
# slurm/precession/cluster.env, and edit it); run from anywhere.
#
#   slurm/precession/submit.sh                    # all the stages, each after the one before
#   slurm/precession/submit.sh generate exact     # only these
#   slurm/precession/submit.sh train              # e.g. after adding SERIES or SIZES
#
# The stages:
#   generate  GENERATE_TASKS jobs (an array) claiming shards of both datasets
#             until none is left
#   exact     the waveforms of the validation set with integrated angles
#   refine    per iteration, FOLDS jobs training the priors, then
#             GENERATE_TASKS refitting the envelopes
#   train     a job per series and size: train, then validate
# Each starts once the stages it needs, submitted with it, have succeeded.
# Stages already done (precession_scale.py pending) are not submitted, nor
# regressors already validated, so running it again with more SERIES or
# SIZES submits only those. Everything is resumable besides: a stage
# submitted again after a failure redoes nothing that is there. Jobs
# reaching their walltime checkpoint and are requeued (job.sh).
#
# Follow with (from the repository: source slurm/precession/cluster.env sets DATA)
#   squeue --me
#   tail -f $DATA/logs/*.log
#   .venv/bin/python visualization/precession_scale.py status $DATA/validation $DATA/train
# and, as the train jobs end, plot the learning curve:
#   .venv/bin/python visualization/precession_learning_curve.py --directory $DATA/runs
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
[[ -f slurm/precession/cluster.env ]] || {
    echo "no slurm/precession/cluster.env: cp slurm/precession/cluster.env.example slurm/precession/cluster.env and edit it" >&2
    exit 1
}
source slurm/precession/cluster.env
source slurm/lib.sh
check_environment
resolve_data
S=visualization/precession_scale.py
STAGES=("$@")
[[ ${#STAGES[@]} -gt 0 ]] || STAGES=(generate exact refine train)

# the datasets: made, or checked against the configuration already there
for dataset in "train $N_TRAIN 1" "validation $N_VALIDATION 2"; do
    read -r name n seed <<< "$dataset"
    # shellcheck disable=SC2086
    OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu "$PYTHON" "$S" init "$DATA/$name" --n "$n" --seed "$seed" \
        --shard-size "$SHARD_SIZE" --mass-ratio $MASS_RATIO --smoothing "$SMOOTHING" 2>/dev/null | head -1
done

# the stages done already (the datasets made, the envelopes refined) are skipped
PENDING=" $(OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu "$PYTHON" "$S" pending "$DATA/validation" "$DATA/train" \
    --refine-iterations "$REFINE_ITERATIONS" | tr '\n' ' ')"
pending() { [[ "$PENDING" == *" $1 "* ]] || { echo "$1: done"; return 1; }; }

sbatch_common

GENERATED="" EXACT="" REFINED=""
for stage in "${STAGES[@]}"; do
    case "$stage" in
    generate)
        pending generate || continue
        # shellcheck disable=SC2046
        GENERATED=$(submit precession generate "$GENERATE_CPUS" "$GENERATE_MEM" "$GENERATE_TIME" \
            --array="0-$((GENERATE_TASKS - 1))" -- generate)
        echo "generate: $GENERATED (array of $GENERATE_TASKS)" ;;
    exact)
        pending exact || continue
        # shellcheck disable=SC2046
        EXACT=$(submit precession exact "$EXACT_CPUS" "$EXACT_MEM" "$EXACT_TIME" $(after "$GENERATED") -- exact)
        echo "exact: $EXACT" ;;
    refine)
        previous="$GENERATED"
        for (( iteration = 1; iteration <= REFINE_ITERATIONS; iteration++ )); do
            pending "refine$iteration" || continue
            # shellcheck disable=SC2046
            prior=$(submit precession "prior$iteration" "$PRIOR_CPUS" "$PRIOR_MEM" "$PRIOR_TIME" \
                --array="0-$((FOLDS - 1))" $(after "$previous") -- prior "$iteration")
            # shellcheck disable=SC2046
            previous=$(submit precession "refit$iteration" "$GENERATE_CPUS" "$GENERATE_MEM" "$GENERATE_TIME" \
                --array="0-$((GENERATE_TASKS - 1))" $(after "$prior") -- refit "$iteration")
            echo "refinement $iteration: priors $prior, refits $previous"
        done
        REFINED="$previous" ;;
    train)
        source_="base"
        (( REFINE_ITERATIONS > 0 )) && source_="refine$REFINE_ITERATIONS"
        for series in "${SERIES[@]}"; do
            # name|options, or name|options|sizes (SIZES otherwise)
            IFS='|' read -r name options sizes <<< "$series"
            skipped=()
            for n in ${sizes:-$SIZES}; do
                (( n <= N_TRAIN )) || continue
                [[ -f "$DATA/runs/validate_${name}_$n.npz" ]] && { skipped+=("$n"); continue; }
                if [[ " $options " == *" --mlp "* ]]; then
                    mem="$((8 + 16 * n / 1048576))G"
                    extra=(${MLP_GRES:+--gres="$MLP_GRES"})
                    # the defaults first: the options of the series override them
                    # shellcheck disable=SC2206
                    arguments=(--mlp-steps "$MLP_STEPS" --mlp-batch "$(mlp_batch "$n")" $options)
                else
                    (( n <= KRR_MAX )) || continue
                    # the kernel matrix and its eigenvectors, and some room
                    mem="$((8 + 4 * n * n * 8 / 1000000000))G"
                    extra=()
                    # shellcheck disable=SC2206
                    arguments=($options)
                fi
                # shellcheck disable=SC2046
                id=$(submit precession "${name}_$n" "$TRAIN_CPUS" "$mem" "$TRAIN_TIME" "${extra[@]}" \
                    $(after "$REFINED" "$EXACT") -- train "$name" "$n" --source "$source_" "${arguments[@]}")
                echo "train $name, $n binaries ($source_): $id"
            done
            [[ ${#skipped[@]} -eq 0 ]] || echo "train $name: done for ${skipped[*]}"
        done ;;
    *)
        echo "unknown stage $stage: generate, exact, refine or train" >&2; exit 1 ;;
    esac
done

echo
echo "logs in $DATA/logs; to follow, from $ROOT:"
echo "  source slurm/precession/cluster.env   # sets DATA=$DATA"
echo "  squeue --me; tail -f \$DATA/logs/*.log"
echo "  .venv/bin/python visualization/precession_scale.py status $DATA/validation $DATA/train"
