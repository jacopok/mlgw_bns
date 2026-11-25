#!/usr/bin/env bash
# The metric models of the four ranges, retrained on TEOBResumS waveforms
# without the merger-time jumps (the same draws, downsampling, metrics, sizes
# and width), and their whole validation. Submits everything, chained.
#   PYTHON=... EOB=<build dir> METRICS=<metric_study dir> OLD=<modes_scale dir> slurm/modes/clean_release.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$EOB" METRICS RELEASE="$PWD/clean_release"
LOGS="$RELEASE/logs"; mkdir -p "$LOGS"
S=(sbatch --parsable --kill-on-invalid-dep=yes -o "$LOGS/%x-%A_%a.log" -p short -t 2:55:00)
declare -A DATASETS=([2000]=release_lambda2000_effective [4000]=release_lambda4000 [8000]=release_lambda8000 [12000]=release_lambda12000)
for L in 2000 4000 8000 12000; do
    old="$OLD/${DATASETS[$L]}" new="$RELEASE/lambda$L/data"
    gen=$("${S[@]}" -J "gen-$L" --array=0-1 slurm/modes/clean_generate.sh "$old" "$new" generate)
    pca=$("${S[@]}" -J "pca-$L" -d "afterok:$gen" slurm/modes/clean_generate.sh "$old" "$new" pca)
    models=() ids=()
    for n in 8192 16384; do
        base="$RELEASE/lambda$L/runs/metric${n}_g0.3/model"
        fit=$("${S[@]}" -J "mod-$L-$n" -d "afterok:$pca" slurm/modes/metric_model.sh "$L" "$new" "$n" 0.3)
        # the models trained before, on the new validation set
        "${S[@]}" -J "oldval-$L-$n" -d "afterok:$pca" -c 32 --mem 64G --wrap "OMP_NUM_THREADS=1 PYTHONPATH=$PWD:$EOB $PYTHON visualization/modes_scale.py validate $new/validation --model $METRICS/release/lambda$L/runs/metric${n}_g0.3/model --jobs 32 --n 8192 --out $RELEASE/lambda$L/runs/validate_old_metric${n}_g0.3.npz" > /dev/null
        models+=("$METRICS/release/lambda$L/runs/metric${n}_g0.3/model" "$base"); ids+=("$fit")
        i=0
        for steps in "residuals mismatches" "mismatch_vs_power mode_subset" "precessing"; do
            # shellcheck disable=SC2086
            "${S[@]}" -J "bn-$L-$n-${steps%% *}" -d "afterok:$fit" slurm/modes/metric_bench.sh "$base" "$RELEASE/lambda$L/bench/metric$n" 1000 $steps > /dev/null
        done
    done
    "${S[@]}" -J "chk-$L" -d "afterok:$(IFS=:; echo "${ids[*]}")" slurm/modes/metric_checks.sh "$L" "${models[@]}" > /dev/null
    echo "$L: generate $gen, pca $pca, models ${ids[*]}"
done
