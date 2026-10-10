#!/bin/bash
# More integrated binaries for the learning curves, q in [1, 1.5] like the beat A/B
set -e
cd "$(dirname "$0")/../.."
S=visualization/precession_regression_study.py
D=visualization/precession_experiments
for seed in 11 12 13; do
  [ -f $D/train_$seed.npz ] || uv run python $S generate --n 4096 --seed $seed --mass-ratio 1 1.5 --out $D/train_$seed.npz
done
[ -f $D/validation_21.npz ] || uv run python $S generate --n 768 --seed 21 --mass-ratio 1 1.5 --out $D/validation_21.npz
echo GENERATE-DONE
