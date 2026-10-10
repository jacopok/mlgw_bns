#!/bin/bash
# Coarser envelopes to pin the split between the two carriers: stronger
# smoothing, or fewer B-spline cells (N = 4096, unrefined, 256 validation).
cd "$(dirname "$0")/../.."
S=visualization/precession_regression_study.py
D=visualization/precession_experiments
AB=visualization/precession_beat_ab
run() {  # name, grid options
  name=$1; shift
  echo "=== $name [$(date +%H:%M)]"
  [ -f $D/$name.joblib ] || uv run python $S train --data $AB/train_elliptic.npz "$@" \
      --refit-cache $D/train_elliptic_$name.npy --out $D/$name.joblib
  uv run python $S validate --regressor $D/$name.joblib --data $AB/validation.npz --oracle fit \
      --refit-cache $D/validation_$name.npy --out $D/validate_oracle_fit_$name.npz
  uv run python $S validate --regressor $D/$name.joblib --data $AB/validation.npz --out $D/validate_$name.npz
}
run smooth1e-4 --smoothing 1e-4
run smooth1e-2 --smoothing 1e-2
run cells64 --cells 64
run cells32 --cells 32
echo "SMOOTHING-DONE [$(date +%H:%M)]"
