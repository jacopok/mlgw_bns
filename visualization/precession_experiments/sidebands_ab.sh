#!/bin/bash
# Option 1: sideband carriers, A/B on the 4096 binaries of the elliptic-beat A/B
cd "$(dirname "$0")/../.."
S=visualization/precession_regression_study.py
D=visualization/precession_experiments
AB=visualization/precession_beat_ab
for k in 0 1 2; do
  [ -f $D/sb$k.joblib ] || uv run python $S train --data $AB/train_elliptic.npz --sidebands $k \
      --refit-cache $D/train_elliptic_sb$k.npy --out $D/sb$k.joblib
  echo "=== sidebands $k"
  uv run python $S validate --regressor $D/sb$k.joblib --data $AB/validation.npz --out $D/validate_sb$k.npz
done
echo SIDEBANDS-DONE
