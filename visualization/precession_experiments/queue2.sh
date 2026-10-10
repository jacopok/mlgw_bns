#!/bin/bash
# The best targets so far (smoothing 1e-2) through the learning curve, with
# and without refinement, by kernel ridge and by the MLP; 1024 validation.
cd "$(dirname "$0")/../.."
S=visualization/precession_regression_study.py
D=visualization/precession_experiments
AB=visualization/precession_beat_ab
TRAIN="$AB/train_elliptic.npz $D/train_11.npz $D/train_12.npz $D/train_13.npz"
VAL="$AB/validation.npz $D/validation_21.npz"
G="--smoothing 1e-2 --refit-cache $D/train_all_smooth1e-2.npy"
step() { echo "=== $1 [$(date +%H:%M)]"; }

step "kernel ridge, smoothing 1e-2, N = 4096"
uv run python $S validate --regressor $D/smooth1e-2.joblib --data $VAL --out $D/validate_krr_smooth_4096.npz
step "kernel ridge, smoothing 1e-2, N = 16384"
[ -f $D/krr_smooth_16384.joblib ] || uv run python $S train --data $TRAIN $G --out $D/krr_smooth_16384.joblib
uv run python $S validate --regressor $D/krr_smooth_16384.joblib --data $VAL --out $D/validate_krr_smooth_16384.npz
for n in 4096 16384; do
  step "kernel ridge, smoothing 1e-2, refined, N = $n"
  [ -f $D/krr_smooth_refined_$n.joblib ] || uv run python $S train --data $TRAIN $G --n-train $n --refine 2 \
      --save-refined $D/smooth_refined_$n.npy --out $D/krr_smooth_refined_$n.joblib
  uv run python $S validate --regressor $D/krr_smooth_refined_$n.joblib --data $VAL --out $D/validate_krr_smooth_refined_$n.npz
  step "MLP 4x256 gelu, smoothing 1e-2, refined, N = $n"
  epochs=$((n == 4096 ? 3000 : 1500))
  [ -f $D/mlp_smooth_refined_$n.joblib ] || uv run python $S train --data $TRAIN $G --n-train $n \
      --coefficients $D/smooth_refined_$n.npy --mlp --mlp-epochs $epochs --out $D/mlp_smooth_refined_$n.joblib
  uv run python $S validate --regressor $D/mlp_smooth_refined_$n.joblib --data $VAL --out $D/validate_mlp_smooth_refined_$n.npz
done
echo "QUEUE2-DONE [$(date +%H:%M)]"
