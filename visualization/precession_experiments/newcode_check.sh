#!/bin/bash
# The 4096-binary regressors of queue2.sh again, with the streaming PCA and
# the float32 perceptron: the same validation numbers are expected
cd "$(dirname "$0")/../.."
S=visualization/precession_regression_study.py
D=visualization/precession_experiments
AB=visualization/precession_beat_ab
TRAIN="$AB/train_elliptic.npz $D/train_11.npz $D/train_12.npz $D/train_13.npz"
VAL="$AB/validation.npz $D/validation_21.npz"
G="--smoothing 1e-2 --refit-cache $D/train_all_smooth1e-2.npy"
step() { echo "=== $1 [$(date +%H:%M)]"; }
step "kernel ridge, smoothing 1e-2, refined, N = 4096, new code"
[ -f $D/krr_smooth_refined_4096_newcode.joblib ] || .venv/bin/python $S train --data $TRAIN $G --n-train 4096 \
    --coefficients $D/smooth_refined_4096.npy --out $D/krr_smooth_refined_4096_newcode.joblib
.venv/bin/python $S validate --regressor $D/krr_smooth_refined_4096_newcode.joblib --data $VAL --exact-cache $D/exact_1024.npz --out $D/validate_krr_smooth_refined_4096_newcode.npz
step "MLP, smoothing 1e-2, refined, N = 4096, float32"
[ -f $D/mlp_smooth_refined_4096_newcode.joblib ] || .venv/bin/python $S train --data $TRAIN $G --n-train 4096 \
    --coefficients $D/smooth_refined_4096.npy --mlp --out $D/mlp_smooth_refined_4096_newcode.joblib
.venv/bin/python $S validate --regressor $D/mlp_smooth_refined_4096_newcode.joblib --data $VAL --exact-cache $D/exact_1024.npz --out $D/validate_mlp_smooth_refined_4096_newcode.npz
echo "NEWCODE-DONE [$(date +%H:%M)]"
