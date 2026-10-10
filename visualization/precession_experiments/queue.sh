#!/bin/bash
# Floors, refinement, learning curves (kernel ridge and MLP), m-oversampling, PCA size.
# Runs after sidebands_ab.sh; each step uses all cores, so one at a time.
cd "$(dirname "$0")/../.."
S=visualization/precession_regression_study.py
D=visualization/precession_experiments
AB=visualization/precession_beat_ab
TRAIN="$AB/train_elliptic.npz $D/train_11.npz $D/train_12.npz $D/train_13.npz"
VAL="$AB/validation.npz $D/validation_21.npz"
until grep -q SIDEBANDS-DONE $D/sidebands_ab.log; do sleep 30; done
step() { echo "=== $1 [$(date +%H:%M)]"; }

step "oracle fit (sidebands 0, 256 validation)"
uv run python $S validate --regressor $D/sb0.joblib --data $AB/validation.npz --oracle fit --out $D/validate_oracle_fit.npz
step "oracle pca (sidebands 0, 64+64 components, 256 validation)"
uv run python $S validate --regressor $D/sb0.joblib --data $AB/validation.npz --oracle pca --out $D/validate_oracle_pca.npz

step "refine 2 (4096)"
[ -f $D/sb0_refined.joblib ] || uv run python $S train --data $AB/train_elliptic.npz --refine 2 --out $D/sb0_refined.joblib
uv run python $S validate --regressor $D/sb0_refined.joblib --data $AB/validation.npz --out $D/validate_sb0_refined.npz

for n in 1024 2048 4096 8192 16384; do
  step "kernel ridge, N = $n (1024 validation)"
  [ -f $D/krr_$n.joblib ] || uv run python $S train --data $TRAIN --n-train $n --out $D/krr_$n.joblib
  uv run python $S validate --regressor $D/krr_$n.joblib --data $VAL --out $D/validate_krr_$n.npz
done

for n in 4096 16384; do
  step "MLP 4x256 gelu, N = $n (1024 validation)"
  epochs=$((n == 4096 ? 3000 : 1500))
  [ -f $D/mlp_$n.joblib ] || uv run python $S train --data $TRAIN --n-train $n --mlp --mlp-epochs $epochs --out $D/mlp_$n.joblib
  uv run python $S validate --regressor $D/mlp_$n.joblib --data $VAL --out $D/validate_mlp_$n.npz
done

step "m-oversampled training set, N = 4096 (half with max m >= 0.3)"
[ -f $D/train_highm_31.npz ] || uv run python $S generate --n 4096 --seed 31 --mass-ratio 1 1.5 --high-m-fraction 0.5 --out $D/train_highm_31.npz
[ -f $D/krr_highm.joblib ] || uv run python $S train --data $D/train_highm_31.npz --out $D/krr_highm.joblib
uv run python $S validate --regressor $D/krr_highm.joblib --data $VAL --out $D/validate_krr_highm.npz

step "PCA 128+128 components, N = 4096"
[ -f $D/krr_4096_pca128.joblib ] || uv run python $S train --data $TRAIN --n-train 4096 --components 128 128 --out $D/krr_4096_pca128.joblib
uv run python $S validate --regressor $D/krr_4096_pca128.joblib --data $VAL --out $D/validate_krr_4096_pca128.npz
uv run python $S validate --regressor $D/sb0.joblib --data $AB/validation.npz --oracle pca --out $D/validate_oracle_pca.npz > /dev/null
uv run python $S validate --regressor $D/krr_4096_pca128.joblib --data $AB/validation.npz --oracle pca --out $D/validate_oracle_pca128.npz
echo "QUEUE-DONE [$(date +%H:%M)]"
