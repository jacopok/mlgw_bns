#!/bin/bash
# Carriers on the precession-averaged rate, with smoothing 1e-2: the data
# integrated anew (the switch targets depend on the carriers), on the same
# binaries as the other runs; N = 4096, 1024 validation.
cd "$(dirname "$0")/../.."
S=visualization/precession_regression_study.py
D=visualization/precession_experiments
O="--mass-ratio 1 1.5 --averaged-precession --smoothing 1e-2"
step() { echo "=== $1 [$(date +%H:%M)]"; }
step "generate on the averaged carriers"
[ -f $D/avg_validation_2.npz ] || uv run python $S generate --n 256 --seed 2 $O --out $D/avg_validation_2.npz
[ -f $D/avg_validation_21.npz ] || uv run python $S generate --n 768 --seed 21 $O --out $D/avg_validation_21.npz
[ -f $D/avg_train_1.npz ] || uv run python $S generate --n 4096 --seed 1 $O --out $D/avg_train_1.npz
step "kernel ridge, averaged precession + smoothing 1e-2, N = 4096"
[ -f $D/krr_averaged_4096.joblib ] || uv run python $S train --data $D/avg_train_1.npz --out $D/krr_averaged_4096.joblib
uv run python $S validate --regressor $D/krr_averaged_4096.joblib --data $D/avg_validation_2.npz --oracle fit --out $D/validate_oracle_fit_averaged.npz
uv run python $S validate --regressor $D/krr_averaged_4096.joblib --data $D/avg_validation_2.npz $D/avg_validation_21.npz --out $D/validate_krr_averaged_4096.npz
echo "AVERAGED-DONE [$(date +%H:%M)]"
