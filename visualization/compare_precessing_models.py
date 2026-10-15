"""Which part of a precessing model is the error: the co-precessing modes, or
the precession angles?

Runs :func:`validate_model.precessing_full_waveform_mismatches` (the same
binaries, against TEOBResumS) for every modes model, with the angles either
integrated or taken from each precession regressor; and the (2,2)
per-mode mismatches of :func:`validate_model.per_mode_mismatches` of each
modes model. Saves everything to ``compare_precessing_models.npz``.

    python compare_precessing_models.py \
        --models ../mlgw_bns/data/default_hom modes_scale/local/runs/krr_16384/model \
        --regressors precession_scale/run/runs/mlp_n1048576_best139_1048576.joblib
"""

import argparse

import numpy as np

import validate_model
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.precession_regression import PrecessionRegressor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--regressors", nargs="*", default=[])
    parser.add_argument("--n", type=int, default=validate_model.N_FULL_WAVEFORM_MISMATCHES)
    parser.add_argument("--out", default="compare_precessing_models.npz")
    args = parser.parse_args()
    validate_model.N_MISMATCH_WAVEFORMS = args.n
    all_modes = validate_model.MODES

    regressors = {"integrated": None}
    for filename in args.regressors:
        regressors[filename] = PrecessionRegressor.load(filename)

    results = {}
    for model_filename in args.models:
        validate_model.MODES = all_modes
        model = validate_model.load_model(model_filename)
        print(f"{model_filename}: (2,2) per-mode mismatches")
        validate_model.MODES = [Mode(2, 2)]
        (optimised, shifted), = validate_model.per_mode_mismatches(model).values()
        results[f"{model_filename} | mode22 optimised"] = optimised
        results[f"{model_filename} | mode22 predicted-shift"] = shifted
        for name, regressor in regressors.items():
            print(f"{model_filename}, angles {name}: precessing vs TEOBResumS")
            validate_model.PRECESSION_REGRESSOR = regressor
            mismatches, _, _ = validate_model.precessing_full_waveform_mismatches(model, args.n)
            results[f"{model_filename} | precessing {name}"] = mismatches

    np.savez(args.out, **results)
    print(f"\n{'':95} median      90%      99%      max")
    for key, values in results.items():
        print(f"{key:95}", "  ".join(f"{q:.1e}" for q in np.quantile(values, [0.5, 0.9, 0.99, 1])))


if __name__ == "__main__":
    main()
