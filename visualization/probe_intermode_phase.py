"""Per-mode phase of the aligned-spin surrogate relative to TEOBResumS.

Reproduces validate_model.full_waveform_mismatches' setup (default_hom,
inclination 1, training-distribution parameters, predict_modes_dict vs
get_teob_modes_dict) and, at the optimal (t_c, phi_c) of the full-waveform
mismatch, reports each mode's residual phase against TEOBResumS (the
power-weighted mean of arg(true* predicted)), its power fraction, and the
full-waveform mismatch with and without that mode's constant offset removed.
A constant per-mode phase offset is exactly what a per-mode mismatch
optimises away and a full-waveform mismatch at a single inclination only
sees in proportion to that mode's power.

Run with: python visualization/probe_intermode_phase.py [N]
"""
import sys

import numpy as np

from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model import Model
from mlgw_bns.model_validation import ValidateModel
from mlgw_bns.mode_model import ParametersWithExtrinsic

MODES = [Mode(2, 2), Mode(2, 1), Mode(3, 1), Mode(3, 2), Mode(3, 3), Mode(4, 3), Mode(4, 4)]
if "--subset" in sys.argv:  # what the precession scripts load
    MODES = [Mode(2, 2), Mode(2, 1), Mode(3, 3), Mode(4, 4)]
    sys.argv.remove("--subset")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12

model = Model.default_for_testing(modes=MODES)  # the packaged default_hom
validator = ValidateModel(model.mode_models[Mode(2, 2)])
frequencies = validator.frequencies
frequencies = frequencies[frequencies >= 20.0]
psd = validator.psd_at_frequencies(frequencies)
weights = np.gradient(frequencies) / psd

generator = model.dataset.make_parameter_generator(20)


def report(intrinsic):
    params = ParametersWithExtrinsic(
        mass_ratio=intrinsic.mass_ratio, lambda_1=intrinsic.lambda_1,
        lambda_2=intrinsic.lambda_2, chi_1=intrinsic.chi_1, chi_2=intrinsic.chi_2,
        distance_mpc=100.0, inclination=1.0, total_mass=2.8,
    )
    predicted = model.predict_modes_dict(frequencies, params)
    true = model.get_teob_modes_dict(frequencies, params)
    support = np.ones(frequencies.size, bool)
    for array in true.values():
        support &= np.abs(array) > 0
    f = frequencies[support]
    true = {k: v[support] for k, v in true.items()}
    predicted = {k: v[support] for k, v in predicted.items()}
    w = weights[support]

    mismatch, t_c, phi_c = validator.full_waveform_mismatch(
        true, predicted, frequencies=f, return_shifts=True)
    shifted = {(l, m): v * np.exp(2j * np.pi * f * t_c + 1j * m * phi_c)
               for (l, m), v in predicted.items()}
    total = np.sum(np.abs(sum(true.values())) ** 2 * w)

    offsets = []
    corrected = dict(shifted)
    for key in sorted(true):
        overlap = np.sum(np.conj(true[key]) * shifted[key] * w)
        offset = np.angle(overlap)
        power = np.sum(np.abs(true[key]) ** 2 * w) / total
        corrected[key] = shifted[key] * np.exp(-1j * offset)
        if power > 1e-4:
            offsets.append(f"{key}: {offset:+.2f} rad ({100 * power:.2f}%)")
    mismatch_fixed = validator.full_waveform_mismatch(true, corrected, frequencies=f)
    return (f"q={intrinsic.mass_ratio:.2f} chi=({intrinsic.chi_1:+.2f},{intrinsic.chi_2:+.2f})  "
            f"full mismatch {mismatch:.2e}  (every mode's constant offset removed: "
            f"{mismatch_fixed:.2e})  " + "  ".join(offsets))


for _ in range(N):
    intrinsic = next(generator)
    try:
        print(report(intrinsic), flush=True)
    except RuntimeError as error:  # TEOBResumS' root finder, occasionally
        print(f"skipped (q={intrinsic.mass_ratio:.2f}): {error}", flush=True)
