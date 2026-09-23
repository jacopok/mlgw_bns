"""Surrogate vs EOB co-precessing multipoles on the model grid, per mode.

The "network" term of validate_precessing_against_teob (the two sources
twisted along the same angles) came out ~2e-3 on the shipped default_hom,
against ~1.5e-6 documented. This isolates it: aligned spins, no twist.
The cause turned out to be the 4-mode subset these scripts load: the
(3, 3) and (4, 4) read the wrong reference-phase column (fixed in
Model._mode_phases_column; see probe_intermode_phase.py). With the fix
the network term is ~1e-5.
"""
import numpy as np

import validate_precessing_against_teob as v
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model import Model
from mlgw_bns.model_validation import ValidateModel
from mlgw_bns.precessing_model import PrecessingParametersWithExtrinsic

model = Model.default_for_testing(modes=v.MODES)
validator = ValidateModel(model.mode_models[Mode(2, 2)])
frequencies = validator.frequencies
band = (frequencies >= v.BAND_LO) & (frequencies <= v.BAND_HI)
fb = frequencies[band]

rng = np.random.default_rng(v.SEED)
generator = model.dataset.make_parameter_generator(v.SEED)
for index in range(5):
    intrinsic = next(generator)
    v.random_spin_vector(rng, 0.0), v.random_spin_vector(rng, 0.0)
    params = PrecessingParametersWithExtrinsic(
        mass_ratio=intrinsic.mass_ratio, lambda_1=intrinsic.lambda_1,
        lambda_2=intrinsic.lambda_2, chi_1=(0, 0, intrinsic.chi_1),
        chi_2=(0, 0, intrinsic.chi_2), distance_mpc=100.0, inclination=0.0,
        total_mass=v.TOTAL_MASS,
    ).aligned()
    surrogate = model.coprecessing_modes_dict(frequencies, params)
    eob = model.coprecessing_modes_dict(frequencies, params, source="eob")
    per_mode = {
        key: validator.full_waveform_mismatch(
            {key: eob[key][band]}, {key: surrogate[key][band]}, frequencies=fb
        )
        for key in surrogate
    }
    summed = validator.full_waveform_mismatch(
        {k: eob[k][band] for k in eob}, {k: surrogate[k][band] for k in surrogate},
        frequencies=fb,
    )
    print(f"q={intrinsic.mass_ratio:.2f} chi=({intrinsic.chi_1:+.2f},{intrinsic.chi_2:+.2f}) "
          f"L=({intrinsic.lambda_1:.0f},{intrinsic.lambda_2:.0f})  "
          + "  ".join(f"{k}:{m:.1e}" for k, m in per_mode.items())
          + f"  all-face-on-sum:{summed:.1e}")
