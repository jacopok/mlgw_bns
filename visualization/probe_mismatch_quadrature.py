"""Is the precessing mismatch floor a quadrature artefact of the model grid?

validate_precessing_against_teob evaluates the mismatch integral on the
surrogate's own sparse grid (Delta f ~ 0.3% f). For a multi-mode precessing
strain the integrand carries cross terms between co-precessing multipoles of
different n, whose stationary times differ by hundreds of seconds in the
early inspiral: they oscillate on ~mHz scales and should average out, but a
sparse trapezoid rule aliases them instead. Here the same waveforms are
compared on the model grid and on a uniform grid fine enough to resolve them.

Run with: python visualization/probe_mismatch_quadrature.py [N_BINARIES]
"""
import sys

import numpy as np

import validate_precessing_against_teob as v
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model import Model
from mlgw_bns.model_validation import ValidateModel
from mlgw_bns.precessing_model import PrecessingModel, PrecessingParametersWithExtrinsic

BAND_HI = 512.0  # keeps the fine grid (df = 1/512 Hz) at ~250k points
FINE_DF = 1.0 / 512.0

model = Model.default_for_testing(modes=v.MODES)
v.SRATE_HZ = model.dataset.effective_srate_hz
precessing = PrecessingModel(model)
validator = ValidateModel(model.mode_models[Mode(2, 2)])
coarse = validator.frequencies
coarse = coarse[(coarse >= v.BAND_LO) & (coarse <= BAND_HI)]
fine = np.arange(v.BAND_LO, BAND_HI, FINE_DF)

rng = np.random.default_rng(v.SEED)
generator = model.dataset.make_parameter_generator(v.SEED)
n_binaries = int(sys.argv[1]) if len(sys.argv) > 1 else 6
for index in range(n_binaries):
    intrinsic = next(generator)
    chi_1 = v.random_spin_vector(rng, intrinsic.chi_1)
    chi_2 = v.random_spin_vector(rng, intrinsic.chi_2)
    orientation = v.random_orientation(rng)
    params = PrecessingParametersWithExtrinsic(
        mass_ratio=intrinsic.mass_ratio, lambda_1=intrinsic.lambda_1,
        lambda_2=intrinsic.lambda_2, chi_1=chi_1, chi_2=chi_2,
        distance_mpc=v.DISTANCE_MPC, inclination=orientation["inclination"],
        azimuth=orientation["azimuth"], total_mass=v.TOTAL_MASS,
        reference_frequency_hz=v.SPIN_REFERENCE_FREQUENCY_HZ,
    )
    f_plus, f_cross = v.antenna_patterns(orientation["theta"], orientation["phi"],
                                         orientation["psi"])
    angles = precessing.euler_angles(params, float(validator.frequencies[0]))
    try:
        results = {}
        for name, grid in (("model grid", coarse), ("fine grid", fine)):
            hp_t, hc_t, inside = v.teob_polarizations(
                intrinsic, chi_1, chi_2, params.inclination, params.azimuth, grid)
            teob = f_plus * hp_t + f_cross * hc_t
            for source in ("surrogate", "eob"):
                hp, hc = precessing.predict(grid, params, source=source, angles=angles,
                                            reanchor=False)
                strain = (f_plus * hp + f_cross * hc)[inside]
                results[(name, source)] = validator.full_waveform_mismatch(
                    {(2, 2): teob}, {(2, 2): strain}, frequencies=grid[inside])
    except RuntimeError as error:
        print(f"binary {index + 1}: TEOBResumS failed ({error})")
        continue
    print(f"binary {index + 1}: max beta {angles.beta.max():.3f}  "
          + "  ".join(f"{g}/{s} {m:.2e}" for (g, s), m in results.items()), flush=True)
