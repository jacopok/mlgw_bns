"""The precessing JAX predictor against TEOBResumS, for one binary."""

import jax
import numpy as np

from mlgw_bns import Model
from mlgw_bns.dataset_generation import eobrun
from mlgw_bns.batched_precession import batch_arguments
from mlgw_bns.higher_order_modes import Mode, mode_to_k, srate_interp_scaling
from mlgw_bns.model_validation import ValidateModel
from mlgw_bns.precessing_model import PrecessingModel, PrecessingParametersWithExtrinsic

model = Model.default_for_testing()
precessing = PrecessingModel(model)

initial_frequency = 10.0  # Hz, where TEOBResumS starts
params = PrecessingParametersWithExtrinsic(
    mass_ratio=1.3,
    lambda_1=400.0,
    lambda_2=600.0,
    chi_1=(0.3, 0.0, 0.05),
    chi_2=(0.0, -0.3, 0.02),
    distance_mpc=100.0,
    inclination=1.0,
    azimuth=np.pi / 2 + 0.3,  # TEOBResumS' coalescence_angle: convention="teob"
    total_mass=2.8,
    # TEOBResumS imposes the spins at 0.95 initial_frequency
    reference_frequency_hz=0.95 * initial_frequency,
)

largest_l = max(mode.l for mode in model.modes)
f, hp_re, hp_im, hc_re, hc_im, _, hlm, dynamics = eobrun(dict(
    M=params.total_mass,
    q=params.mass_ratio,
    LambdaAl2=params.lambda_1,
    LambdaBl2=params.lambda_2,
    chi1x=params.chi_1[0], chi1y=params.chi_1[1], chi1z=params.chi_1[2],
    chi2x=params.chi_2[0], chi2y=params.chi_2[1], chi2z=params.chi_2[2],
    distance=params.distance_mpc,
    inclination=params.inclination,
    coalescence_angle=params.azimuth,
    use_spins=2,
    domain=1,
    initial_frequency=initial_frequency,
    df=1 / 2048,
    # as the training data: lower, the m = 3, 4 mergers sit above
    # TEOBResumS' SPA ceiling, whose linear phase continuation rings in band
    srate_interp=model.dataset.srate_hz * model.dataset.total_mass
    / params.total_mass * srate_interp_scaling(model.modes),
    interp_uniform_grid="yes",
    use_geometric_units="no",
    output_hpc="no",
    arg_out="yes",
    # the co-precessing multipoles of the model, twisted into every
    # inertial multipole they feed
    use_mode_lm=sorted(mode_to_k(mode) for mode in model.modes),
    use_mode_lm_inertial=list(range(mode_to_k(Mode(largest_l, largest_l)) + 1)),
))
# TEOBResumS' own polarizations, in its Fourier convention
f = np.asarray(f)
hp_teob = np.asarray(hp_re) + 1j * np.asarray(hp_im)
hc_teob = np.asarray(hc_re) + 1j * np.asarray(hc_im)

# TEOBResumS' orbital phase is zero where its integration starts
params.reference_phase = precessing.teob_reference_phase(
    params,
    start_momega=dynamics["MOmega"][0],
    start_orbital_phase=dynamics["phi"][0],
    start_phase_22=hlm["1"][1][0],  # (l, m) = (2, 2)
)

# compare on (a subset of) TEOBResumS' own grid
nodes = np.flatnonzero((f >= 20.0) & (f <= 2000.0))[::64]
predict = jax.jit(precessing.jax_predict(convention="teob"))
hp, hc = (np.asarray(h[0]) for h in predict(*batch_arguments([params], f[nodes])))

# Einstein Telescope PSD, time shift and constant phase optimised
validator = ValidateModel(model.mode_models[Mode(2, 2)])
for name, ours, theirs in (("h_+", hp, hp_teob), ("h_x", hc, hc_teob)):
    mismatch = validator.mismatch(theirs[nodes], ours, frequencies=f[nodes], max_delta_t=1e-3)
    print(f"{name}: mismatch {mismatch:.1e}")
