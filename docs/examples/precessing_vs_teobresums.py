"""The precessing JAX predictor against TEOBResumS, for one binary."""

import jax
import numpy as np
from EOBRun_module import EOBRunPy

from mlgw_bns import Model
from mlgw_bns.batched_precession import batch_arguments
from mlgw_bns.higher_order_modes import Mode, mode_to_k
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
    azimuth=0.3,
    total_mass=2.8,
    # TEOBResumS imposes the spins at 0.95 initial_frequency
    reference_frequency_hz=0.95 * initial_frequency,
)

largest_l = max(mode.l for mode in model.modes)
f, hp_re, hp_im, hc_re, hc_im, _, hlm, dynamics = EOBRunPy(dict(
    M=params.total_mass,
    q=params.mass_ratio,
    LambdaAl2=params.lambda_1,
    LambdaBl2=params.lambda_2,
    chi1x=params.chi_1[0], chi1y=params.chi_1[1], chi1z=params.chi_1[2],
    chi2x=params.chi_2[0], chi2y=params.chi_2[1], chi2z=params.chi_2[2],
    distance=params.distance_mpc,
    inclination=params.inclination,
    coalescence_angle=np.pi / 2 + params.azimuth,
    use_spins=2,
    domain=1,
    initial_frequency=initial_frequency,
    df=1 / 2048,
    srate_interp=4096.0,
    interp_uniform_grid="yes",
    use_geometric_units="no",
    output_hpc="no",
    arg_out="yes",
    # the co-precessing multipoles of the model, twisted into every
    # inertial multipole they feed
    use_mode_lm=sorted(mode_to_k(mode) for mode in model.modes),
    use_mode_lm_inertial=list(range(mode_to_k(Mode(largest_l, largest_l)) + 1)),
))
# mlgw_bns' Fourier transform is the complex conjugate of TEOBResumS', and
# its h_x has the opposite sign
f = np.asarray(f)
hp_teob = np.conj(np.asarray(hp_re) + 1j * np.asarray(hp_im))
hc_teob = -np.conj(np.asarray(hc_re) + 1j * np.asarray(hc_im))

# TEOBResumS' orbital phase is zero where its integration starts
params.reference_phase = precessing.teob_reference_phase(
    params,
    start_momega=dynamics["MOmega"][0],
    start_orbital_phase=dynamics["phi"][0],
    start_phase_22=hlm["1"][1][0],  # (l, m) = (2, 2)
)

# compare on (a subset of) TEOBResumS' own grid
nodes = np.flatnonzero((f >= 20.0) & (f <= 2000.0))[::64]
predict = jax.jit(precessing.jax_predict())
hp, hc = (np.asarray(h[0]) for h in predict(*batch_arguments([params], f[nodes])))

# Einstein Telescope PSD, time shift and constant phase optimised
validator = ValidateModel(model.mode_models[Mode(2, 2)])
for name, ours, theirs in (("h_+", hp, hp_teob), ("h_x", hc, hc_teob)):
    mismatch = validator.mismatch(theirs[nodes], ours, frequencies=f[nodes], max_delta_t=1e-3)
    print(f"{name}: mismatch {mismatch:.1e}")
