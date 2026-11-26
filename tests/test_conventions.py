"""The ``convention`` switch: LALSuite's and TEOBResumS' conventions.

The comparisons with the reference codes use a joint overlap of both
polarizations, maximised over a time shift only: a constant phase, a
polarization angle or a per-polarization optimisation would absorb exactly
the signs these tests are about (the sign of h_x, the azimuth, the Fourier
convention). See docs/explanation/conventions.md.
"""

import numpy as np
import pytest

from mlgw_bns import Model, ParametersWithExtrinsic
from mlgw_bns.conventions import (
    LAL_AZIMUTH,
    aligned_internal_phase,
    check_convention,
    external_azimuth,
    internal_azimuth,
)
from mlgw_bns.precessing_model import PrecessingModel, PrecessingParametersWithExtrinsic


def joint_overlap(a_plus, a_cross, b_plus, b_cross, fixed_phase=True):
    """Normalised (a|b) of the two polarizations together on a uniform
    grid, maximised over a time shift (and over a constant phase if not
    ``fixed_phase``)."""
    n = 8 * len(a_plus)
    z = np.fft.ifft(np.conj(a_plus) * b_plus + np.conj(a_cross) * b_cross, n) * n
    norm = np.sqrt(
        (np.vdot(a_plus, a_plus) + np.vdot(a_cross, a_cross)).real
        * (np.vdot(b_plus, b_plus) + np.vdot(b_cross, b_cross)).real
    )
    return (z.real.max() if fixed_phase else np.abs(z).max()) / norm


@pytest.fixture(scope="module")
def model():
    return Model.default_for_testing()


def test_the_conversions_invert_each_other():
    for convention in ("lal", "teob"):
        for azimuth in (0.0, 0.3, 2.0):
            assert np.isclose(
                external_azimuth(internal_azimuth(azimuth, convention), convention), azimuth)
    with pytest.raises(ValueError):
        check_convention("lalsuite")
    # the same binary: LALSuite's pi / 2 is TEOBResumS' coalescence angle pi
    assert np.isclose(internal_azimuth(LAL_AZIMUTH, "lal"), internal_azimuth(np.pi, "teob"))
    assert np.isclose(aligned_internal_phase(0.0, "lal"), internal_azimuth(np.pi, "teob"))


def test_teob_is_the_conjugate_of_lal(model):
    """The same binary in the two conventions: the polarizations are the
    complex conjugates of each other, aligned and precessing."""
    frequencies = np.linspace(20.0, 1000.0, 200)
    aligned = ParametersWithExtrinsic(
        mass_ratio=1.3, lambda_1=400.0, lambda_2=600.0, chi_1=0.1, chi_2=-0.05,
        distance_mpc=100.0, inclination=1.0, total_mass=2.8, coalescence_phase=0.4,
    )
    lal = model.predict(frequencies, aligned)
    teob_angle = external_azimuth(aligned_internal_phase(0.4, "lal"), "teob")
    teob = model.predict(
        frequencies, ParametersWithExtrinsic(**{**aligned.__dict__, "coalescence_phase": teob_angle}),
        convention="teob")
    for x, y in zip(lal, teob):
        assert np.allclose(np.conj(x), y, rtol=1e-12, atol=1e-12 * np.abs(x).max())

    precessing = PrecessingModel(model)
    params = PrecessingParametersWithExtrinsic(
        mass_ratio=1.3, lambda_1=400.0, lambda_2=600.0, chi_1=(0.3, 0.0, 0.05),
        chi_2=(0.0, -0.3, 0.02), distance_mpc=100.0, inclination=1.0, total_mass=2.8,
        azimuth=0.7, reference_phase=0.2, reference_frequency_hz=20.0,
    )
    lal = precessing.predict(frequencies, params)
    params.azimuth = external_azimuth(internal_azimuth(0.7, "lal"), "teob")
    teob = precessing.predict(frequencies, params, convention="teob")
    for x, y in zip(lal, teob):
        assert np.allclose(np.conj(x), y, rtol=1e-10, atol=1e-10 * np.abs(x).max())


def test_euler_angles_in_convention(model):
    params = PrecessingParametersWithExtrinsic(
        mass_ratio=1.3, lambda_1=400.0, lambda_2=600.0, chi_1=(0.3, 0.0, 0.05),
        chi_2=(0.0, -0.3, 0.02), distance_mpc=100.0, inclination=1.0, total_mass=2.8,
        reference_frequency_hz=20.0,
    )
    angles = PrecessingModel(model).euler_angles(params, 20.0)
    alpha, beta, gamma = angles.in_convention("lal")
    assert np.array_equal(alpha, angles.alpha) and np.array_equal(beta, angles.beta)
    assert np.array_equal(gamma, -angles.gamma)
    # LALSuite's minimal rotation, d gamma = -d alpha cos beta
    momega = np.geomspace(angles.momega[5], angles.momega[-5], 400)
    a, b, g = angles.at_momega(momega)
    ratio = np.gradient(-g, momega) / (np.gradient(a, momega) * np.cos(b))
    assert np.allclose(np.median(ratio), -1.0, atol=1e-3)
    assert np.array_equal(angles.in_convention("teob")[2], angles.gamma)


def test_teob_convention_reproduces_teobresums(model):
    """TEOBResumS' own frequency-domain precessing polarizations, with its
    own coalescence_angle and Fourier convention, as convention="teob"
    gives them; the orbital phase from teob_reference_phase."""
    pytest.importorskip("EOBRun_module")
    try:  # TEOBResumS' MSUN_S -> IAU, 1 + 1.6e-8
        from mlgw_bns.constants import TEOBRESUMS_MASS_FACTOR
    except ImportError:
        TEOBRESUMS_MASS_FACTOR = 1.0
    from mlgw_bns.dataset_generation import eobrun
    from mlgw_bns.higher_order_modes import Mode, mode_to_k, srate_interp_scaling

    precessing = PrecessingModel(model)
    initial_frequency = 20.0
    params = PrecessingParametersWithExtrinsic(
        mass_ratio=1.3, lambda_1=400.0, lambda_2=600.0, chi_1=(0.3, 0.0, 0.05),
        chi_2=(0.0, -0.3, 0.02), distance_mpc=100.0, inclination=1.0,
        azimuth=np.pi / 2 + 0.3,  # TEOBResumS' coalescence_angle
        total_mass=2.8, reference_frequency_hz=0.95 * initial_frequency,
    )
    largest_l = max(mode.l for mode in model.modes)
    f, hp_re, hp_im, hc_re, hc_im, _, hlm, dynamics = eobrun(dict(
        M=params.total_mass * TEOBRESUMS_MASS_FACTOR, q=params.mass_ratio,
        LambdaAl2=params.lambda_1, LambdaBl2=params.lambda_2,
        chi1x=params.chi_1[0], chi1y=params.chi_1[1], chi1z=params.chi_1[2],
        chi2x=params.chi_2[0], chi2y=params.chi_2[1], chi2z=params.chi_2[2],
        distance=params.distance_mpc, inclination=params.inclination,
        coalescence_angle=params.azimuth, use_spins=2, domain=1,
        initial_frequency=initial_frequency, df=1 / 2048,
        srate_interp=model.dataset.srate_hz * model.dataset.total_mass
        / params.total_mass * srate_interp_scaling(model.modes),
        interp_uniform_grid="yes", use_geometric_units="no", output_hpc="no",
        arg_out="yes",
        use_mode_lm=sorted(mode_to_k(mode) for mode in model.modes),
        use_mode_lm_inertial=list(range(mode_to_k(Mode(largest_l, largest_l)) + 1)),
    ))
    params.reference_phase = precessing.teob_reference_phase(
        params, start_momega=dynamics["MOmega"][0],
        start_orbital_phase=dynamics["phi"][0], start_phase_22=hlm["1"][1][0],
    )
    f = np.asarray(f)
    nodes = np.flatnonzero((f >= 40.0) & (f <= 1000.0))[::8]
    teob = [(np.asarray(re) + 1j * np.asarray(im))[nodes]
            for re, im in ((hp_re, hp_im), (hc_re, hc_im))]
    hp, hc = precessing.predict(f[nodes], params, convention="teob")
    assert joint_overlap(hp, hc, *teob) > 0.99  # 0.9989
    assert joint_overlap(hp, -hc, *teob, fixed_phase=False) < 0.5  # 0.14
    assert joint_overlap(np.conj(hp), np.conj(hc), *teob, fixed_phase=False) < 0.5


@pytest.mark.parametrize("tilt", [0.6, 1.2])
def test_lal_convention_reproduces_imrphenomxphm(model, tilt):
    """LALSuite's IMRPhenomXPHM from the same LVK parameters, through
    lvk_to_precessing. The two models fix the orbital phase at f_ref each in
    its own way (as do IMRPhenomXPHM and IMRPhenomPv2 between themselves), so
    a constant phase is left free; the sign of h_x and the Fourier
    convention are not."""
    lal = pytest.importorskip("lal")
    lalsimulation = pytest.importorskip("lalsimulation")
    from mlgw_bns.spin_conversion import lvk_to_precessing

    mass_1, mass_2, f_ref = 1.5, 1.2, 20.0
    lvk = dict(theta_jn=0.8, phi_jl=2.0, tilt_1=tilt, tilt_2=0.9 * tilt, phi_12=1.0,
               a_1=0.5, a_2=0.3)
    orientation = lvk_to_precessing(**lvk, mass_1=mass_1, mass_2=mass_2,
                                    reference_frequency_hz=f_ref, phase=0.4)
    params = PrecessingParametersWithExtrinsic(
        mass_ratio=mass_1 / mass_2, lambda_1=0.0, lambda_2=0.0,
        chi_1=tuple(orientation.chi_1), chi_2=tuple(orientation.chi_2),
        distance_mpc=100.0, inclination=float(orientation.inclination),
        azimuth=float(orientation.azimuth), total_mass=mass_1 + mass_2,
        reference_phase=float(orientation.reference_phase), reference_frequency_hz=f_ref,
    )
    frequencies = np.arange(20.0, 1024.0, 1 / 16)
    hp, hc = PrecessingModel(model).predict(frequencies, params)

    sequence = lal.CreateREAL8Vector(len(frequencies))
    sequence.data[:] = frequencies
    spins = lalsimulation.SimInspiralTransformPrecessingNewInitialConditions(
        lvk["theta_jn"], lvk["phi_jl"], lvk["tilt_1"], lvk["tilt_2"], lvk["phi_12"],
        lvk["a_1"], lvk["a_2"], mass_1 * lal.MSUN_SI, mass_2 * lal.MSUN_SI, f_ref, 0.4)
    reference = lalsimulation.SimInspiralChooseFDWaveformSequence(
        0.4, mass_1 * lal.MSUN_SI, mass_2 * lal.MSUN_SI, *spins[1:], f_ref,
        100e6 * lal.PC_SI, spins[0], lal.CreateDict(), lalsimulation.IMRPhenomXPHM, sequence)
    hp_lal, hc_lal = (h.data.data for h in reference)
    assert joint_overlap(hp, hc, hp_lal, hc_lal, fixed_phase=False) > 0.98  # 0.996-0.998
    assert joint_overlap(hp, -hc, hp_lal, hc_lal, fixed_phase=False) < 0.5
