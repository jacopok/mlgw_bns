"""Tests for the LVK spin angles (:mod:`mlgw_bns.spin_conversion`)."""

import numpy as np
import pytest

from mlgw_bns.precessing_model import PrecessingParametersWithExtrinsic
from mlgw_bns.spin_conversion import lal_precessing_spins, lvk_to_precessing


def random_angles(n, seed=1):
    rng = np.random.default_rng(seed)
    return dict(
        theta_jn=np.arccos(rng.uniform(-1, 1, n)),
        phi_jl=rng.uniform(0, 2 * np.pi, n),
        tilt_1=np.arccos(rng.uniform(-1, 1, n)),
        tilt_2=np.arccos(rng.uniform(-1, 1, n)),
        phi_12=rng.uniform(0, 2 * np.pi, n),
        a_1=rng.uniform(0, 0.9, n),
        a_2=rng.uniform(0, 0.9, n),
        mass_1=rng.uniform(1.2, 2.5, n),
        mass_2=rng.uniform(1.0, 1.2, n),
        reference_frequency_hz=rng.uniform(5, 50, n),
        phase=rng.uniform(0, 2 * np.pi, n),
    )


def test_matches_lalsimulation():
    """The port reproduces SimInspiralTransformPrecessingNewInitialConditions
    (the residual is the solar-mass time constant, to 1e-11)."""
    lal = pytest.importorskip("lal")
    lalsimulation = pytest.importorskip("lalsimulation")
    angles = random_angles(300)
    ours = np.array(lal_precessing_spins(**angles))
    theirs = np.array([
        lalsimulation.SimInspiralTransformPrecessingNewInitialConditions(
            angles["theta_jn"][i], angles["phi_jl"][i], angles["tilt_1"][i],
            angles["tilt_2"][i], angles["phi_12"][i], angles["a_1"][i], angles["a_2"][i],
            angles["mass_1"][i] * lal.MSUN_SI, angles["mass_2"][i] * lal.MSUN_SI,
            angles["reference_frequency_hz"][i], angles["phase"][i],
        )
        for i in range(300)
    ]).T
    np.testing.assert_allclose(ours, theirs, atol=1e-9)


def test_phase_rotates_the_spins_and_line_of_sight_together():
    """LALSuite's frame with ``phase`` is ours with the spins and the line of
    sight rotated by ``-phase`` about L: the separation is fixed to x there,
    here the separation moves instead."""
    angles = random_angles(50)
    inclination, *spins = lal_precessing_spins(**angles)
    orientation = lvk_to_precessing(**angles)
    np.testing.assert_allclose(orientation.inclination, inclination)
    phase = angles["phase"]
    for ours, theirs in ((orientation.chi_1, spins[:3]), (orientation.chi_2, spins[3:])):
        x = ours[:, 0] * np.cos(-phase) - ours[:, 1] * np.sin(-phase)
        y = ours[:, 0] * np.sin(-phase) + ours[:, 1] * np.cos(-phase)
        np.testing.assert_allclose(x, theirs[0], atol=1e-12)
        np.testing.assert_allclose(y, theirs[1], atol=1e-12)
        np.testing.assert_allclose(ours[:, 2], theirs[2], atol=1e-12)
    np.testing.assert_allclose(orientation.azimuth, np.pi / 2)
    np.testing.assert_allclose(orientation.reference_phase, phase)


def test_aligned_spins():
    """With no tilt J is along L: the inclination is theta_jn and the spins
    are along z, whatever phi_jl and phi_12."""
    angles = random_angles(20)
    angles["tilt_1"] = np.zeros(20)
    angles["tilt_2"] = np.full(20, np.pi)
    orientation = lvk_to_precessing(**angles)
    np.testing.assert_allclose(orientation.inclination, angles["theta_jn"], atol=1e-12)
    np.testing.assert_allclose(orientation.chi_1[:, :2], 0.0, atol=1e-12)
    np.testing.assert_allclose(orientation.chi_1[:, 2], angles["a_1"], atol=1e-12)
    np.testing.assert_allclose(orientation.chi_2[:, 2], -angles["a_2"], atol=1e-12)


def test_spin_magnitudes_and_tilts_are_kept():
    angles = random_angles(50)
    orientation = lvk_to_precessing(**angles)
    for chi, a, tilt in ((orientation.chi_1, angles["a_1"], angles["tilt_1"]),
                         (orientation.chi_2, angles["a_2"], angles["tilt_2"])):
        np.testing.assert_allclose(np.linalg.norm(chi, axis=1), a, atol=1e-12)
        np.testing.assert_allclose(chi[:, 2], a * np.cos(tilt), atol=1e-12)


def test_jax_matches_numpy():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    angles = random_angles(30)
    ours = lvk_to_precessing(**angles)
    theirs = jax.jit(lambda kw: lvk_to_precessing(**kw, xp=jnp))(
        {key: jnp.asarray(value) for key, value in angles.items()})
    for got, want in zip(theirs, ours):
        np.testing.assert_allclose(np.asarray(got), want, atol=1e-12)


def test_from_lvk():
    angles = {key: value[0] for key, value in random_angles(1).items()}
    params = PrecessingParametersWithExtrinsic.from_lvk(
        **angles, lambda_1=300.0, lambda_2=500.0, distance_mpc=40.0)
    orientation = lvk_to_precessing(**angles)
    assert params.mass_ratio == pytest.approx(angles["mass_1"] / angles["mass_2"])
    assert params.total_mass == pytest.approx(angles["mass_1"] + angles["mass_2"])
    np.testing.assert_allclose(params.chi_1_vector, orientation.chi_1)
    assert params.reference_frequency_hz == angles["reference_frequency_hz"]
    assert params.inclination == pytest.approx(orientation.inclination)
