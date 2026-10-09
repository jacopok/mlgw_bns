"""Tests for the regressed precession angles (:mod:`mlgw_bns.precession_regression`)."""

from dataclasses import astuple

import numpy as np
import pytest

from mlgw_bns.precessing_model import PrecessingModel

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from mlgw_bns.batched_precession import (  # noqa: E402
    N_STEPS,
    TabulatedAngles,
    batch_arguments,
    integrate_angles,
)
from mlgw_bns.precession_regression import (  # noqa: E402
    AngleGrid,
    PrecessionRegressor,
    RegressedAngles,
    TrainingRanges,
    _tail,
    carrier_table,
    fit_all_envelopes,
    fit_envelopes,
    frozen_precession_constants,
    nutation,
    reference_frame,
    training_data,
)
from mlgw_bns.twist_waveform import _pn_precession_derivatives  # noqa: E402
from mlgw_bns.taylorf2 import SUN_MASS_SECONDS  # noqa: E402

from .test_batched_precession import BINARIES, FREQUENCIES  # noqa: E402

ROWS = np.array([
    # two in-plane spins, unequal masses
    [1.7, 300.0, 900.0, 0.3, -0.1, 0.1, -0.1, 0.25, -0.2],
    # nearly equal masses: the two carriers nearly coincide
    [1.02, 50.0, 2000.0, -0.25, 0.2, -0.15, 0.1, 0.1, 0.3],
    # aligned spins
    [1.1, 1000.0, 1500.0, 0.0, 0.0, 0.2, 0.0, 0.0, -0.1],
])
OMEGA_REFERENCE = np.array([6e-4, 1.5e-3, 9e-4])


def rotation_matrix(alpha, beta, gamma):
    """R_z(alpha) R_y(beta) R_z(-gamma), stacked along the first axis."""

    def about_z(angle):
        c, s = np.cos(angle), np.sin(angle)
        zero, one = np.zeros_like(angle), np.ones_like(angle)
        return np.stack([np.stack([c, -s, zero], -1), np.stack([s, c, zero], -1),
                         np.stack([zero, zero, one], -1)], -2)

    c, s = np.cos(beta), np.sin(beta)
    zero, one = np.zeros_like(beta), np.ones_like(beta)
    about_y = np.stack([np.stack([c, zero, s], -1), np.stack([zero, one, zero], -1),
                        np.stack([-s, zero, c], -1)], -2)
    return about_z(alpha) @ about_y @ about_z(-gamma)


def rotation_difference(angles_1, angles_2):
    """The angle between the rotations of two sets of Euler angles, where it
    is small (the Frobenius norm of their difference, over the square root of
    two)."""
    difference = rotation_matrix(*angles_1) - rotation_matrix(*angles_2)
    return np.sqrt(np.sum(difference**2, axis=(-2, -1)) / 2)


@pytest.fixture(scope="module")
def grid():
    return AngleGrid()


def test_envelopes_reproduce_the_integrated_rotation(grid):
    """With envelopes and switch state fitted to a binary's own angles (no
    regression), the representation gives back the integrated rotation on
    both sides of the reference frequency, and the integration from the
    switch carries it to the merger."""
    integrate = jax.jit(lambda row, omega: astuple(integrate_angles(
        row[0], row[1], row[2], row[3:6], row[6:9], omega / np.pi,
        grid.omega_min / np.pi, N_STEPS, grid.omega_max / np.pi,
    )))
    momega = np.geomspace(grid.omega_min, grid.omega_max, 3000)
    below = grid.x_of(momega) <= grid.x_switch
    x, zeta, g, switch = training_data(grid, ROWS, OMEGA_REFERENCE, n_jobs=1)
    for i, (row, omega_reference) in enumerate(zip(ROWS, OMEGA_REFERENCE)):
        frame = reference_frame(np, grid, row, omega_reference)
        coefficients, residuals = fit_envelopes(grid, frame, x[i], zeta[i], g[i])
        assert max(residuals) < 1e-5
        phases, rates = carrier_table(np, grid, frame)
        tail = _tail(grid, frame, *(jax.numpy.asarray(a) for a in (coefficients, phases, rates, switch[i])))
        angles = RegressedAngles(
            coefficients, phases, rates, frame.rotation, frame.g_reference,
            *(np.asarray(a) for a in tail), grid,
        )
        exact = TabulatedAngles(*(np.asarray(a) for a in integrate(row, omega_reference)))
        expected = [np.asarray(a) for a in exact.at_momega(jax.numpy.asarray(momega))]
        difference = rotation_difference(expected, angles.at_momega(momega))
        assert np.max(difference[below]) < 1e-5
        assert np.max(difference[~below]) < 2e-4


@pytest.mark.parametrize("q", [1.0, 1.02, 1.5])
def test_nutation_matches_the_frozen_precession(grid, q):
    """The closed-form nutation frequency is that of the full right-hand side
    integrated at a fixed orbital frequency, from equal masses (where the
    linearized beat is ~50% off) to unequal ones."""
    from scipy.integrate import solve_ivp

    row = ROWS[1].copy()
    row[0] = q
    omega = 1e-3
    _, state, _ = integrate_angles(
        row[0], row[1], row[2], row[3:6], row[6:9], OMEGA_REFERENCE[1] / np.pi,
        omega / np.pi, 256, full_state=True,
    )
    s_a, s_b, l_hat = (np.asarray(state[0, i:i + 3]) for i in (0, 3, 6))
    nu = q / (1 + q) ** 2
    a10 = float(reference_frame(np, grid, row, omega).a10_tidal)

    def rhs(_, y):
        d_sa, d_sb, d_l, *_ = _pn_precession_derivatives(nu, q, omega, y[0:3], y[3:6], y[6:9], a10)
        return np.concatenate([d_sa, d_sb, d_l])

    def projection_rate(t, y):
        d = rhs(t, y)
        return d[0:3] @ y[6:9] + y[0:3] @ d[6:9]

    l_magnitude, alpha_a, alpha_b, k = frozen_precession_constants(np, nu, q, omega)
    u_a, u_b = s_a @ l_hat, s_b @ l_hat
    y = u_a / q + u_b
    j = l_magnitude * l_hat + s_a + s_b
    frequency, _, _ = nutation(
        np, q, s_a @ s_a, s_b @ s_b, l_magnitude, alpha_a, alpha_b, k,
        alpha_a * u_a + alpha_b * u_b - k * y**2, j @ j, y,
    )
    solution = solve_ivp(
        rhs, (0, 5 * 2 * np.pi / frequency), np.concatenate([s_a, s_b, l_hat]),
        method="DOP853", rtol=1e-11, atol=1e-14, events=projection_rate,
    )
    extrema = solution.t_events[0]
    assert len(extrema) >= 8
    integrated = 2 * np.pi / np.mean(extrema[2:] - extrema[:-2])
    assert abs(frequency / integrated - 1) < 5e-4


def test_grids_pickled_before_the_elliptic_beat_keep_the_linear_one():
    import pickle

    state = dict(AngleGrid().__dict__)
    del state["elliptic_beat"]
    old = AngleGrid.__new__(AngleGrid)
    old.__setstate__(state)
    assert not old.elliptic_beat
    assert pickle.loads(pickle.dumps(AngleGrid())).elliptic_beat


@pytest.fixture(scope="module")
def regressor(grid):
    """A small regressor: enough to exercise the machinery, not to be accurate."""
    intrinsic, omega_reference = TrainingRanges().sample(48, seed=3)
    x, zeta, g, switch = training_data(
        grid, intrinsic, omega_reference, n_steps=512, batch=16, n_jobs=1
    )
    coefficients, _ = fit_all_envelopes(grid, intrinsic, omega_reference, x, zeta, g, n_jobs=1)
    return PrecessionRegressor.train(
        intrinsic, omega_reference, coefficients, switch, n_components=(8, 8), grid=grid
    )


def test_numpy_and_jax_angles_agree(regressor):
    angles = jax.jit(regressor.jax_angles())(ROWS, 2.8, OMEGA_REFERENCE / (np.pi * 2.8 * SUN_MASS_SECONDS))
    momega = np.geomspace(3e-4, 0.25, 50)
    for i, row in enumerate(ROWS):
        expected = regressor.angles(row, OMEGA_REFERENCE[i]).at_momega(momega)
        one = jax.tree_util.tree_map(lambda array: array[i], angles)
        got = [np.asarray(a) for a in one.at_momega(jax.numpy.asarray(momega))]
        assert np.max(rotation_difference(expected, got)) < 1e-9


def test_regressor_in_the_jax_predictor(default_model, regressor, tmp_path):
    """The regressed angles stand in for the integration in jax_predict, and
    survive a save and load."""
    filename = str(tmp_path / "precession.joblib")
    regressor.save(filename)
    loaded = PrecessionRegressor.load(filename)
    arguments = batch_arguments(BINARIES, FREQUENCIES)
    precessing = PrecessingModel(default_model)
    h_plus, h_cross = jax.jit(precessing.jax_predict(precession_regressor=loaded))(*arguments)
    reference, _ = jax.jit(precessing.jax_predict())(*arguments)
    assert h_plus.shape == reference.shape
    assert np.all(np.isfinite(np.asarray(h_plus))) and np.all(np.isfinite(np.asarray(h_cross)))
