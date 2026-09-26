"""Tests for the batched, per-mode evaluation (:mod:`mlgw_bns.batched`)."""

import dataclasses

import numpy as np
import pytest

from mlgw_bns.batched import BatchedSurrogate, mode_polarizations
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.mode_model import ParametersWithExtrinsic
from mlgw_bns.model import DEFAULT_MODES, Model

ALL_MODES = [(mode.l, mode.m) for mode in DEFAULT_MODES]
FOUR_MODES = [(2, 2), (2, 1), (3, 3), (4, 4)]

# From below the trained band at every mass (the post-Newtonian
# continuation) to above it at the heaviest (the zero padding).
FREQUENCIES = np.geomspace(1.5, 3000.0, 240)

# Relative agreement, per mode, between a single-row batched evaluation and
# `Model.predict_modes_dict`, which run the regressors with the same
# operations: what is left is the rounding of phases of up to ~1e6 rad
# (measured: at most 7e-9).
SINGLE_ROW_RTOL = 1e-7


def random_rows(n, seed, lambda_max=5000.0):
    rng = np.random.default_rng(seed)
    intrinsic = np.column_stack(
        [
            rng.uniform(1.0, 3.0, n),
            rng.uniform(5.0, lambda_max, n),
            rng.uniform(5.0, lambda_max, n),
            rng.uniform(-0.5, 0.5, n),
            rng.uniform(-0.5, 0.5, n),
        ]
    )
    return intrinsic, rng.uniform(2.0, 4.0, n), rng.uniform(10.0, 500.0, n)


def with_regressor_outputs_of(target, source, intrinsic):
    """Make ``target`` use ``source``'s regressor outputs for ``intrinsic``.

    The kernel-ridge sums cancel so strongly that their rounding depends on
    the batch size and the backend (see the module docstring); fixing
    their outputs isolates everything downstream, which must then agree
    to rounding.
    """
    xp = target._xp
    residuals = source._residuals(np, intrinsic)
    mode_phases = source._mode_phases(np, intrinsic)
    time_shifts = source._time_shifts(np, intrinsic)

    def rows(x):
        # the rows of `intrinsic` that `x` holds, in order
        x = np.asarray(x)
        return [int(np.argmin(np.abs(intrinsic - row).sum(axis=1))) for row in x]

    target._residuals = lambda _xp, x: [
        (xp.asarray(a[rows(x)]), xp.asarray(p[rows(x)])) for a, p in residuals
    ]
    target._mode_phases = lambda _xp, x: xp.asarray(mode_phases[rows(x)])
    target._time_shifts = lambda _xp, x: xp.asarray(time_shifts[rows(x)])
    return target


def test_single_rows_match_predict_modes_dict(default_model):
    intrinsic, total_mass, distance = random_rows(6, seed=1, lambda_max=12000.0)
    inclination = np.linspace(0.2, 2.9, len(intrinsic))
    for i in range(len(intrinsic)):
        amp, phase = default_model.predict_modes_amp_phase(
            intrinsic[i],
            total_mass[i],
            FREQUENCIES,
            modes=ALL_MODES,
            distance_mpc=distance[i],
        )
        h_plus, h_cross = mode_polarizations(amp, phase, ALL_MODES, inclination[i])
        expected = default_model.predict_modes_dict(
            FREQUENCIES,
            ParametersWithExtrinsic(
                *intrinsic[i], distance[i], inclination[i], total_mass[i]
            ),
        )
        for j, mode in enumerate(ALL_MODES):
            got = h_plus[0, j] - 1j * h_cross[0, j]
            scale = np.max(np.abs(expected[mode]))
            np.testing.assert_allclose(
                got / scale, expected[mode] / scale, rtol=0, atol=SINGLE_ROW_RTOL
            )


def test_polarizations_sum_to_predict(default_model):
    intrinsic, total_mass, distance = random_rows(1, seed=2)
    params = ParametersWithExtrinsic(*intrinsic[0], distance[0], 0.7, total_mass[0])
    amp, phase = default_model.predict_modes_amp_phase(
        intrinsic, total_mass, FREQUENCIES, distance_mpc=distance
    )
    h_plus, h_cross = mode_polarizations(amp, phase, default_model.modes, 0.7)
    expected_plus, expected_cross = default_model.predict(FREQUENCIES, params)
    scale = np.max(np.abs(expected_plus))
    np.testing.assert_allclose(
        h_plus.sum(axis=1)[0] / scale,
        expected_plus / scale,
        rtol=0,
        atol=SINGLE_ROW_RTOL,
    )
    np.testing.assert_allclose(
        h_cross.sum(axis=1)[0] / scale,
        expected_cross / scale,
        rtol=0,
        atol=SINGLE_ROW_RTOL,
    )


def test_batch_matches_single_rows_given_the_regressors(default_model):
    """Everything but the regressors is row-by-row identical in a batch."""
    intrinsic, total_mass, distance = random_rows(20, seed=3)
    reference = default_model.batched_surrogate(FOUR_MODES)
    surrogate = with_regressor_outputs_of(
        BatchedSurrogate(default_model, FOUR_MODES), reference, intrinsic
    )
    surrogate.chunk_rows = 7  # also exercise the threaded chunks
    amp, phase, tf = surrogate(
        intrinsic, total_mass, FREQUENCIES, distance, return_tf=True
    )
    for i in range(len(intrinsic)):
        row = surrogate(
            intrinsic[i : i + 1],
            total_mass[i],
            FREQUENCIES,
            distance[i],
            return_tf=True,
        )
        np.testing.assert_allclose(amp[i], row[0][0], rtol=1e-12, atol=0)
        np.testing.assert_allclose(phase[i], row[1][0], rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(tf[i], row[2][0], rtol=1e-9, atol=1e-9)


def test_batch_matches_single_rows_to_the_regressor_rounding(default_model):
    """End to end, batch and single rows differ only by the rounding of the
    ill-conditioned kernel-ridge sums (see the module docstring), which
    reaches ~1e-2 rad in phase near the merger."""
    intrinsic, total_mass, distance = random_rows(20, seed=4)
    amp, phase = default_model.predict_modes_amp_phase(
        intrinsic, total_mass, FREQUENCIES, modes=FOUR_MODES, distance_mpc=distance
    )
    for i in range(len(intrinsic)):
        row_amp, row_phase = default_model.predict_modes_amp_phase(
            intrinsic[i],
            total_mass[i],
            FREQUENCIES,
            modes=FOUR_MODES,
            distance_mpc=distance[i],
        )
        support = row_amp[0] > 0
        assert np.array_equal(support, amp[i] > 0)
        assert np.max(np.abs(phase[i] - row_phase[0])[support]) < 0.1
        assert np.max(np.abs(amp[i][support] / row_amp[0][support] - 1)) < 1e-2


def test_mode_subset_and_order(default_model):
    intrinsic, total_mass, _ = random_rows(3, seed=5)
    everything = default_model.predict_modes_amp_phase(
        intrinsic, total_mass, FREQUENCIES
    )
    subset = default_model.predict_modes_amp_phase(
        intrinsic, total_mass, FREQUENCIES, modes=[(4, 4), (2, 2)]
    )
    for j, mode in enumerate([(4, 4), (2, 2)]):
        k = ALL_MODES.index(mode)
        np.testing.assert_allclose(subset[0][:, j], everything[0][:, k], rtol=1e-12)
        np.testing.assert_allclose(subset[1][:, j], everything[1][:, k], rtol=1e-12)


def test_per_row_frequency_grids(default_model):
    intrinsic, total_mass, _ = random_rows(3, seed=6)
    grids = np.stack([FREQUENCIES * (1 + 0.01 * i) for i in range(3)])
    amp, phase = default_model.predict_modes_amp_phase(
        intrinsic, total_mass, grids, modes=FOUR_MODES
    )
    for i in range(3):
        row_amp, row_phase = default_model.predict_modes_amp_phase(
            intrinsic[i], total_mass[i], grids[i], modes=FOUR_MODES
        )
        # single rows on either side: bit-identical regressors
        np.testing.assert_allclose(amp[i], row_amp[0], rtol=1e-2)
        support = row_amp[0] > 0
        assert np.max(np.abs(phase[i] - row_phase[0])[support]) < 0.1


def test_frequency_regions(default_model):
    intrinsic, _, _ = random_rows(1, seed=7)
    frequencies = np.concatenate([[-1.0, 0.0], FREQUENCIES])
    amp, phase, tf = default_model.predict_modes_amp_phase(
        intrinsic, 4.0, frequencies, modes=FOUR_MODES, return_tf=True
    )
    # non-positive frequencies: zero
    assert np.all(amp[..., :2] == 0) and np.all(phase[..., :2] == 0)
    assert np.all(tf[..., :2] == 0)
    # above the trained band (at total mass 4, 2925 Hz * 2.8 / 4): zero
    fmax = (
        default_model.dataset.frequencies_hz[-1]
        * default_model.dataset.total_mass
        / 4.0
    )
    assert np.all(amp[..., frequencies > fmax] == 0)
    assert np.all(amp[..., (frequencies > 0) & (frequencies < fmax)] != 0)
    assert np.all(np.isfinite(phase)) and np.all(np.isfinite(tf))


def test_out_of_range_rows_are_nan(default_model):
    intrinsic, total_mass, _ = random_rows(4, seed=8)
    intrinsic[1, 0] = 3.5  # q
    total_mass[2] = 5.0
    intrinsic[3, 1] = 1.0  # lambda_1 below the model's range
    surrogate = default_model.batched_surrogate(FOUR_MODES)
    np.testing.assert_array_equal(
        surrogate.valid(intrinsic, total_mass), [True, False, False, False]
    )
    amp, phase = surrogate(intrinsic, total_mass, FREQUENCIES)
    assert np.all(np.isfinite(amp[0])) and np.all(np.isfinite(phase[0]))
    assert np.all(np.isnan(amp[1:])) and np.all(np.isnan(phase[1:]))


def test_parameter_ranges_setter_applies_to_every_mode():
    model = Model.default_for_testing()
    relaxed = dataclasses.replace(
        model.parameter_ranges, lambda1_range=(0.0, 5000.0), lambda2_range=(0.0, 5000.0)
    )
    reference_amplitude = model.mode_models[
        Mode(3, 3)
    ].dataset.amplitude_reference_parameters
    model.parameter_ranges = relaxed
    params = ParametersWithExtrinsic(1.5, 1.0, 2.0, 0.1, 0.0, 100.0, 0.5, 2.7)
    # every mode accepts it, not just the first one
    model.predict_modes_dict(FREQUENCIES, params)
    amp, _ = model.predict_modes_amp_phase(
        np.array([[1.5, 1.0, 2.0, 0.1, 0.0]]), 2.7, FREQUENCIES
    )
    assert np.all(np.isfinite(amp))
    # the checks change, not the trained datasets
    assert (
        model.mode_models[Mode(3, 3)].dataset.amplitude_reference_parameters
        == reference_amplitude
    )


def test_time_frequency_map(default_model):
    """t(f) is -(1/2 pi) d phase / d f, in and below the band."""
    intrinsic, _, _ = random_rows(3, seed=9)
    total_mass = np.array([2.2, 2.8, 3.6])
    frequencies = np.geomspace(2.0, 1500.0, 50)
    step = 1e-5
    amp, phase, tf = default_model.predict_modes_amp_phase(
        intrinsic, total_mass, frequencies, modes=FOUR_MODES, return_tf=True
    )
    # fixed regressor outputs, so the finite difference sees no jitter
    surrogate = with_regressor_outputs_of(
        BatchedSurrogate(default_model, FOUR_MODES),
        default_model.batched_surrogate(FOUR_MODES),
        intrinsic,
    )
    _, up = surrogate(intrinsic, total_mass, frequencies * (1 + step))
    _, down = surrogate(intrinsic, total_mass, frequencies * (1 - step))
    numerical = -(up - down) / (2 * step * frequencies) / (2 * np.pi)
    # well inside the spline intervals and below the merger
    inspiral = frequencies < 300
    np.testing.assert_allclose(tf[..., inspiral], numerical[..., inspiral], rtol=1e-4)
    # emitted before the merger, earlier for higher m at a given frequency
    assert np.all(tf[..., inspiral] < 0)
    assert np.all(tf[:, 3, inspiral] < tf[:, 0, inspiral])


def test_mode_subset_model_uses_the_right_mode_phase_columns(default_model):
    """A model loaded with a subset of the trained modes must read each
    mode's own column of the shared mode-phases predictor."""
    subset = Model.default_for_testing(modes=[Mode(*mode) for mode in FOUR_MODES])
    params = ParametersWithExtrinsic(1.4, 300.0, 700.0, 0.1, -0.1, 100.0, 0.8, 2.8)
    expected = default_model.predict_modes_dict(FREQUENCIES, params)
    got = subset.predict_modes_dict(FREQUENCIES, params)
    for mode in FOUR_MODES:
        np.testing.assert_allclose(got[mode], expected[mode], rtol=1e-10, atol=0)


def test_matches_teobresums(default_model):
    """The batched modes against the EOB ground truth, with tidal
    deformabilities up to 5000 (the public TEOBResumS release fails on
    some higher ones)."""
    from mlgw_bns.model_validation import ValidateModel

    validator = ValidateModel(default_model.mode_models[Mode(2, 2)])
    frequencies = validator.frequencies
    band = (frequencies >= 20.0) & (frequencies <= 2048.0)
    intrinsic, total_mass, _ = random_rows(4, seed=10)
    amp, phase = default_model.predict_modes_amp_phase(
        intrinsic, total_mass, frequencies[band]
    )
    h_plus, h_cross = mode_polarizations(amp, phase, default_model.modes, 1.0)
    mismatches = []
    for i in range(len(intrinsic)):
        params = ParametersWithExtrinsic(*intrinsic[i], 1.0, 1.0, total_mass[i])
        true = default_model.get_teob_modes_dict(frequencies[band], params)
        predicted = {
            mode: h_plus[i, j] - 1j * h_cross[i, j] for j, mode in enumerate(ALL_MODES)
        }
        mismatches.append(
            validator.full_waveform_mismatch(
                true, predicted, frequencies=frequencies[band]
            )
        )
    # same bound as `test_default_model_full_waveform_mismatch`
    assert np.max(mismatches) < 4e-3


# ---------------------------------------------------------------------- #
# JAX
# ---------------------------------------------------------------------- #


def test_jax_matches_numpy_given_the_regressors(default_model):
    pytest.importorskip("jax")
    intrinsic, total_mass, distance = random_rows(8, seed=11)
    numpy_surrogate = default_model.batched_surrogate(FOUR_MODES)
    jax_surrogate = with_regressor_outputs_of(
        BatchedSurrogate(default_model, FOUR_MODES, backend="jax"),
        numpy_surrogate,
        intrinsic,
    )
    expected = numpy_surrogate(
        intrinsic, total_mass, FREQUENCIES, distance, return_tf=True
    )
    got = [
        np.asarray(a)
        for a in jax_surrogate(
            intrinsic, total_mass, FREQUENCIES, distance, return_tf=True
        )
    ]
    np.testing.assert_allclose(got[0], expected[0], rtol=1e-11, atol=0)
    np.testing.assert_allclose(got[1], expected[1], rtol=1e-13, atol=1e-7)
    np.testing.assert_allclose(got[2], expected[2], rtol=1e-8, atol=1e-8)


def test_jax_function_end_to_end(default_model):
    """The jitted public function, including a single waveform as N = 1;
    against numpy only to the regressors' rounding."""
    jax = pytest.importorskip("jax")
    predict = jax.jit(default_model.jax_modes_amp_phase(FOUR_MODES, return_tf=True))
    intrinsic, total_mass, distance = random_rows(4, seed=12)
    expected = default_model.predict_modes_amp_phase(
        intrinsic,
        total_mass,
        FREQUENCIES,
        modes=FOUR_MODES,
        distance_mpc=distance,
        return_tf=True,
    )
    for rows in [slice(0, 4), slice(0, 1)]:
        amp, phase, tf = (
            np.asarray(a)
            for a in predict(
                intrinsic[rows], total_mass[rows], FREQUENCIES, distance[rows]
            )
        )
        support = expected[0][rows] > 0
        assert np.array_equal(amp > 0, support)
        assert np.max(np.abs(phase - expected[1][rows])[support]) < 0.1
        assert np.max(np.abs(amp[support] / expected[0][rows][support] - 1)) < 1e-2


def test_model_to_jax_waveform(default_model):
    jax = pytest.importorskip("jax")
    from mlgw_bns.jax_predict import model_to_jax_waveform

    predict = jax.jit(model_to_jax_waveform(default_model))
    params = ParametersWithExtrinsic(
        1.3, 400.0, 600.0, 0.1, -0.05, 100.0, 0.6, 2.7, 0.3
    )
    h_plus, h_cross = predict(
        np.array([1.3, 400.0, 600.0, 0.1, -0.05]), FREQUENCIES, 2.7, 100.0, 0.6, 0.3
    )
    expected_plus, expected_cross = default_model.predict(FREQUENCIES, params)
    scale = np.max(np.abs(expected_plus))
    # to the regressors' rounding, see test_jax_function_end_to_end
    assert np.max(np.abs(np.asarray(h_plus) - expected_plus)) < 2e-2 * scale
    assert np.max(np.abs(np.asarray(h_cross) - expected_cross)) < 2e-2 * scale


def test_spin_weighted_spherical_harmonic():
    from mlgw_bns.batched import spin_weighted_spherical_harmonic
    from mlgw_bns.special_func import spinsphericalharm

    for ell, emm in [(2, 2), (2, -1), (3, 3), (4, -4), (4, 3)]:
        for iota, phi in [(0.3, 0.0), (1.9, 0.7)]:
            expected = complex(*spinsphericalharm(-2, ell, emm, phi, iota))
            got = spin_weighted_spherical_harmonic(ell, emm, iota, phi)
            assert got == pytest.approx(expected, rel=1e-13, abs=1e-15)
