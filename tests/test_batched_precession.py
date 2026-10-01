"""Tests for the batched JAX precessing waveform (:mod:`mlgw_bns.batched_precession`)."""

import numpy as np
import pytest

from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model_validation import ValidateModel
from mlgw_bns.precessing_model import (
    PrecessingModel,
    PrecessingParametersWithExtrinsic,
    stationary_phase_transform,
    stationary_phase_window,
)

# From 20 Hz the numpy angles integrate from 10 Hz, a few seconds a binary.
FREQUENCIES = np.geomspace(20.0, 2048.0, 1500)

BINARIES = [
    # reference frequency at the bottom of the band: a short backward leg
    PrecessingParametersWithExtrinsic(
        mass_ratio=1.4, lambda_1=300.0, lambda_2=900.0,
        chi_1=(0.3, -0.1, 0.1), chi_2=(-0.1, 0.25, -0.2),
        distance_mpc=80.0, inclination=1.1, total_mass=2.7, azimuth=0.4,
        reference_phase=0.9, merger_time=0.02, reference_frequency_hz=20.0,
    ),
    # inside the band: the backward leg crosses the band, odd modes and all
    PrecessingParametersWithExtrinsic(
        mass_ratio=2.3, lambda_1=50.0, lambda_2=2000.0,
        chi_1=(-0.35, 0.2, -0.15), chi_2=(0.1, 0.1, 0.3),
        distance_mpc=150.0, inclination=2.4, total_mass=3.1, azimuth=5.0,
        reference_phase=4.0, merger_time=-0.05, reference_frequency_hz=50.0,
    ),
    # aligned spins
    PrecessingParametersWithExtrinsic(
        mass_ratio=1.1, lambda_1=1000.0, lambda_2=1500.0,
        chi_1=(0.0, 0.0, 0.2), chi_2=(0.0, 0.0, -0.1),
        distance_mpc=40.0, inclination=0.3, total_mass=2.5, azimuth=1.0,
        reference_phase=2.0, merger_time=0.0, reference_frequency_hz=30.0,
    ),
]


def test_jax_precessing_waveform_matches_numpy(default_model):
    """One batch with all the binaries, against PrecessingModel.predict;
    the precession angles are integrated differently (see the module
    docstring), to ~3e-5 rad."""
    jax = pytest.importorskip("jax")
    from mlgw_bns.batched_precession import batch_arguments

    precessing = PrecessingModel(default_model)
    predict = jax.jit(precessing.jax_predict())
    h_plus, h_cross = predict(*batch_arguments(BINARIES, FREQUENCIES))
    validator = ValidateModel(default_model.mode_models[Mode(2, 2)])
    for i, params in enumerate(BINARIES):
        expected = precessing.predict(FREQUENCIES, params)
        for got, want in zip((h_plus[i], h_cross[i]), expected):
            got = np.asarray(got)
            assert np.all(np.isfinite(got))
            mismatch = validator.full_waveform_mismatch(
                {(2, 2): want}, {(2, 2): got}, frequencies=FREQUENCIES
            )
            assert mismatch < 1e-8, (i, mismatch)
            # nothing optimised: the same time, phase and amplitude
            assert np.linalg.norm(got - want) < 1e-3 * np.linalg.norm(want), i


def test_twist_is_continuous_through_the_reference_frequency():
    """Below the reference frequency the PN integration runs backwards, and
    alpha = atan2(L_y, L_x) flips by pi as L passes through z there: gamma
    must follow, or every odd-m co-precessing multipole changes sign."""
    from mlgw_bns.precessing_model import euler_angles, twist_modes_frequency_domain

    params = BINARIES[0]
    mass_seconds = params.aligned().mass_sum_seconds
    angles = euler_angles(params, 9.0, largest_mode_m=2)
    for ell, n in [(2, 1), (2, 2), (3, 3)]:
        # the (ell, n) multipole reads the angles at the reference here
        crossing = n / 2.0 * params.reference_frequency_hz
        frequencies = np.linspace(0.98, 1.02, 401) * crossing
        unit = {(ell, n): np.ones_like(frequencies, dtype=complex)}
        positive, _ = twist_modes_frequency_domain(
            unit, frequencies, angles, mass_seconds
        )
        twisted = positive[(ell, n)]
        assert np.max(np.abs(np.diff(twisted))) < 1e-2, (ell, n)


def test_stationary_phase_transform_matches_the_analytic_one(default_model):
    """The window fit agrees with X = phi - f phi', phi' from the batched
    surrogate's analytic derivative, which the JAX path uses."""
    row = np.array([[1.7, 400.0, 1200.0, 0.2, -0.3]])
    for key in [(2, 2), (2, 1), (3, 3)]:
        for frequency in [4.75, 9.5, 30.0]:
            window = stationary_phase_window(frequency)
            _, phase = default_model.predict_modes_amp_phase(
                row, 2.8, window, modes=[key]
            )
            _, at, time = default_model.predict_modes_amp_phase(
                row, 2.8, np.array([frequency]), modes=[key], return_tf=True
            )
            fitted = stationary_phase_transform(window, phase[0, 0], frequency)
            analytic = at[0, 0, 0] + 2 * np.pi * frequency * time[0, 0, 0]
            # the phase is ~4e5 rad at 4.75 Hz
            assert abs(fitted - analytic) < 1e-5, (key, frequency)
