"""Tests for the precessing extension of the surrogate.

The twist is a rotation, so it has two properties which hold whatever
the waveform being twisted is, and which between them pin down all the
conventions involved: with no in-plane spin it must be the identity, and
in general it must conserve the power in each :math:`\\ell`.
"""

import numpy as np
import pytest

from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model import Model
from mlgw_bns.precessing_model import (
    EulerAngles,
    PrecessingModel,
    PrecessingParametersWithExtrinsic,
    check_aligned_spin_limit,
    newtonian_time_to_merger,
    twist_modes_frequency_domain,
)
from mlgw_bns.special_func import spinsphericalharm, wigner_d_function
from mlgw_bns.spherical_harmonics import Y_2_pos_2, spin_weighted_harmonic
from mlgw_bns.twist_waveform import twist_modes


@pytest.fixture(name="frequencies")
def fixture_frequencies():
    return np.linspace(20.0, 1024.0, 512)


@pytest.fixture(name="precessing_params")
def fixture_precessing_params():
    return PrecessingParametersWithExtrinsic(
        mass_ratio=1.3,
        lambda_1=200.0,
        lambda_2=500.0,
        chi_1=(0.1, -0.05, 0.15),
        chi_2=(-0.08, 0.06, -0.1),
        distance_mpc=100.0,
        inclination=0.9,
        total_mass=2.8,
        azimuth=0.4,
    )


def test_spherical_harmonic_aliases_agree_with_general_formula():
    """The named aliases must be the general formula, not a second implementation."""

    inclination, phase = 0.7, 0.3

    def as_complex(l, m):
        # spinsphericalharm returns the real and imaginary parts separately
        real, imaginary = spinsphericalharm(-2, l, m, phase, inclination)
        return real + 1j * imaginary

    np.testing.assert_allclose(Y_2_pos_2(inclination, phase), as_complex(2, 2))
    np.testing.assert_allclose(
        spin_weighted_harmonic(3, -3, inclination, phase), as_complex(3, -3)
    )


def test_wigner_d_function_is_vectorized():
    """A scalar angle gives a scalar, an array gives an array of the same shape."""

    angles = np.linspace(0.0, np.pi, 7)
    assert isinstance(wigner_d_function(2, 2, 2, 0.3), float)
    assert wigner_d_function(2, 2, 2, angles).shape == angles.shape
    assert wigner_d_function(2, 2, 2, angles)[3] == pytest.approx(
        wigner_d_function(2, 2, 2, angles[3])
    )


def test_twist_conserves_power():
    """Wigner-D matrices are unitary, so they preserve the power in each :math:`\\ell`."""

    times = np.linspace(0.0, 10.0, 128)
    amplitude = 1.0 + 0.1 * np.sin(times)
    coprecessing = {(2, 2): amplitude * np.exp(-1j * 3.0 * times)}

    alpha = 0.3 * times
    beta = 0.4 + 0.1 * np.cos(times)
    gamma = -0.2 * times

    # the m = 0 multipole is produced automatically for each ell
    positive_m, negative_m, zero_m = twist_modes(
        coprecessing, alpha, beta, gamma, lm_inertial=[(2, 2), (2, 1)]
    )

    power_out = sum(
        np.abs(mode) ** 2
        for twisted in (positive_m, negative_m, zero_m)
        for mode in twisted.values()
    )
    # the (2,2) co-precessing mode has a (2,-2) partner of equal magnitude
    np.testing.assert_allclose(power_out, 2.0 * amplitude**2, rtol=1e-10)


def test_zero_opening_angle_is_a_z_rotation():
    """With ``beta = 0`` the twist only shifts phases, by :math:`m(\\alpha - \\gamma)`."""

    times = np.linspace(0.0, 10.0, 64)
    phase = 3.0 * times
    coprecessing = {(2, 2): np.exp(1j * phase)}

    alpha = 0.3 * times
    gamma = -0.2 * times

    twisted, _, _ = twist_modes(
        coprecessing, alpha, np.zeros_like(times), gamma, lm_inertial=[(2, 2)]
    )

    # twist_modes works in the ``h = A exp(-i phi)`` convention, in which
    # the twist adds ``m (alpha - gamma)`` to phi; the input here is
    # ``exp(+i phase)``, so phi = -phase
    expected = np.exp(1j * (phase - 2.0 * (alpha - gamma)))
    np.testing.assert_allclose(twisted[(2, 2)], expected, atol=1e-12)


def test_newtonian_time_to_merger_scaling():
    """:math:`t_c \\propto \\eta^{-1} (M\\Omega)^{-8/3}`."""

    assert newtonian_time_to_merger(0.25, 2e-3) / newtonian_time_to_merger(
        0.25, 1e-3
    ) == pytest.approx(2.0 ** (-8.0 / 3.0))
    assert newtonian_time_to_merger(0.125, 1e-3) == pytest.approx(
        2.0 * newtonian_time_to_merger(0.25, 1e-3)
    )


def test_euler_angles_cover_the_whole_inspiral(precessing_params):
    """The PN integration must reach merger, not stop at its own cutoff."""

    model = Model.default_for_testing()
    angles = PrecessingModel(model).euler_angles(precessing_params, 20.0)

    assert angles.momega[0] < 1e-3
    assert angles.momega[-1] > 0.1
    assert np.all(np.diff(angles.momega) > 0)
    assert angles.beta.max() > 0.0


def test_aligned_spin_limit_of_the_twist(frequencies, precessing_params):
    """Without in-plane spin the precessing pipeline must be the aligned-spin one."""

    check_aligned_spin_limit(
        Model.default_for_testing(), frequencies, precessing_params
    )


def test_precessing_prediction_is_finite(frequencies, precessing_params):
    """A generic precessing configuration produces a usable waveform."""

    precessing = PrecessingModel(Model.default_for_testing())
    h_plus, h_cross = precessing.predict(frequencies, precessing_params)

    assert h_plus.shape == frequencies.shape
    assert np.all(np.isfinite(h_plus))
    assert np.all(np.isfinite(h_cross))
    assert np.max(np.abs(h_plus)) > 0.0
