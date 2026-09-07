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
    eob_orbital_frequency_rate,
    newtonian_time_to_merger,
    twist_modes_frequency_domain,
)
from mlgw_bns.special_func import spinsphericalharm, wigner_d_function
from mlgw_bns.spherical_harmonics import Y_2_pos_2, spin_weighted_harmonic
from mlgw_bns.twist_waveform import integrate_pn_spin_precession, twist_modes


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


def test_orbital_frequency_integration_covers_the_inspiral(precessing_params):
    """Marching the precession against Momega must reach merger with a
    strictly increasing frequency grid, a real opening angle, and a time
    axis reconstructed by quadrature."""

    kwargs = dict(
        nu=precessing_params.eta,
        chi1vec=precessing_params.chi_1_vector,
        chi2vec=precessing_params.chi_2_vector,
        f0=2.0 * 20.0 * precessing_params.total_mass * 4.925e-6 / 4.0,
    )
    in_freq = integrate_pn_spin_precession(
        **kwargs, independent_variable="orbital_frequency"
    )

    assert np.all(np.diff(in_freq["Momega"]) > 0)
    assert in_freq["Momega"][0] < 1e-3
    assert in_freq["Momega"][-1] > 0.1
    assert in_freq["beta"].max() > 0.0
    assert in_freq["t"] is not None
    assert np.all(np.diff(in_freq["t"]) > 0)


def test_eob_orbital_frequency_rate_is_positive_and_climbing(precessing_params):
    """The rate built from a (2,2) phase must be positive over the band
    and climb with frequency at close to the Newtonian slope of 11/3."""

    model = Model.default_for_testing()
    reference_frequencies = np.geomspace(20.0, 2000.0, 400)
    amplitude, phase = model.predict_amplitude_phase_mode(
        Mode(2, 2), reference_frequencies, precessing_params.aligned()
    )
    mass_sum_seconds = precessing_params.aligned().mass_sum_seconds
    rate = eob_orbital_frequency_rate(
        reference_frequencies, amplitude * np.exp(1j * phase), mass_sum_seconds
    )

    momega = np.pi * mass_sum_seconds * reference_frequencies
    values = np.array([rate(m) for m in momega])
    assert np.all(values > 0)
    assert values[-1] > values[0]
    slope = np.polyfit(np.log(momega), np.log(values), 1)[0]
    assert 2.5 < slope < 4.5  # Newtonian energy balance gives 11/3


def test_anchoring_to_the_reference_phase_moves_only_precessing_waveforms(
    frequencies, precessing_params
):
    """Integrating the angles along the surrogate's orbital-frequency
    track changes a precessing waveform but leaves the aligned limit
    exactly where it was."""

    precessing = PrecessingModel(Model.default_for_testing())

    plain = precessing.euler_angles(precessing_params, float(frequencies[0]))
    anchored = precessing.euler_angles(
        precessing_params, float(frequencies[0]), anchor_to_reference_phase=True
    )
    hp_plain, _ = precessing.predict(
        frequencies, precessing_params, angles=plain, reanchor=False
    )
    hp_anchored, _ = precessing.predict(
        frequencies, precessing_params, angles=anchored, reanchor=False
    )
    scale = np.max(np.abs(hp_plain))
    assert np.max(np.abs(hp_anchored - hp_plain)) > 1e-3 * scale

    aligned = PrecessingParametersWithExtrinsic(
        mass_ratio=precessing_params.mass_ratio,
        lambda_1=precessing_params.lambda_1,
        lambda_2=precessing_params.lambda_2,
        chi_1=(0.0, 0.0, precessing_params.chi_1_vector[2]),
        chi_2=(0.0, 0.0, precessing_params.chi_2_vector[2]),
        distance_mpc=precessing_params.distance_mpc,
        inclination=precessing_params.inclination,
        total_mass=precessing_params.total_mass,
    )
    with_anchor = precessing.predict(
        frequencies, aligned,
        angles=precessing.euler_angles(
            aligned, float(frequencies[0]), anchor_to_reference_phase=True
        ),
        reanchor=False,
    )
    without = precessing.predict(frequencies, aligned, reanchor=False)
    np.testing.assert_allclose(with_anchor[0], without[0], rtol=1e-8, atol=0.0)


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


def test_reanchored_without_time_is_a_no_op():
    """An ``EulerAngles`` carrying no integration time is returned unchanged."""

    angles = EulerAngles(
        momega=np.linspace(1e-3, 0.1, 50),
        alpha=np.linspace(0.0, 3.0, 50),
        beta=np.full(50, 0.2),
        gamma=np.linspace(0.0, -2.0, 50),
    )
    frequencies = np.linspace(20.0, 1024.0, 128)
    reference = np.exp(-1j * frequencies**2)  # any chirp-like phase

    assert angles.reanchored(frequencies, reference, 1.0e-5, 20.0) is angles


def test_reanchoring_follows_the_reference_phase(frequencies, precessing_params):
    r"""The re-anchored ``momega`` axis tracks the (2,2) phase's own
    stationary-phase time-frequency map, not the PN one."""

    model = Model.default_for_testing()
    precessing = PrecessingModel(model)
    angles = precessing.euler_angles(precessing_params, float(frequencies[0]))

    coprecessing = model.coprecessing_modes_dict(
        frequencies, precessing_params.aligned()
    )
    mass_sum_seconds = precessing_params.aligned().mass_sum_seconds
    reanchored = angles.reanchored(
        frequencies, coprecessing[(2, 2)], mass_sum_seconds, float(frequencies[0])
    )

    assert reanchored is not angles
    assert np.all(np.isfinite(reanchored.momega))
    assert np.all(np.diff(reanchored.momega) >= 0.0)
    # the angle values are carried over untouched
    np.testing.assert_array_equal(reanchored.alpha, angles.alpha)
    np.testing.assert_array_equal(reanchored.beta, angles.beta)

    # the (2,2) angles, looked up at pi M f on the re-anchored axis, must
    # match the ones the surrogate (2,2) phase implies: at the frequency
    # where the re-anchored beta peaks, the plain-PN axis puts a
    # different beta
    target = np.pi * mass_sum_seconds * frequencies
    _, beta_anchored, _ = reanchored.at_momega(target)
    _, beta_plain, _ = angles.at_momega(target)
    assert np.max(np.abs(beta_anchored - beta_plain)) > 1e-4


def test_reanchoring_changes_a_precessing_waveform_but_not_an_aligned_one(
    frequencies, precessing_params
):
    """Re-anchoring moves a precessing waveform and leaves the aligned limit alone."""

    precessing = PrecessingModel(Model.default_for_testing())

    anchored = precessing.predict(frequencies, precessing_params, reanchor=True)
    plain = precessing.predict(frequencies, precessing_params, reanchor=False)
    scale = np.max(np.abs(plain[0]))
    assert np.max(np.abs(anchored[0] - plain[0])) > 1e-3 * scale

    aligned = PrecessingParametersWithExtrinsic(
        mass_ratio=precessing_params.mass_ratio,
        lambda_1=precessing_params.lambda_1,
        lambda_2=precessing_params.lambda_2,
        chi_1=(0.0, 0.0, precessing_params.chi_1_vector[2]),
        chi_2=(0.0, 0.0, precessing_params.chi_2_vector[2]),
        distance_mpc=precessing_params.distance_mpc,
        inclination=precessing_params.inclination,
        total_mass=precessing_params.total_mass,
    )
    with_reanchor = precessing.predict(frequencies, aligned, reanchor=True)
    without = precessing.predict(frequencies, aligned, reanchor=False)
    np.testing.assert_allclose(with_reanchor[0], without[0], rtol=1e-10, atol=0.0)
