r"""Precessing binaries from the LVK spin angles.

Parameter estimation codes (LALInference, bilby, ...) describe the
orientation of a precessing binary with the angles of LALSuite's
``SimInspiralTransformPrecessingNewInitialConditions``, all at the
reference frequency :math:`f_{\rm ref}`:

* ``theta_jn``, the angle between the total angular momentum
  :math:`\vec{J}` and the line of sight :math:`\hat{N}`;
* ``phi_jl``, the azimuth of :math:`\hat{L}` on its cone about
  :math:`\vec{J}`, measured from the plane of :math:`\vec{J}` and
  :math:`\hat{N}`;
* ``tilt_1``, ``tilt_2``, the angles between the spins and :math:`\hat{L}`,
  ``phi_12`` the azimuth of the second spin about :math:`\hat{L}` measured
  from the first, and ``a_1``, ``a_2`` their magnitudes;
* ``phase``, the orbital phase.

:func:`lvk_to_precessing` is a port of that function, vectorised and
generic over ``numpy`` and ``jax.numpy``: it gives the spins in the frame
of :class:`~mlgw_bns.precessing_model.PrecessingParametersWithExtrinsic`,
:math:`\hat{z} = \hat{L}_0`, and the line of sight in it. As in LALSuite,
:math:`\vec{J} = \vec{L} + \vec{S}_1 + \vec{S}_2` with the 1PN orbital
angular momentum.

The orbital phase enters only as ``reference_phase``: the spins are those
LALSuite gives for ``phiRef = 0``, with the line of sight at azimuth
:math:`\pi / 2`, and the orbital separation is rotated by ``phase`` with
respect to them, which is the same binary as LALSuite's (the spins and the
line of sight rotated by ``-phase`` about :math:`\hat{L}_0`, the
separation along :math:`\hat{x}`). So ``phase`` only rotates the
co-precessing multipole :math:`m` by :math:`e^{i m \, {\rm phase}}`, and the
precession angles do not depend on it.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .taylorf2 import SUN_MASS_SECONDS


class PrecessingOrientation(NamedTuple):
    """The output of :func:`lvk_to_precessing`; arrays of the input shape,
    the spins with a trailing axis of length 3."""

    inclination: object
    azimuth: object
    reference_phase: object
    chi_1: object
    chi_2: object


def _rotate_z(angle, x, y, z, xp):
    """LALSimInspiral's ``ROTATEZ``: active rotation by ``angle`` about z."""
    cos, sin = xp.cos(angle), xp.sin(angle)
    return x * cos - y * sin, x * sin + y * cos, z


def _rotate_y(angle, x, y, z, xp):
    """LALSimInspiral's ``ROTATEY``: active rotation by ``angle`` about y."""
    cos, sin = xp.cos(angle), xp.sin(angle)
    return x * cos + z * sin, y, -x * sin + z * cos


def orbital_angular_momentum(total_mass, eta, reference_frequency_hz, xp=np):
    r""":math:`|\vec{L}|` at 1PN, in units of the solar mass squared, as in
    ``SimInspiralTransformPrecessingNewInitialConditions``:
    :math:`\eta M^2 / v \, (1 + v^2 (3/2 + \eta / 6))`, :math:`v = (\pi M
    f_{\rm ref})^{1/3}`."""
    v = xp.cbrt(np.pi * total_mass * SUN_MASS_SECONDS * reference_frequency_hz)
    return eta * total_mass**2 / v * (1.0 + v**2 * (1.5 + eta / 6.0))


def lal_precessing_spins(
    theta_jn,
    phi_jl,
    tilt_1,
    tilt_2,
    phi_12,
    a_1,
    a_2,
    mass_1,
    mass_2,
    reference_frequency_hz,
    phase,
    xp=np,
):
    r"""``SimInspiralTransformPrecessingNewInitialConditions``, vectorised.

    Masses in solar masses, the heavier first. Returns ``(inclination,
    s1x, s1y, s1z, s2x, s2y, s2z)``: the angle between :math:`\hat{L}_0` and
    :math:`\hat{N}`, and the dimensionless spins in LALSuite's frame,
    :math:`\hat{z} = \hat{L}_0`, :math:`\hat{x}` along the orbital
    separation, and the line of sight at azimuth :math:`\pi / 2 - {\rm
    phase}`.
    """
    shape = xp.broadcast_shapes(*(xp.shape(value) for value in (
        theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2,
        reference_frequency_hz, phase)))
    theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2, f_ref, phase = (
        xp.broadcast_to(xp.asarray(value, dtype=float), shape) for value in (
            theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2,
            reference_frequency_hz, phase))
    zero, one = xp.zeros(shape), xp.ones(shape)

    # starting frame: L along z, s1 in the x-z plane rotated by phase, s2 at
    # azimuth phi_12 from it
    l_hat = (zero, zero, one)
    s1_hat = (xp.sin(tilt_1) * xp.cos(phase), xp.sin(tilt_1) * xp.sin(phase), xp.cos(tilt_1))
    s2_hat = (
        xp.sin(tilt_2) * xp.cos(phi_12 + phase),
        xp.sin(tilt_2) * xp.sin(phi_12 + phase),
        xp.cos(tilt_2),
    )

    total_mass = mass_1 + mass_2
    eta = mass_1 * mass_2 / total_mass**2
    l_mag = orbital_angular_momentum(total_mass, eta, f_ref, xp=xp)
    s1 = [mass_1**2 * a_1 * c for c in s1_hat]
    s2 = [mass_2**2 * a_2 * c for c in s2_hat]
    j = (s1[0] + s2[0], s1[1] + s2[1], l_mag + s1[2] + s2[2])
    j_norm = xp.sqrt(j[0] ** 2 + j[1] ** 2 + j[2] ** 2)
    theta_0 = xp.arccos(j[2] / j_norm)
    phi_0 = xp.arctan2(j[1], j[0])

    # J along z
    s1_hat = _rotate_z(-phi_0, *s1_hat, xp)
    s2_hat = _rotate_z(-phi_0, *s2_hat, xp)
    l_hat = _rotate_y(-theta_0, *l_hat, xp)
    s1_hat = _rotate_y(-theta_0, *s1_hat, xp)
    s2_hat = _rotate_y(-theta_0, *s2_hat, xp)

    # L at azimuth phi_jl about J (it is at azimuth pi now)
    l_hat = _rotate_z(phi_jl - np.pi, *l_hat, xp)
    s1_hat = _rotate_z(phi_jl - np.pi, *s1_hat, xp)
    s2_hat = _rotate_z(phi_jl - np.pi, *s2_hat, xp)

    # the line of sight in the y-z plane, at theta_jn from J
    n_hat = (zero, xp.sin(theta_jn), xp.cos(theta_jn))
    inclination = xp.arccos(xp.clip(
        n_hat[0] * l_hat[0] + n_hat[1] * l_hat[1] + n_hat[2] * l_hat[2], -1.0, 1.0))

    # L back along z
    theta_lj = xp.arccos(xp.clip(l_hat[2], -1.0, 1.0))
    phi_l = xp.arctan2(l_hat[1], l_hat[0])
    s1_hat = _rotate_z(-phi_l, *s1_hat, xp)
    s2_hat = _rotate_z(-phi_l, *s2_hat, xp)
    n_hat = _rotate_z(-phi_l, *n_hat, xp)
    s1_hat = _rotate_y(-theta_lj, *s1_hat, xp)
    s2_hat = _rotate_y(-theta_lj, *s2_hat, xp)
    n_hat = _rotate_y(-theta_lj, *n_hat, xp)

    # the line of sight at azimuth pi/2 - phase
    phi_n = xp.arctan2(n_hat[1], n_hat[0])
    s1_hat = _rotate_z(np.pi / 2.0 - phi_n - phase, *s1_hat, xp)
    s2_hat = _rotate_z(np.pi / 2.0 - phi_n - phase, *s2_hat, xp)

    return (
        inclination,
        *(a_1 * c for c in s1_hat),
        *(a_2 * c for c in s2_hat),
    )


def lvk_to_precessing(
    theta_jn,
    phi_jl,
    tilt_1,
    tilt_2,
    phi_12,
    a_1,
    a_2,
    mass_1,
    mass_2,
    reference_frequency_hz,
    phase,
    xp=np,
) -> PrecessingOrientation:
    r"""The orientation of a precessing binary given its LVK spin angles.

    Masses in solar masses, the heavier first (``a_1``, ``tilt_1`` are its
    spin's); all angles in radians, at ``reference_frequency_hz``. See the
    module docstring.

    Returns
    -------
    PrecessingOrientation
        ``inclination``, ``azimuth``, ``reference_phase``, ``chi_1`` and
        ``chi_2`` of
        :class:`~mlgw_bns.precessing_model.PrecessingParametersWithExtrinsic`.
    """
    inclination, *spins = lal_precessing_spins(
        theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2,
        reference_frequency_hz, xp.zeros_like(xp.asarray(phase, dtype=float)), xp=xp,
    )
    phase = xp.asarray(phase, dtype=float)
    return PrecessingOrientation(
        inclination=inclination,
        azimuth=xp.full_like(inclination, np.pi / 2.0),
        reference_phase=xp.broadcast_to(phase, xp.shape(inclination)),
        chi_1=xp.stack(spins[:3], axis=-1),
        chi_2=xp.stack(spins[3:], axis=-1),
    )
