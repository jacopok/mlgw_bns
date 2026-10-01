r"""Batched precessing waveforms, in JAX.

:meth:`PrecessingModel.predict <mlgw_bns.precessing_model.PrecessingModel.predict>`
for ``N`` binaries in one call, as a pure JAX function: it can be wrapped
in :func:`jax.jit`, ``vmap``-ed and differentiated.
:meth:`PrecessingModel.jax_predict
<mlgw_bns.precessing_model.PrecessingModel.jax_predict>` builds one.
Every stage is the numpy one, run on :mod:`jax.numpy`:

* the co-precessing multipoles come from the batched surrogate,
  :meth:`Model.jax_modes_amp_phase <mlgw_bns.model.Model.jax_modes_amp_phase>`;
* their orbital phase is set at the reference frequency with
  :func:`~mlgw_bns.precessing_model.orbital_phase_from_transforms`, from the
  stationary-phase transform :math:`X = \Psi - f \Psi'` of the surrogate's
  own phase and its analytic derivative (``return_tf``) instead of the cubic
  window fit of :func:`~mlgw_bns.precessing_model.stationary_phase_transform`,
  which it matches to ~1e-6 rad;
* the twist and the projection are
  :func:`~mlgw_bns.precessing_model.twist_coefficients`, which applies
  :func:`~mlgw_bns.precessing_model.twist_modes_frequency_domain` and
  :func:`~mlgw_bns.precessing_model.polarizations_from_inertial_modes` to a
  unit multipole: the polarizations are linear in the co-precessing
  multipoles, and :func:`precessing_mode_components` returns them and their
  coefficients separately, as mode-by-mode relative binning needs.

The one stage that is not shared is the integration of the PN
spin-precession equations (the same right-hand side,
:func:`mlgw_bns.twist_waveform._pn_precession_derivatives`). The numpy
path integrates them in time with SciPy's adaptive DOP853, 3--12 s per
binary; here they are integrated with DOP853's own coefficients on a fixed
grid (:func:`integrate_angles`), against the orbital frequency, which is
the same trajectory: :math:`\mathrm{d}y/\mathrm{d}\Omega = (\mathrm{d}y /
\mathrm{d}t) / \dot\Omega`, with the full 3.5PN + tidal :math:`\dot\Omega`
(positive throughout the BNS parameter space). The grid is uniform in

.. math::
    x = \ln \Omega - \kappa \, \Omega_{\rm lo} / \Omega ,

which is uniform in :math:`1/\Omega` at low frequency, where
:math:`\hat{L}` winds around :math:`\vec{J}` hundreds of times (the
precession phase grows as :math:`\Omega^{-1}`), and in
:math:`\ln \Omega` near the merger; :data:`KAPPA` sets the balance. The
state between the nodes is a cubic Hermite interpolant with the exact
derivatives of the right-hand side, and the Euler angles are computed from
the interpolated :math:`\hat{L}` (:class:`TabulatedAngles`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import numpy as np
from scipy.integrate._ivp import dop853_coefficients as dop853

from .precessing_model import (
    orbital_phase_from_transforms,
    reference_phase_keys,
    twist_coefficients,
)
from .taylorf2 import SUN_MASS_SECONDS
from .twist_waveform import (
    _pn_precession_derivatives,
    alpha_initial_condition,
    eob_mrg_momg,
    nu_to_X1,
    tidal_flux_coefficient,
)

if TYPE_CHECKING:
    from .model import Model
    from .precessing_model import PrecessingParametersWithExtrinsic

#: Steps of each of the two legs of the integration, from the reference
#: frequency down to the start and up to the merger. With 1024 the angles
#: differ from the numpy ones by up to ~3e-4 rad; with 2048 and 4096 by
#: ~3e-5 rad, the numpy path's own interpolation error.
N_STEPS = 2048

#: Weight of the :math:`1/\Omega` part of the integration variable, against
#: the :math:`\ln \Omega` one; see the module docstring.
KAPPA = 16.0

# DOP853's Butcher tableau, the twelve stages of its eighth-order solution
_A = np.asarray(dop853.A[: dop853.N_STAGES, : dop853.N_STAGES])
_B = np.asarray(dop853.B)


def _jnp():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    return jax, jnp


@dataclass
class _Binary:
    """The constants of one binary's precession equations."""

    nu: Any
    q: Any
    a10_tidal: Any
    kappa_omega: Any  # kappa * Omega_lo


def _rhs(xp, binary: _Binary, y):
    r""":math:`\mathrm{d}y / \mathrm{d}x` for the state ``y = [S_A, S_B,
    \hat{L}, \alpha - \gamma, \Omega]`` of one binary.

    With :math:`\dot\gamma = \dot\alpha \cos\beta`, :math:`\alpha -
    \gamma` grows as :math:`(L_x \dot{L}_y - L_y \dot{L}_x) / (1 + L_z)`,
    which is regular where :math:`\hat{L}` passes through :math:`\hat{z}`
    --- at the reference point, in particular --- unlike :math:`\dot\gamma`
    alone (which the time-domain integration sets to zero exactly there,
    harmlessly only for an adaptive step).
    """
    omega = y[10]
    d_sa, d_sb, d_l, _, d_omega, _ = _pn_precession_derivatives(
        binary.nu, binary.q, omega, y[0:3], y[3:6], y[6:9], binary.a10_tidal, xp=xp
    )
    l_x, l_y, l_z = y[6], y[7], y[8]
    d_alpha_minus_gamma = (l_x * d_l[1] - l_y * d_l[0]) / (1.0 + l_z)
    # dOmega/dx, from dx/dOmega = 1/Omega + kappa Omega_lo / Omega^2
    omega_rate = omega**2 / (omega + binary.kappa_omega)
    scale = omega_rate / d_omega
    return xp.concatenate([
        d_sa * scale, d_sb * scale, d_l * scale,
        xp.stack([d_alpha_minus_gamma * scale, omega_rate]),
    ])


def _leg(binary: _Binary, y0, step, n_steps: int):
    """``n_steps`` fixed DOP853 steps of ``step`` in ``x`` from ``y0``: the
    states and their derivatives at the ``n_steps + 1`` nodes."""
    jax, jnp = _jnp()

    def advance(y, _):
        k = [_rhs(jnp, binary, y)]
        for i in range(1, len(_B)):
            k.append(
                _rhs(jnp, binary, y + step * sum(_A[i, j] * k[j] for j in range(i) if _A[i, j]))
            )
        return y + step * sum(b * k_i for b, k_i in zip(_B, k) if b), (y, k[0])

    last, (states, derivatives) = jax.lax.scan(advance, y0, None, length=n_steps)
    return (
        jnp.concatenate([states, last[None]]),
        jnp.concatenate([derivatives, _rhs(jnp, binary, last)[None]]),
    )


@dataclass
class TabulatedAngles:
    r"""The Euler angles of one binary, from :func:`integrate_angles`.

    The JAX counterpart of :class:`~mlgw_bns.precessing_model.EulerAngles`,
    with the same :meth:`at_momega`. The state ``[\hat{L}, \alpha -
    \gamma]`` is tabulated at ``2 n + 1`` nodes uniform in :math:`x` on
    either side of the reference, ``x[n]``, with its derivatives.
    """

    x: Any  # (2n + 1,)
    state: Any  # (2n + 1, 4): L_x, L_y, L_z, alpha - gamma
    derivative: Any  # (2n + 1, 4), d state / dx
    kappa_omega: Any
    omega_lo: Any
    omega_hi: Any
    #: alpha just above the reference, where L = z and atan2 is undefined
    alpha_reference: Any

    def _x(self, omega):
        _, jnp = _jnp()
        return jnp.log(omega) - self.kappa_omega / omega

    def at_momega(self, momega):
        r""":math:`(\alpha, \beta, \gamma)` at the orbital frequencies
        ``momega``, held at the end values outside the integrated range, as
        :meth:`EulerAngles.at_momega
        <mlgw_bns.precessing_model.EulerAngles.at_momega>`."""
        _, jnp = _jnp()
        omega = jnp.clip(momega, self.omega_lo, self.omega_hi)
        x = self._x(omega)
        n = (self.x.shape[0] - 1) // 2
        x_lo, x_reference, x_hi = self.x[0], self.x[n], self.x[-1]
        # each leg is uniform in x; the backward one may have zero length
        width_back = jnp.where(x_reference > x_lo, (x_reference - x_lo) / n, 1.0)
        width_forward = (x_hi - x_reference) / n
        index = jnp.where(
            x >= x_reference,
            n + jnp.floor((x - x_reference) / width_forward),
            jnp.floor((x - x_lo) / width_back),
        )
        index = jnp.clip(index, 0, 2 * n - 1).astype(int)
        width = jnp.where(x >= x_reference, width_forward, width_back)[..., None]
        t = ((x - self.x[index]) / width[..., 0])[..., None]
        # cubic Hermite basis
        t2, t3 = t * t, t * t * t
        state = (
            (2 * t3 - 3 * t2 + 1) * self.state[index]
            + (t3 - 2 * t2 + t) * width * self.derivative[index]
            + (-2 * t3 + 3 * t2) * self.state[index + 1]
            + (t3 - t2) * width * self.derivative[index + 1]
        )
        l_x, l_y, l_z, alpha_minus_gamma = (state[..., i] for i in range(4))
        in_plane = jnp.hypot(l_x, l_y)
        alpha = jnp.where(
            in_plane > 0, jnp.arctan2(l_y, l_x), self.alpha_reference
        )
        beta = jnp.arctan2(in_plane, l_z)
        # below the reference atan2 flips alpha by pi, and gamma with it:
        # the continuation of backward_gamma
        return alpha, beta, alpha - alpha_minus_gamma


def integrate_angles(
    mass_ratio,
    lambda_1,
    lambda_2,
    chi_1,
    chi_2,
    reference_frequency_22,
    initial_frequency_22,
    n_steps: int = N_STEPS,
) -> TabulatedAngles:
    r"""The PN spin-precession dynamics of one binary, on a fixed grid.

    What :func:`~mlgw_bns.precessing_model.euler_angles` integrates (the
    spins given at ``reference_frequency_22``, integrated down to
    ``initial_frequency_22`` and up to TEOBResumS' stopping frequency
    :math:`1.1\,\Omega_{\rm mrg}`), in JAX; see the module docstring.

    Parameters
    ----------
    mass_ratio, lambda_1, lambda_2 : scalar
        :math:`q \geq 1` and the tidal polarizabilities.
    chi_1, chi_2 : array of shape (3,)
        Dimensionless spin vectors at the reference frequency.
    reference_frequency_22, initial_frequency_22 : scalar
        :math:`(2, 2)` GW frequencies in geometric units (:math:`M f`): where
        the spins are given, and the lowest needed.
    n_steps : int
        Steps of each of the two legs.

    Returns
    -------
    TabulatedAngles
    """
    _, jnp = _jnp()
    nu = mass_ratio / (1.0 + mass_ratio) ** 2
    mass_a = nu_to_X1(nu, jnp)
    mass_b = 1.0 - mass_a
    omega_reference = np.pi * reference_frequency_22
    omega_lo = jnp.minimum(np.pi * initial_frequency_22, omega_reference)
    omega_hi = 1.1 * eob_mrg_momg(nu, mass_a, mass_b, chi_1[2], chi_2[2], xp=jnp)
    binary = _Binary(
        nu=nu,
        q=mass_a / mass_b,
        a10_tidal=tidal_flux_coefficient(nu, lambda_1, lambda_2, xp=jnp),
        kappa_omega=KAPPA * omega_lo,
    )

    def x_of(omega):
        return jnp.log(omega) - binary.kappa_omega / omega

    gamma_0 = alpha_initial_condition(
        mass_a / mass_b, *(chi_1[i] for i in range(3)), *(chi_2[i] for i in range(3)),
        reference_frequency_22, xp=jnp,
    )
    y0 = jnp.concatenate([
        chi_1 * mass_a**2,
        chi_2 * mass_b**2,
        jnp.array([0.0, 0.0, 1.0]),
        jnp.stack([0.0 * gamma_0, omega_reference]),
    ])
    # alpha where L = z, the limit from above (as the numpy path, which takes
    # the next sample's)
    d_l = _rhs(jnp, binary, y0)[6:8]
    alpha_reference = jnp.where(
        jnp.hypot(*d_l) > 0, jnp.arctan2(d_l[1], d_l[0]), 0.0
    )
    y0 = y0.at[9].set(alpha_reference - gamma_0)
    x_reference = x_of(omega_reference)
    back_step = (x_of(omega_lo) - x_reference) / n_steps
    forward_step = (x_of(omega_hi) - x_reference) / n_steps
    # the two legs side by side, in one scan
    jax, _ = _jnp()
    (back, forward), (back_derivative, forward_derivative) = jax.vmap(
        lambda step: _leg(binary, y0, step, n_steps)
    )(jnp.stack([back_step, forward_step]))

    steps = jnp.arange(n_steps + 1)
    x = jnp.concatenate([
        (x_reference + back_step * steps)[:0:-1], x_reference + forward_step * steps
    ])
    state = jnp.concatenate([back[:0:-1], forward])[:, 6:10]
    derivative = jnp.concatenate([back_derivative[:0:-1], forward_derivative])[:, 6:10]
    return TabulatedAngles(
        x=x,
        state=state,
        derivative=derivative,
        kappa_omega=binary.kappa_omega,
        omega_lo=omega_lo,
        omega_hi=omega_hi,
        alpha_reference=alpha_reference,
    )


def batch_arguments(
    params: Sequence["PrecessingParametersWithExtrinsic"], frequencies
) -> tuple:
    """The arguments of a :func:`precessing_waveform` function for the
    binaries ``params``, as numpy arrays.

    Parameters
    ----------
    params : sequence of PrecessingParametersWithExtrinsic
        One per row; ``reference_frequency_hz`` must be set.
    frequencies : array
        Shape ``(k,)`` or ``(len(params), k)``, in Hz.
    """
    if any(p.reference_frequency_hz is None for p in params):
        raise ValueError("the JAX precessing waveform needs a reference frequency")
    intrinsic = np.array([
        [p.mass_ratio, p.lambda_1, p.lambda_2, *p.chi_1_vector, *p.chi_2_vector]
        for p in params
    ])
    names = ("total_mass", "distance_mpc", "inclination", "azimuth",
             "reference_phase", "reference_frequency_hz", "merger_time")
    return (intrinsic, np.asarray(frequencies, dtype=float)) + tuple(
        np.array([getattr(p, name) for p in params], dtype=float) for name in names
    )


def _reference_rotation(
    predict_reference, keys, aligned, total_mass, distance_mpc, reference_phase,
    reference_frequency_hz,
):
    r"""The rotation :math:`\phi_0 - \phi_{\rm orb}` that brings the orbital
    phase at the reference frequency to ``reference_phase``: co-precessing
    multipole :math:`m` is multiplied by :math:`e^{i m \times {\rm rotation}}`.

    From the stationary-phase transforms :math:`X = \Psi - f \Psi' = \Psi +
    2 \pi f t(f)` of the :func:`reference_phase_keys` multipoles, with
    ``predict_reference`` their batched surrogate (``return_tf=True``).
    """
    _, jnp = _jnp()
    centres = jnp.stack([key[1] / 2.0 * reference_frequency_hz for key in keys], axis=1)
    _, reference, time = predict_reference(aligned, total_mass, centres, distance_mpc)
    transforms = {
        key: reference[:, i, i] + 2 * np.pi * centres[:, i] * time[:, i, i]
        for i, key in enumerate(keys)
    }
    return reference_phase - orbital_phase_from_transforms(transforms, xp=jnp)


def precessing_mode_components(
    model: "Model", modes: Optional[Sequence] = None, n_steps: int = N_STEPS
) -> Callable:
    r"""A JAX function giving each co-precessing multipole and its twist,
    batched.

    The precessing polarizations are linear in the co-precessing multipoles,

    .. math::
        \tilde{h}_{+, \times}(f) = \sum_{\ell m} c^{+, \times}_{\ell m}(f) \,
        \tilde{h}^{\rm co}_{\ell m}(f) \,,

    where the coefficients :math:`c_{\ell m}` hold the twist (the Euler
    angles at the multipole's own stationary-phase point) and the projection
    on the sky (:func:`~mlgw_bns.precessing_model.twist_coefficients`). The
    co-precessing multipoles carry the fast orbital phase, the coefficients
    only the slow precession: this is the split that mode-by-mode relative
    binning needs (Leslie, Dai and Pratten, `arXiv:2109.09872
    <https://arxiv.org/abs/2109.09872>`_, eqs. 1--4).

    Returns ``predict(...) -> (coprecessing, c_plus, c_cross)``, with the
    arguments of :func:`precessing_waveform` and three complex arrays of
    shape ``(N, n_modes, k)``, the modes in the order of ``modes``.
    ``coprecessing`` already has its orbital phase set at the reference
    frequency and the merger-time shift; :func:`precessing_waveform` is
    ``(sum(c_plus * coprecessing), sum(c_cross * coprecessing))`` over the
    modes.

    Parameters
    ----------
    model : Model
        The aligned-spin surrogate.
    modes : sequence of (l, m), optional
        Co-precessing multipoles; defaults to all of ``model.modes``.
    n_steps : int
        Steps of each leg of the precession integration
        (:func:`integrate_angles`).
    """
    jax, jnp = _jnp()
    modes = [tuple(int(i) for i in lm) for lm in (model.modes if modes is None else modes)]
    largest_m = max(m for _, m in model.modes)
    predict_modes = model.jax_modes_amp_phase(modes)
    # from the model's multipoles, as the numpy path, whichever are twisted:
    # the odd-m one picks the branch of the orbital phase
    keys = reference_phase_keys(model.modes)
    predict_reference = model.jax_modes_amp_phase(keys, return_tf=True)

    def one_binary(row, frequencies, mass_seconds, inclination, azimuth, reference_frequency):
        angles = integrate_angles(
            row[0], row[1], row[2], row[3:6], row[6:9],
            reference_frequency * mass_seconds,
            2.0 * frequencies[0] * mass_seconds / largest_m,
            n_steps,
        )
        coefficients = twist_coefficients(
            modes, frequencies, angles, mass_seconds, inclination, azimuth, xp=jnp
        )
        return (
            jnp.stack([coefficients[key][0] for key in modes]),
            jnp.stack([coefficients[key][1] for key in modes]),
        )

    def predict(
        intrinsic,
        frequencies,
        total_mass,
        distance_mpc,
        inclination,
        azimuth,
        reference_phase,
        reference_frequency_hz,
        merger_time=0.0,
    ):
        intrinsic = jnp.atleast_2d(jnp.asarray(intrinsic, jnp.float64))
        n_rows = intrinsic.shape[0]

        def per_row(value):
            return jnp.broadcast_to(jnp.asarray(value, jnp.float64), (n_rows,))

        total_mass, distance_mpc, inclination, azimuth = map(
            per_row, (total_mass, distance_mpc, inclination, azimuth)
        )
        reference_phase, reference_frequency_hz, merger_time = map(
            per_row, (reference_phase, reference_frequency_hz, merger_time)
        )
        frequencies = jnp.asarray(frequencies, jnp.float64)
        frequencies = jnp.broadcast_to(
            frequencies if frequencies.ndim == 2 else frequencies[None, :],
            (n_rows, frequencies.shape[-1]),
        )
        aligned = jnp.stack([intrinsic[:, i] for i in (0, 1, 2, 5, 8)], axis=1)
        rotation = _reference_rotation(
            predict_reference, keys, aligned, total_mass, distance_mpc,
            reference_phase, reference_frequency_hz,
        )

        amp, phase = predict_modes(aligned, total_mass, frequencies, distance_mpc)
        emms = jnp.asarray([m for _, m in modes], dtype=float)
        phase = (
            phase
            + emms[None, :, None] * rotation[:, None, None]
            - 2 * np.pi * frequencies[:, None, :] * merger_time[:, None, None]
        )
        c_plus, c_cross = jax.vmap(one_binary)(
            intrinsic, frequencies, total_mass * SUN_MASS_SECONDS,
            inclination, azimuth, reference_frequency_hz,
        )
        return amp * jnp.exp(1j * phase), c_plus, c_cross

    return predict


def precessing_waveform(
    model: "Model", modes: Optional[Sequence] = None, n_steps: int = N_STEPS
) -> Callable:
    r"""A JAX function reproducing :meth:`PrecessingModel.predict
    <mlgw_bns.precessing_model.PrecessingModel.predict>`, batched.

    Returns ``predict(intrinsic, frequencies, total_mass, distance_mpc,
    inclination, azimuth, reference_phase, reference_frequency_hz,
    merger_time=0.0) -> (h_plus, h_cross)``, where ``intrinsic`` has shape
    ``(N, 9)``, rows :math:`[q, \Lambda_1, \Lambda_2, \vec{\chi}_1,
    \vec{\chi}_2]`, ``frequencies`` ``(k,)`` or ``(N, k)`` (positive and
    increasing, in Hz) and the rest are scalars or of shape ``(N,)``; the
    polarizations have shape ``(N, k)``. The parameters are those of
    :class:`~mlgw_bns.precessing_model.PrecessingParametersWithExtrinsic`;
    the reference frequency is required. The Euler angles are integrated
    from the lowest frequency of each row, as the numpy path does. The sum
    over the modes of :func:`precessing_mode_components`.

    Parameters
    ----------
    model : Model
        The aligned-spin surrogate.
    modes : sequence of (l, m), optional
        Co-precessing multipoles to twist; defaults to all of ``model.modes``.
    n_steps : int
        Steps of each leg of the precession integration
        (:func:`integrate_angles`).
    """
    _, jnp = _jnp()
    components = precessing_mode_components(model, modes, n_steps)

    def predict(*args, **kwargs):
        coprecessing, c_plus, c_cross = components(*args, **kwargs)
        return (
            jnp.sum(c_plus * coprecessing, axis=1),
            jnp.sum(c_cross * coprecessing, axis=1),
        )

    return predict
