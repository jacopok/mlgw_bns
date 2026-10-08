r"""Regressed precession angles: a prototype.

:mod:`~mlgw_bns.batched_precession` integrates the PN spin-precession
equations of every binary it evaluates, two legs of 2048 sequential DOP853
steps, and that integration is most of the cost of a precessing waveform.
Here the Euler angles come instead from a regressor trained on those
integrations, and the only integrals left are quadratures of known
functions.

Why not regress the Euler angles themselves
-------------------------------------------
In the frame of the reference frequency (:math:`\hat{z} = \hat{L}` there)
:math:`\hat{L}` passes close to :math:`\hat{z}` once per precession cycle,
and :math:`\alpha` swings by :math:`\sim\pi` each time: :math:`\mathrm{d}
\alpha / \mathrm{d}x` is a comb of spikes whose positions move with the
parameters, which neither a PCA nor a regressor can follow. In a frame
along the total angular momentum :math:`\vec{J}` the angles are smooth for
one spin, but the two spins precess at different rates, and :math:`\alpha_J`
beats at the difference: when the two in-plane spins are comparable,
:math:`\hat{L}` circles :math:`\vec{J}` or not depending on which is the
larger, so the winding number of :math:`\alpha_J` jumps across parameter
space.

The representation
------------------
In the frame of :math:`\vec{J}` at the reference frequency (reached from the
reference frame by the minimal rotation :math:`R_0`), the in-plane part
:math:`\zeta = L_x + i L_y` of :math:`\hat{L}` is, to first order in
:math:`S / L`, a sum of the two normal modes of the precession of the two
spins about :math:`\vec{J}`, with slowly varying amplitudes:

.. math::
    \zeta(x) = c_0(x) + c_1(x)\, e^{i \Phi_1(x)} + c_2(x)\, e^{i \Phi_2(x)} .

The carrier phases :math:`\Phi_k = \int \lambda_k \, \mathrm{d}x` are the
integrals of the normal-mode frequencies of the linearized two-spin
precession (:func:`carrier_rates`), written with the PN coefficients of the
right-hand side at the spin projections of the reference point: these are
the *derivatives of the precession angles*, which are known in closed form,
and are integrated by a quadrature. Everything that winds is in the
carriers, which are not regressed and so carry no regression error; the
regressor only sees the envelopes :math:`c_k`, which are smooth in
:math:`x` and in the parameters.

The third angle follows from the minimal-rotation condition: in the
:math:`\vec{J}` frame :math:`G = \alpha_J - \gamma_J` grows as
:math:`\mathrm{d}G / \mathrm{d}x = \mathrm{Im}(\zeta^* \zeta') / (1 + \cos
\beta_J)` --- secularly, by :math:`\sim \beta_J^2 / 2` per radian of
precession, and with oscillations at the carriers and at their difference
--- and is represented the same way,

.. math::
    G(x) = g_0(x) + \mathrm{Re}\left[ g_1(x)\, e^{i \Phi_1} + g_2(x)\,
    e^{i \Phi_2} + g_{12}(x)\, e^{i (\Phi_1 - \Phi_2)} \right] .

The seven envelopes are cubic B-splines uniform in the integration
variable :math:`x` of :mod:`~mlgw_bns.batched_precession`, on a grid common
to all binaries (:class:`AngleGrid`), fitted by penalized least squares to
the integrated angles of each training binary (:func:`fit_envelopes`); their
coefficients are compressed by PCA and regressed from the parameters by
kernel ridge (:class:`PrecessionRegressor`).

From :math:`(\zeta, G)` the co-precessing frame is :math:`R = R_0 \,
R_{\min}(\hat{L}_J) \, R_z(G)`, with :math:`R_{\min}(\hat{n})` the minimal
rotation taking :math:`\hat{z}` to :math:`\hat{n}`: this is
:math:`R_z(\alpha_J) R_y(\beta_J) R_z(-\gamma_J)` written without
:math:`\alpha_J`, which is undefined where :math:`\hat{L} = \vec{J}`. Its
Euler angles in the reference frame are what :class:`RegressedAngles`
returns, as :class:`~mlgw_bns.batched_precession.TabulatedAngles` does.

The last cycles: integration
----------------------------
Above :attr:`AngleGrid.x_switch` (~165 Hz for a binary neutron star) the
carriers turn too slowly to tell the envelopes apart: a least-squares fit
there picks a split that depends on the carriers' phases, which wind with
the parameters, and the regression cannot follow it. There are only a few
precession cycles left, though, and the PN equations are integrated over
them (:func:`_tail`, :data:`N_TAIL_STEPS` fixed DOP853 steps, ~2% of what
the whole band takes), from the state the regressor gives at the switch:
:math:`\hat{L}` and :math:`\alpha - \gamma` from the envelopes, and the two
spins, whose in-plane components in the frame of :math:`\vec{J}` are fitted
on the carriers as :math:`\zeta` is and whose envelopes there are regressed
alongside (:func:`fit_switch_spins`).

Where two carriers nearly coincide (:math:`q \simeq 1`) the split between
their envelopes is again not determined; :func:`refine_envelopes` pulls each
binary towards what a regressor trained on the others predicts for it.

Status
------
A prototype. Trained on 16384 binaries (uniform in :class:`TrainingRanges`;
~25 minutes to integrate and fit on four CPU cores, ~15 to train), against
the integration (``visualization/precession_regression_study.py``, 256
held-out binaries, total masses 2.4--3.2, random lines of sight, 20--2048
Hz, ET noise, nothing optimized): waveform mismatches of 5.8e-5 median,
1.8e-3 at the 90th percentile, 6e-2 at worst. The error falls with the
training set roughly as :math:`N^{-0.6}` (1.1e-4 median with 4096, 7.9e-5
with 8192), but only for :math:`q \gtrsim 1.5` (median 3e-5 for
:math:`q > 2`). Near equal masses (:math:`q < 1.5`: median 2--7e-4, 90th
percentile ~1e-2) it does not, nor with twice as many binaries there: the
beat of the two spins is a nonlinear function of them that the carriers get
wrong by up to ~20% (the linearized model is good to ~1% for :math:`q >
1.5`), and the envelopes wind across the parameter space with the error.
The representation itself gives back the integrated rotation to ~1e-6 rad
below the switch and ~1e-4 rad above it. On four CPU cores the angles take
10 ms for one binary and 0.8 ms each in a batch of 128, against 155 ms and
29 ms for the integration (mostly the 48 sequential steps above the
switch); the whole precessing waveform 25 ms and 7.5 ms, against 155 ms and
34 ms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Optional, Tuple

import numpy as np

from .batched_precession import (
    KAPPA,
    N_STEPS,
    _Binary,
    _jnp,
    _reference_state,
    integrate_angles,
)
from .taylorf2 import SUN_MASS_SECONDS
from .twist_waveform import (
    _orbital_frequency_rate,
    _spin_orbit_coefficients,
    nu_to_X1,
    orbital_angular_momentum,
    tidal_flux_coefficient,
)

#: Lowest orbital frequency :math:`M \Omega` of the angles: the (4, 4) at
#: 20 Hz for a total mass of 1.6, the (2, 2) at 5.7 Hz for 2.8.
OMEGA_MIN = 2.5e-4

#: Highest orbital frequency: the (2, 1) at the top of the surrogate's band
#: (:math:`M f = 0.0403`). Lower than TEOBResumS' stopping frequency
#: everywhere in the BNS parameter space (0.33--0.44), so the angles are
#: never held there.
OMEGA_MAX = 0.26

#: B-spline cells of the envelopes.
N_CELLS = 128

#: Nodes of the quadrature giving the carrier phases.
N_QUADRATURE = 1024

#: Smallest splitting of the two normal-mode frequencies, relative to their
#: mean: where the linearized system is degenerate its frequencies meet with
#: a square-root branch point, which this rounds off.
SPLITTING_FLOOR = 1e-3

#: Penalty on the second differences of the envelopes' B-spline
#: coefficients, and on their size, relative to the mean diagonal of the
#: least-squares system: they pick the smoothest envelopes where two
#: carriers nearly coincide.
SMOOTHING_PENALTY = 1e-6
RIDGE_PENALTY = 1e-9

#: Where the regression hands over to the integration, in ``x``: an orbital
#: frequency of 7.2e-3, a (2, 2) frequency of ~165 Hz for a total mass of
#: 2.8. Above it the carriers turn too slowly to tell the envelopes apart,
#: and the last few precession cycles are cheap to integrate.
X_SWITCH = -5.5

#: Fixed DOP853 steps from :data:`X_SWITCH` to :data:`OMEGA_MAX`. The angles
#: are within ~1e-4 rad of the 2048-step integration's: the error of the
#: cubic Hermite interpolation between the steps, which falls as their
#: number to the fourth power.
N_TAIL_STEPS = 48

#: The integration from the switch steps uniformly in :math:`\ln \Omega -
#: \kappa \Omega_{\rm switch} / \Omega`, with this :math:`\kappa`,
#: closer to uniform in the precession phase, which turns fastest at the
#: switch: 48 such steps are as accurate as ~100 uniform in :math:`\ln
#: \Omega`.
TAIL_KAPPA = 16.0

#: Penalty pulling the envelopes towards a prior where they are not
#: determined by the data, between the two carriers where they nearly
#: coincide (:func:`refine_envelopes`).
PRIOR_PENALTY = 1e-4


@dataclass(frozen=True)
class AngleGrid:
    """The orbital frequencies the angles are represented on.

    The angles are regressed from ``omega_min`` up to ``x_switch``
    (:attr:`x_range`), and integrated from there to ``omega_max``.

    Parameters
    ----------
    omega_min, omega_max : float
        Range of :math:`M \\Omega`.
    n_cells : int
        Cubic B-spline cells of the envelopes, uniform in ``x``.
    n_quadrature : int
        Nodes of the carrier quadrature, uniform in ``x``.
    x_switch : float
        Where the integration takes over.
    n_tail_steps : int
        Its fixed DOP853 steps.
    """

    omega_min: float = OMEGA_MIN
    omega_max: float = OMEGA_MAX
    n_cells: int = N_CELLS
    n_quadrature: int = N_QUADRATURE
    x_switch: float = X_SWITCH
    n_tail_steps: int = N_TAIL_STEPS
    tail_kappa: float = TAIL_KAPPA

    @property
    def kappa_omega(self) -> float:
        """:math:`\\kappa \\Omega_{\\min}` of the variable ``x``."""
        return KAPPA * self.omega_min

    def x_of(self, omega, xp=np):
        """The integration variable of :mod:`~mlgw_bns.batched_precession`,
        :math:`x = \\ln \\Omega - \\kappa \\Omega_{\\min} / \\Omega`."""
        return xp.log(omega) - self.kappa_omega / omega

    @property
    def x_range(self) -> Tuple[float, float]:
        """The range of ``x`` the angles are regressed on."""
        return float(self.x_of(self.omega_min)), float(self.x_switch)

    @property
    def x_end(self) -> float:
        """``x`` at ``omega_max``, where the integration stops."""
        return float(self.x_of(self.omega_max))

    @property
    def omega_switch(self) -> float:
        return float(self.omega_of(self.x_switch))

    @property
    def tail_kappa_omega(self) -> float:
        return self.tail_kappa * self.omega_switch

    def tail_x_of(self, omega, xp=np):
        """The variable the integration from the switch steps uniformly in."""
        return xp.log(omega) - self.tail_kappa_omega / omega

    @property
    def tail_x_range(self) -> Tuple[float, float]:
        return float(self.tail_x_of(self.omega_switch)), float(self.tail_x_of(self.omega_max))

    @property
    def n_coefficients(self) -> int:
        """B-spline coefficients per envelope."""
        return self.n_cells + 3

    def omega_of(self, x) -> np.ndarray:
        """The inverse of :meth:`x_of` (numpy)."""
        x = np.asarray(x, dtype=float)
        # x(ln omega) is increasing and concave: Newton converges from anywhere
        log_omega = np.full_like(x, np.log(self.omega_max))
        for _ in range(100):
            residual = log_omega - self.kappa_omega * np.exp(-log_omega) - x
            log_omega -= residual / (1.0 + self.kappa_omega * np.exp(-log_omega))
        return np.exp(log_omega)

    @property
    def coefficient_x(self) -> np.ndarray:
        """Where each B-spline coefficient's basis function peaks."""
        lo, hi = self.x_range
        return lo + (np.arange(self.n_coefficients) - 1) * (hi - lo) / self.n_cells

    def quadrature_nodes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(x, omega, d omega / dx)`` at the carrier quadrature nodes."""
        x = np.linspace(*self.x_range, self.n_quadrature)
        omega = self.omega_of(x)
        omega[0] = self.omega_min
        return x, omega, omega**2 / (omega + self.kappa_omega)


def _bspline(xp, grid: AngleGrid, x):
    """Cell index and the four uniform cubic B-spline weights at ``x``: the
    value is ``sum(coefficients[..., cell + r] * weights[..., r])``."""
    lo, hi = grid.x_range
    u = (x - lo) * (grid.n_cells / (hi - lo))
    cell = xp.clip(xp.floor(u), 0, grid.n_cells - 1)
    t = u - cell
    t2 = t * t
    t3 = t2 * t
    weights = xp.stack(
        [(1 - t) ** 3, 3 * t3 - 6 * t2 + 4, -3 * t3 + 3 * t2 + 3 * t + 1, t3], axis=-1
    ) / 6.0
    return cell.astype(int), weights


def _minimal_rotation(xp, n):
    """The rotation taking :math:`\\hat{z}` to the unit vector ``n``, about
    :math:`\\hat{z} \\times n`, as a ``(..., 3, 3)`` array; ``n_z > -1``."""
    n_x, n_y, n_z = n[..., 0], n[..., 1], n[..., 2]
    scale = 1.0 / (1.0 + n_z)
    one = xp.ones_like(n_z)
    return xp.stack([
        xp.stack([1 - n_x * n_x * scale, -n_x * n_y * scale, n_x], axis=-1),
        xp.stack([-n_x * n_y * scale, 1 - n_y * n_y * scale, n_y], axis=-1),
        xp.stack([-n_x, -n_y, n_z * one], axis=-1),
    ], axis=-2)


def _rotation_z(xp, angle):
    cos, sin = xp.cos(angle), xp.sin(angle)
    zero, one = xp.zeros_like(angle), xp.ones_like(angle)
    return xp.stack([
        xp.stack([cos, -sin, zero], axis=-1),
        xp.stack([sin, cos, zero], axis=-1),
        xp.stack([zero, zero, one], axis=-1),
    ], axis=-2)


def euler_angles_of(xp, rotation):
    """:math:`(\\alpha, \\beta, \\gamma)` of ``rotation`` :math:`= R_z(\\alpha)
    R_y(\\beta) R_z(-\\gamma)`, the convention of the precession angles.

    :math:`\\alpha - \\gamma` comes from the trace of the upper block, so it
    is accurate where :math:`\\beta \\simeq 0` and :math:`\\alpha` is not.
    """
    r = rotation
    alpha = xp.arctan2(r[..., 1, 2], r[..., 0, 2])
    beta = xp.arctan2(xp.hypot(r[..., 0, 2], r[..., 1, 2]), r[..., 2, 2])
    alpha_minus_gamma = xp.arctan2(
        r[..., 1, 0] - r[..., 0, 1], r[..., 0, 0] + r[..., 1, 1]
    )
    return alpha, beta, alpha - alpha_minus_gamma


@dataclass
class ReferenceFrame:
    """What the regression needs of one binary at its reference point.

    Built by :func:`reference_frame`, for numpy or JAX.
    """

    nu: Any
    #: :math:`M_A / M_B \\geq 1`
    mass_ratio: Any
    a10_tidal: Any
    spin_a: Any  # S_A at the reference, shape (3,)
    spin_b: Any
    #: :math:`R_0`, from the reference frame to the frame of :math:`\vec{J}`
    rotation: Any
    #: :math:`G` at the reference
    g_reference: Any
    x_reference: Any


def reference_frame(xp, grid: AngleGrid, intrinsic, omega_reference) -> ReferenceFrame:
    """:class:`ReferenceFrame` of the binary ``intrinsic`` :math:`= [q,
    \\Lambda_1, \\Lambda_2, \\vec{\\chi}_1, \\vec{\\chi}_2]`, with the spins
    given at the orbital frequency ``omega_reference``."""
    q = intrinsic[0]
    nu = q / (1.0 + q) ** 2
    mass_a = nu_to_X1(nu, xp)
    mass_b = 1.0 - mass_a
    chi_1, chi_2 = intrinsic[3:6], intrinsic[6:9]
    spin_a, spin_b = chi_1 * mass_a**2, chi_2 * mass_b**2
    a10 = tidal_flux_coefficient(nu, intrinsic[1], intrinsic[2], xp=xp)

    total = spin_a + spin_b
    j = total + xp.stack([0.0 * total[0], 0.0 * total[0],
                          orbital_angular_momentum(nu, omega_reference)])
    rotation = _minimal_rotation(xp, j / xp.sqrt(xp.sum(j * j)))

    binary = _Binary(nu=nu, q=mass_a / mass_b, a10_tidal=a10, kappa_omega=1.0)
    y0, _ = _reference_state(binary, chi_1, chi_2, omega_reference / np.pi)
    alpha_minus_gamma = y0[9] if xp is not np else float(y0[9])
    # R_J(ref) = R_0^T R_z(alpha - gamma) = R_min(L_J) R_z(G)
    in_j = xp.swapaxes(rotation, -1, -2) @ _rotation_z(xp, alpha_minus_gamma)
    l_j = in_j[..., :, 2]
    rest = xp.swapaxes(_minimal_rotation(xp, l_j), -1, -2) @ in_j
    g_reference = xp.arctan2(rest[1, 0], rest[0, 0])
    return ReferenceFrame(
        nu=nu,
        mass_ratio=mass_a / mass_b,
        a10_tidal=a10,
        spin_a=spin_a,
        spin_b=spin_b,
        rotation=rotation,
        g_reference=g_reference,
        x_reference=grid.x_of(omega_reference, xp),
    )


def carrier_rates(xp, frame: ReferenceFrame, omega, domega_dx):
    r"""The normal-mode frequencies :math:`\mathrm{d}\Phi_k / \mathrm{d}x` of
    the two spins' precession about :math:`\vec{J}`.

    The in-plane components :math:`s_{A,B}` of the spins in the frame of
    :math:`\vec{J}` precess, to first order in :math:`S / L`, as
    :math:`\dot{s} = i M s` with

    .. math::
        M = \begin{pmatrix}
            w_A (\cos\beta_J + S_A^J / L) + h S_B^J &
            S_A^J (w_A / L - h) \\
            S_B^J (w_B / L - h) &
            w_B (\cos\beta_J + S_B^J / L) + h S_A^J
        \end{pmatrix} ,

    from :math:`\dot{\vec{S}}_A = w_A \hat{L} \times \vec{S}_A + h \vec{S}_B
    \times \vec{S}_A` and :math:`L \hat{L}_\perp = -(s_A + s_B)`: :math:`w_{A,
    B}` are the spin-orbit precession frequencies of the right-hand side
    (:func:`~mlgw_bns.twist_waveform._spin_orbit_coefficients`), :math:`h =
    v^6 / 2` the spin-spin one, :math:`L` the 2PN orbital angular momentum
    and :math:`S^J`, :math:`\beta_J` the spins' projections on, and the tilt
    of :math:`\hat{L}` from, :math:`\vec{J} = L \hat{L} + \vec{S}_A +
    \vec{S}_B` as :math:`L` shrinks, with the spins' projections on
    :math:`\hat{L}` and their magnitudes held at the reference (which the
    precession conserves to leading order) and the in-plane part of
    :math:`\vec{S}_A \cdot \vec{S}_B` averaged over the beat of the two
    spins. For a single spin these are the exact precession frequency
    :math:`w J / L`.

    To second order in the tilts, each mode is also carried round by the
    other's circling of :math:`\hat{L}` about :math:`\vec{J}`: in the frame
    turning with the other mode at :math:`\Omega`, a spin precesses about
    :math:`w \hat{L} - \Omega \hat{z}`, at :math:`|w \hat{L} - \Omega
    \hat{z}|`. Without this the lighter spin's carrier drifts from its
    mode by up to several radians over the band. Converted to ``x`` with the
    PN :math:`\dot\Omega` of the same spins.
    """
    nu, q = frame.nu, frame.mass_ratio
    spin_a, spin_b = frame.spin_a, frame.spin_b
    v = omega ** (1.0 / 3.0)
    v5_ca, v5_cb, v7_ca, v7_cb, v9_ca, v9_cb = _spin_orbit_coefficients(nu, q, xp)
    sa_l, sb_l = spin_a[2], spin_b[2]
    w_a = v**5 * v5_ca - 1.5 * v**6 * (sa_l / q + sb_l) + v**7 * v7_ca + v**9 * v9_ca
    w_b = v**5 * v5_cb - 1.5 * v**6 * (sa_l + sb_l * q) + v**7 * v7_cb + v**9 * v9_cb
    # the in-plane part of S_A . S_B turns at the beat of the two spins'
    # precession: its average over the beat is zero
    sa_sb = sa_l * sb_l
    omega_dot, _ = _orbital_frequency_rate(
        nu, omega, sa_l, sb_l, sa_sb, xp.sum(spin_a * spin_a),
        xp.sum(spin_b * spin_b), frame.a10_tidal, xp=xp,
    )
    # J, and the projections on it, as L shrinks: the spins' projections on
    # L, and their magnitudes, held at the reference (the precession
    # conserves them to leading order)
    l_magnitude = orbital_angular_momentum(nu, omega)
    sa_s = xp.sum(spin_a * spin_a) + sa_sb
    sb_s = xp.sum(spin_b * spin_b) + sa_sb
    s_l = sa_l + sb_l
    j_magnitude = xp.sqrt(l_magnitude**2 + 2 * l_magnitude * s_l + sa_s + sb_s)
    cos_beta = (l_magnitude + s_l) / j_magnitude
    sa_j = (l_magnitude * sa_l + sa_s) / j_magnitude
    sb_j = (l_magnitude * sb_l + sb_s) / j_magnitude
    h = 0.5 * v**6
    m_aa = w_a * (cos_beta + sa_j / l_magnitude) + h * sb_j
    m_bb = w_b * (cos_beta + sb_j / l_magnitude) + h * sa_j
    m_ab = sa_j * (w_a / l_magnitude - h)
    m_ba = sb_j * (w_b / l_magnitude - h)
    mean = (m_aa + m_bb) / 2.0
    discriminant = (m_aa - m_bb) ** 2 / 4.0 + m_ab * m_ba
    floor = (SPLITTING_FLOOR * mean) ** 2
    split = xp.sqrt((discriminant + xp.sqrt(discriminant**2 + floor**2)) / 2.0)
    fast, slow = mean + split, mean - split
    # Each mode is also carried round by the other's circling of L about J:
    # in the frame turning with the other mode, it precesses about the tilted
    # w L - Omega z, at |w L - Omega z|. The fast mode is the lighter body's
    # spin (w_B > w_A for q > 1), tilting L by beta_B; the slow one spin A's.
    rotation = frame.rotation
    tilt_a = xp.sum((rotation.T @ spin_a)[:2] ** 2) / j_magnitude**2
    tilt_b = xp.sum((rotation.T @ spin_b)[:2] ** 2) / j_magnitude**2
    cross = 2.0 * fast * slow
    fast_shifted = slow + xp.sqrt((fast - slow) ** 2 + cross * tilt_a / (1 + xp.sqrt(1 - tilt_a)))
    slow_shifted = fast - xp.sqrt((fast - slow) ** 2 + cross * tilt_b / (1 + xp.sqrt(1 - tilt_b)))
    scale = domega_dx / omega_dot
    return xp.stack([fast_shifted * scale, slow_shifted * scale])


def _hermite(xp, lo, hi, values, slopes, x):
    """Cubic Hermite interpolation of ``values`` ``(..., n)``, given at ``n``
    nodes uniform on ``[lo, hi]`` with derivatives ``slopes``, at ``x``."""
    n = values.shape[-1] - 1
    width = (hi - lo) / n
    u = (x - lo) / width
    cell = xp.clip(xp.floor(u), 0, n - 1).astype(int)
    t = u - cell
    t2 = t * t
    t3 = t2 * t
    take = lambda array, i: array[..., i]  # noqa: E731
    return (
        (2 * t3 - 3 * t2 + 1) * take(values, cell)
        + (t3 - 2 * t2 + t) * width * take(slopes, cell)
        + (-2 * t3 + 3 * t2) * take(values, cell + 1)
        + (t3 - t2) * width * take(slopes, cell + 1)
    )


def carrier_table(xp, grid: AngleGrid, frame: ReferenceFrame, nodes=None):
    """The carrier phases and their rates at the quadrature nodes, ``(2,
    n_quadrature)`` each, with the phases zero at the reference."""
    x, omega, domega_dx = grid.quadrature_nodes() if nodes is None else nodes
    rates = carrier_rates(xp, frame, xp.asarray(omega), xp.asarray(domega_dx))
    width = (grid.x_range[1] - grid.x_range[0]) / (grid.n_quadrature - 1)
    zero = xp.zeros_like(rates[:, :1])
    phases = xp.concatenate(
        [zero, xp.cumsum(0.5 * width * (rates[:, 1:] + rates[:, :-1]), axis=-1)], axis=-1
    )
    at_reference = _hermite(xp, *grid.x_range, phases, rates, frame.x_reference)
    return phases - at_reference[:, None], rates


def _carriers(xp, grid: AngleGrid, phases, rates, x):
    r""":math:`e^{i \Phi_1}, e^{i \Phi_2}` at ``x``."""
    phase = _hermite(xp, *grid.x_range, phases, rates, x)
    return xp.exp(1j * phase[0]), xp.exp(1j * phase[1])


def g_baseline(xp, grid: AngleGrid, frame: ReferenceFrame, rates, nodes=None):
    r"""B-spline coefficients of the secular growth of :math:`G`, to leading
    order: what the regressor learns of :math:`g_0` is the rest.

    With the two modes taken as the two spins' in-plane components
    :math:`s_{A, B}` in the frame of :math:`\vec{J}`, carried by the slow and
    the fast carrier, :math:`\mathrm{d}G / \mathrm{d}x = (|s_A|^2
    \lambda_{\rm slow} + |s_B|^2 \lambda_{\rm fast}) / (L^2 (1 + \cos
    \beta_J))`, zero at the reference; this is ~90% of :math:`g_0` below ~250 Hz.
    The coefficients are the values at
    :attr:`AngleGrid.coefficient_x`, so the baseline is not quite this
    function; it is the same in training and prediction.
    """
    x, omega, _ = grid.quadrature_nodes() if nodes is None else nodes
    l_squared = orbital_angular_momentum(frame.nu, xp.asarray(omega)) ** 2
    rotation_t = xp.swapaxes(frame.rotation, -1, -2)
    in_plane_a = xp.sum((rotation_t @ frame.spin_a)[:2] ** 2)
    in_plane_b = xp.sum((rotation_t @ frame.spin_b)[:2] ** 2)
    tilt_squared = xp.minimum((in_plane_a + in_plane_b) / l_squared, 1.0)
    rate = (in_plane_a * rates[1] + in_plane_b * rates[0]) / (
        l_squared * (1.0 + xp.sqrt(1.0 - tilt_squared))
    )
    lo, hi = grid.x_range
    width = (hi - lo) / (grid.n_quadrature - 1)
    g = xp.concatenate([xp.zeros(1), xp.cumsum(0.5 * width * (rate[1:] + rate[:-1]))])
    g = g - _hermite(xp, lo, hi, g, rate, frame.x_reference)
    return _hermite(xp, lo, hi, g, rate, xp.clip(xp.asarray(grid.coefficient_x), lo, hi))


#: The envelopes, in the order of the coefficient arrays: three of zeta, four
#: of G (``g_0`` is real).
ENVELOPES = ("c_0", "c_1", "c_2", "g_0", "g_1", "g_2", "g_12")


def _evaluate(xp, grid: AngleGrid, coefficients, carriers, x):
    r"""``(zeta, G - G_ref)`` at ``x`` from the ``(7, n_coefficients)``
    complex envelope coefficients and the carriers there."""
    cell, weights = _bspline(xp, grid, x)
    envelopes = sum(coefficients[:, cell + r] * weights[..., r] for r in range(4))
    e_1, e_2 = carriers
    zeta = envelopes[0] + envelopes[1] * e_1 + envelopes[2] * e_2
    g = envelopes[3].real + (
        envelopes[4] * e_1 + envelopes[5] * e_2 + envelopes[6] * e_1 * xp.conj(e_2)
    ).real
    return zeta, g


def _rotation_from(xp, frame_rotation, zeta, g):
    r""":math:`R_0 \, R_{\min}(\hat{L}_J) \, R_z(G)`."""
    sin_squared = xp.minimum(zeta.real**2 + zeta.imag**2, 1.0)
    l_j = xp.stack([zeta.real, zeta.imag, xp.sqrt(1.0 - sin_squared)], axis=-1)
    return frame_rotation @ _minimal_rotation(xp, l_j) @ _rotation_z(xp, g)


@dataclass
class RegressedAngles:
    r"""The Euler angles of one binary (or a batch, along a first axis), from
    a :class:`PrecessionRegressor`.

    Has the :meth:`at_momega` of
    :class:`~mlgw_bns.batched_precession.TabulatedAngles`, so it can stand in
    for it in the twist. The arrays are numpy or JAX.
    """

    coefficients: Any  # (7, n_coefficients), complex
    phases: Any  # (2, n_quadrature)
    rates: Any  # (2, n_quadrature)
    rotation: Any  # (3, 3)
    g_reference: Any
    #: [L_x, L_y, L_z, alpha - gamma] above x_switch, (n_tail_steps + 1, 4)
    tail_state: Any
    tail_derivative: Any
    grid: AngleGrid = field(default_factory=AngleGrid)

    def _xp(self):
        if isinstance(self.coefficients, np.ndarray):
            return np
        return _jnp()[1]

    def at_momega(self, momega):
        r""":math:`(\alpha, \beta, \gamma)` at the orbital frequencies
        ``momega``, held at the ends of :attr:`grid`."""
        xp = self._xp()
        grid = self.grid
        omega = xp.clip(momega, grid.omega_min, grid.omega_max)
        x = grid.x_of(omega, xp)
        below = xp.minimum(x, grid.x_switch)
        carriers = _carriers(xp, grid, self.phases, self.rates, below)
        zeta, g = _evaluate(xp, grid, self.coefficients, carriers, below)
        regressed = euler_angles_of(
            xp, _rotation_from(xp, self.rotation, zeta, g + self.g_reference)
        )
        l_x, l_y, l_z, alpha_minus_gamma = _hermite(
            xp, *grid.tail_x_range, self.tail_state.T, self.tail_derivative.T,
            xp.clip(grid.tail_x_of(omega, xp), *grid.tail_x_range),
        )
        alpha = xp.arctan2(l_y, l_x)
        integrated = (
            alpha, xp.arctan2(xp.hypot(l_x, l_y), l_z), alpha - alpha_minus_gamma
        )
        above = x > grid.x_switch
        return tuple(xp.where(above, b, a) for a, b in zip(regressed, integrated))


def _register_pytree():
    """Make :class:`RegressedAngles` a JAX pytree (its grid is static)."""
    jax, _ = _jnp()
    if getattr(_register_pytree, "done", False):
        return
    jax.tree_util.register_dataclass(
        RegressedAngles,
        data_fields=[f.name for f in fields(RegressedAngles) if f.name != "grid"],
        meta_fields=["grid"],
    )
    _register_pytree.done = True


# ---------------------------------------------------------------------------
# Training targets
# ---------------------------------------------------------------------------


def j_frame_angles(frame: ReferenceFrame, x, state, derivative, reference_index):
    r""":math:`\zeta` and :math:`G - G_{\rm ref}` in the frame of :math:`\vec{J}`,
    from the integrated :math:`\hat{L}` (the ``state`` and ``derivative`` of
    :class:`~mlgw_bns.batched_precession.TabulatedAngles`, numpy).

    :math:`G` is the trapezoidal quadrature of :math:`\mathrm{Im}(\zeta^*
    \zeta') / (1 + \cos\beta_J)` on the integration nodes.
    """
    rotation = np.asarray(frame.rotation)
    l_j = state[:, :3] @ rotation
    dl_j = derivative[:, :3] @ rotation
    zeta = l_j[:, 0] + 1j * l_j[:, 1]
    dg = (l_j[:, 0] * dl_j[:, 1] - l_j[:, 1] * dl_j[:, 0]) / (1.0 + l_j[:, 2])
    g = np.concatenate([[0.0], np.cumsum(0.5 * np.diff(x) * (dg[1:] + dg[:-1]))])
    return zeta, g - g[reference_index]


def _penalized_least_squares(cell, weights, factors, data, n_coefficients, prior=None):
    """Coefficients of ``sum_b factors[:, b] * spline_b(x)`` fitting ``data``.

    ``factors`` ``(n, n_blocks)`` multiply one B-spline envelope each; complex
    factors and data give complex coefficients. The normal equations are
    assembled from the four nonzero B-splines of each point, with
    :data:`SMOOTHING_PENALTY` and :data:`RIDGE_PENALTY`; with a ``prior``
    ``(n_blocks, n_coefficients)``, the ridge pulls towards it, with
    :data:`PRIOR_PENALTY`, instead of towards zero.
    """
    n_points, n_blocks = factors.shape
    # (n, n_blocks * 4) columns and values of the design matrix
    columns = (
        np.arange(n_blocks)[None, :, None] * n_coefficients
        + cell[:, None, None]
        + np.arange(4)[None, None, :]
    ).reshape(n_points, -1)
    values = (factors[:, :, None] * weights[:, None, :]).reshape(n_points, -1)
    size = n_blocks * n_coefficients
    flat = (columns[:, :, None] * size + columns[:, None, :]).ravel()
    products = (np.conj(values)[:, :, None] * values[:, None, :]).ravel()
    complex_data = np.iscomplexobj(factors) or np.iscomplexobj(data)

    def accumulate(index, weight, length):
        total = np.bincount(index, weights=weight.real, minlength=length)
        if complex_data:
            total = total + 1j * np.bincount(index, weights=weight.imag, minlength=length)
        return total

    gram = accumulate(flat, products, size * size).reshape(size, size)
    rhs = accumulate(columns.ravel(), (np.conj(values) * data[:, None]).ravel(), size)
    scale = np.trace(gram).real / size
    second_difference = np.diff(np.eye(n_coefficients), 2, axis=0)
    penalty = np.kron(
        np.eye(n_blocks),
        SMOOTHING_PENALTY * second_difference.T @ second_difference
        + RIDGE_PENALTY * np.eye(n_coefficients),
    )
    if prior is not None:
        penalty = penalty + PRIOR_PENALTY * np.eye(size)
        rhs = rhs + scale * PRIOR_PENALTY * np.ravel(prior)
    solution = np.linalg.solve(gram + scale * penalty, rhs)
    return solution.reshape(n_blocks, n_coefficients)


def fit_envelopes(grid: AngleGrid, frame: ReferenceFrame, x, zeta, g, prior=None):
    """The ``(7, n_coefficients)`` envelope coefficients fitting ``zeta``
    and ``g`` at the ``x`` within :attr:`AngleGrid.x_range` (numpy), and the
    largest residuals of the two fits; ``prior``, see
    :func:`refine_envelopes`."""
    inside = x <= grid.x_range[1]
    x, zeta, g = x[inside], zeta[inside], g[inside]
    phases, rates = carrier_table(np, grid, frame)
    e_1, e_2 = _carriers(np, grid, phases, rates, x)
    cell, weights = _bspline(np, grid, x)
    n = grid.n_coefficients
    zeta_fit = _penalized_least_squares(
        cell, weights, np.stack([np.ones_like(e_1), e_1, e_2], axis=1), zeta, n,
        None if prior is None else prior[:3],
    )
    e_12 = e_1 * np.conj(e_2)
    g_factors = np.stack(
        [np.ones_like(x)] + [part for e in (e_1, e_2, e_12) for part in (e.real, -e.imag)],
        axis=1,
    )
    g_prior = None if prior is None else np.stack(
        [prior[3].real] + [part for c in prior[4:] for part in (c.real, c.imag)]
    )
    g_fit = _penalized_least_squares(cell, weights, g_factors, g, n, g_prior)
    coefficients = np.concatenate([
        zeta_fit,
        g_fit[:1].astype(complex),
        g_fit[1::2] + 1j * g_fit[2::2],
    ])
    zeta_model, g_model = _evaluate(np, grid, coefficients, (e_1, e_2), x)
    return coefficients, (np.max(np.abs(zeta_model - zeta)), np.max(np.abs(g_model - g)))


#: The state at :attr:`AngleGrid.x_switch` the regressor gives besides the
#: envelopes: for each spin, the values there of the three envelopes of its
#: in-plane components in the frame of :math:`\vec{J}` (real and imaginary
#: parts), then the two spins' components along :math:`\vec{J}`.
N_SWITCH = 14


def fit_switch_spins(grid: AngleGrid, frame: ReferenceFrame, x, spin_a, spin_b) -> np.ndarray:
    r"""The :data:`N_SWITCH` regression targets of the spins at
    :attr:`AngleGrid.x_switch`, from their components in the frame of
    :math:`\vec{J}` at ``x``, ``(n, 3)`` each (numpy).

    The in-plane components are fitted on the carriers as :math:`\zeta` is
    (:func:`fit_envelopes`): they wind with the precession, their envelopes
    do not.
    """
    inside = x <= grid.x_range[1]
    phases, rates = carrier_table(np, grid, frame)
    e_1, e_2 = _carriers(np, grid, phases, rates, x[inside])
    cell, weights = _bspline(np, grid, x[inside])
    at_switch_cell, at_switch_weights = _bspline(np, grid, np.array([grid.x_switch]))
    factors = np.stack([np.ones_like(e_1), e_1, e_2], axis=1)
    targets = []
    for spin in (spin_a, spin_b):
        envelopes = _penalized_least_squares(
            cell, weights, factors, spin[inside, 0] + 1j * spin[inside, 1],
            grid.n_coefficients,
        )
        values = sum(
            envelopes[:, at_switch_cell[0] + r] * at_switch_weights[0, r] for r in range(4)
        )
        targets += [part for value in values for part in (value.real, value.imag)]
    targets += [np.interp(grid.x_switch, x, spin[:, 2]) for spin in (spin_a, spin_b)]
    return np.array(targets)


def _tail(grid: AngleGrid, frame: ReferenceFrame, coefficients, phases, rates, switch):
    r"""Integrate the precession from :attr:`AngleGrid.x_switch` to
    ``omega_max``, from the regressed state there: :math:`\hat{L}` and
    :math:`\alpha - \gamma` of the envelopes, the spins of ``switch`` (their
    magnitudes restored). Returns ``[L_x, L_y, L_z, \alpha - \gamma]`` and
    their ``x`` derivatives at the ``n_tail_steps + 1`` nodes (JAX)."""
    from .batched_precession import _leg

    _, jnp = _jnp()
    x_switch = jnp.asarray(grid.x_switch)
    e_1, e_2 = _carriers(jnp, grid, phases, rates, x_switch)
    zeta, g = _evaluate(jnp, grid, coefficients, (e_1, e_2), x_switch)
    rotation = _rotation_from(jnp, frame.rotation, zeta, g + frame.g_reference)
    alpha, _, gamma = euler_angles_of(jnp, rotation)
    spins = []
    for i, spin in enumerate((frame.spin_a, frame.spin_b)):
        values = switch[6 * i:6 * i + 6:2] + 1j * switch[6 * i + 1:6 * i + 6:2]
        in_plane = values[0] + values[1] * e_1 + values[2] * e_2
        vector = jnp.stack([in_plane.real, in_plane.imag, switch[12 + i]])
        norm = jnp.sqrt(jnp.sum(vector**2))
        scale = jnp.where(norm > 0, jnp.sqrt(jnp.sum(spin**2)) / jnp.where(norm > 0, norm, 1.0), 0.0)
        spins.append(frame.rotation @ (vector * scale))
    y0 = jnp.concatenate([
        spins[0], spins[1], rotation[:, 2],
        jnp.stack([alpha - gamma, jnp.asarray(grid.omega_switch)]),
    ])
    binary = _Binary(
        nu=frame.nu, q=frame.mass_ratio, a10_tidal=frame.a10_tidal,
        kappa_omega=grid.tail_kappa_omega,
    )
    states, derivatives = _leg(
        binary, y0, (grid.tail_x_range[1] - grid.tail_x_range[0]) / grid.n_tail_steps,
        grid.n_tail_steps,
    )
    return states[:, 6:10], derivatives[:, 6:10]


# ---------------------------------------------------------------------------
# The regressor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingRanges:
    """The parameter space a :class:`PrecessionRegressor` is trained on.

    Uniform in each of these, and in the direction of each in-plane spin;
    log-uniform in the reference orbital frequency :math:`M \\Omega_{\\rm
    ref} = \\pi M f_{\\rm ref}` (the default covers 20 Hz for total masses
    of 1.3--10). The mass ratio is :math:`q_{\\min} + (q_{\\max} -
    q_{\\min}) u^p` with :math:`u` uniform and :math:`p` =
    ``mass_ratio_exponent``: :math:`p > 1` puts more of the binaries near
    equal masses, where the two spins precess at comparable rates and the
    envelopes are hardest to regress.
    """

    mass_ratio: Tuple[float, float] = (1.0, 3.0)
    lambda_: Tuple[float, float] = (5.0, 5000.0)
    chi_z: Tuple[float, float] = (-0.5, 0.5)
    chi_in_plane: Tuple[float, float] = (0.0, 0.4)
    omega_reference: Tuple[float, float] = (4e-4, 3e-3)
    mass_ratio_exponent: float = 1.0

    def sample(self, n: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """``n`` binaries: the ``(n, 9)`` intrinsic rows of
        :func:`~mlgw_bns.batched_precession.batch_arguments` and the
        ``(n,)`` reference orbital frequencies."""
        rng = np.random.default_rng(seed)
        intrinsic = np.empty((n, 9))
        low, high = self.mass_ratio
        intrinsic[:, 0] = low + (high - low) * rng.uniform(0.0, 1.0, n) ** self.mass_ratio_exponent
        intrinsic[:, 1:3] = rng.uniform(*self.lambda_, (n, 2))
        for start in (3, 6):
            magnitude = rng.uniform(*self.chi_in_plane, n)
            angle = rng.uniform(0.0, 2.0 * np.pi, n)
            intrinsic[:, start] = magnitude * np.cos(angle)
            intrinsic[:, start + 1] = magnitude * np.sin(angle)
            intrinsic[:, start + 2] = rng.uniform(*self.chi_z, n)
        omega_reference = np.exp(rng.uniform(*np.log(self.omega_reference), n))
        return intrinsic, omega_reference


def regression_features(xp, intrinsic, omega_reference):
    r"""The inputs of the regressor, ``(N, 9)``: :math:`q`, the log of the
    tidal coefficient of :math:`\dot\Omega` (through which alone the tidal
    polarizabilities enter), the six spin components and
    :math:`\ln M\Omega_{\rm ref}`."""
    q = intrinsic[:, 0]
    nu = q / (1.0 + q) ** 2
    a10 = tidal_flux_coefficient(nu, intrinsic[:, 1], intrinsic[:, 2], xp=xp)
    return xp.concatenate(
        [q[:, None], xp.log(a10)[:, None], intrinsic[:, 3:9], xp.log(omega_reference)[:, None]],
        axis=1,
    )


def _pack(coefficients: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(N, 7, n)`` complex coefficients to the real ``zeta`` and ``G``
    blocks the two PCAs see, ``(N, 6 n)`` and ``(N, 7 n)``."""
    zeta = coefficients[:, :3]
    g = coefficients[:, 3:]
    zeta_real = np.concatenate([part for c in range(3) for part in (zeta[:, c].real, zeta[:, c].imag)], axis=1)
    g_real = np.concatenate(
        [g[:, 0].real] + [part for c in range(1, 4) for part in (g[:, c].real, g[:, c].imag)], axis=1
    )
    return zeta_real, g_real


def _unpack(xp, zeta_real, g_real, n: int):
    """Inverse of :func:`_pack`."""
    zeta = [zeta_real[:, 2 * c * n:(2 * c + 1) * n] + 1j * zeta_real[:, (2 * c + 1) * n:(2 * c + 2) * n]
            for c in range(3)]
    g = [g_real[:, :n] + 0j] + [
        g_real[:, (2 * c - 1) * n:2 * c * n] + 1j * g_real[:, 2 * c * n:(2 * c + 1) * n]
        for c in range(1, 4)
    ]
    return xp.stack(zeta + g, axis=1)


def _training_binary(grid: AngleGrid, row, omega_reference, x, state, derivative, stride):
    frame = reference_frame(np, grid, row, omega_reference)
    zeta, g = j_frame_angles(
        frame, x, state[:, 6:10], derivative[:, 6:10], (x.size - 1) // 2
    )
    rotation = np.asarray(frame.rotation)
    switch = fit_switch_spins(
        grid, frame, x, state[:, 0:3] @ rotation, state[:, 3:6] @ rotation
    )
    return x[::stride], zeta[::stride], g[::stride], switch


def training_data(
    grid: AngleGrid,
    intrinsic: np.ndarray,
    omega_reference: np.ndarray,
    n_steps: int = N_STEPS,
    batch: int = 64,
    stride: int = 2,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Integrate the precession of each binary over :attr:`AngleGrid.x_range`.

    Returns ``x``, :math:`\zeta` and :math:`G - G_{\rm ref}`
    (:func:`j_frame_angles`) on every ``stride``-th integration node,
    ``(N, 2 n_steps / stride + 1)`` each (the reference is the middle node),
    and the ``(N, N_SWITCH)`` targets of :func:`fit_switch_spins`.
    """
    from joblib import Parallel, delayed

    jax, jnp = _jnp()

    def one(row, omega):
        angles, state, derivative = integrate_angles(
            row[0], row[1], row[2], row[3:6], row[6:9], omega / np.pi,
            grid.omega_min / np.pi, n_steps, grid.omega_switch / np.pi,
            full_state=True,
        )
        return angles.x, state, derivative

    integrate = jax.jit(jax.vmap(one))
    results = []
    with Parallel(n_jobs=n_jobs) as parallel:
        for start in range(0, len(intrinsic), batch):
            rows = intrinsic[start:start + batch]
            omegas = omega_reference[start:start + batch]
            x, state, derivative = (
                np.asarray(a) for a in integrate(jnp.asarray(rows), jnp.asarray(omegas))
            )
            results += parallel(
                delayed(_training_binary)(
                    grid, rows[i], omegas[i], x[i], state[i], derivative[i], stride
                )
                for i in range(len(rows))
            )
            logging.info(
                "precession regression: integrated %i/%i", start + len(rows), len(intrinsic)
            )
    return tuple(np.array([r[k] for r in results]) for k in range(4))


def _fit_one(grid, row, omega_reference, x, zeta, g, prior):
    frame = reference_frame(np, grid, row, omega_reference)
    return fit_envelopes(grid, frame, x, zeta, g, prior)


def fit_all_envelopes(
    grid: AngleGrid, intrinsic, omega_reference, x, zeta, g, priors=None, n_jobs: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    r""":func:`fit_envelopes` for each binary of :func:`training_data`, in
    parallel: the ``(N, 7, n_coefficients)`` coefficients and the ``(N, 2)``
    largest residuals of the fits of :math:`\zeta` and :math:`G`."""
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one)(
            grid, intrinsic[i], omega_reference[i], x[i], zeta[i], g[i],
            None if priors is None else priors[i],
        )
        for i in range(len(intrinsic))
    )
    return np.array([r[0] for r in results]), np.array([r[1] for r in results])


def refine_envelopes(
    grid: AngleGrid,
    intrinsic, omega_reference, x, zeta, g, coefficients, switch,
    iterations: int = 3,
    folds: int = 4,
    n_jobs: int = -1,
    **train_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Make the envelopes predictable from the parameters, where the data
    leave them free.

    Where the two carriers nearly coincide, the split of :math:`\zeta`
    between their envelopes is not determined by one binary's angles: the
    least-squares fit picks one depending on the carriers' relative phase,
    which winds with the parameters. Each iteration
    refits every binary with :data:`PRIOR_PENALTY` pulling it towards the
    prediction of a :class:`PrecessionRegressor` trained on the other
    ``folds - 1`` folds, so that the undetermined part follows what the
    rest of the training set makes of it.
    """
    n = len(intrinsic)
    fold_of = np.arange(n) % folds
    residuals = None
    for iteration in range(iterations):
        priors = np.empty_like(coefficients)
        for fold in range(folds):
            train = fold_of != fold
            regressor = PrecessionRegressor.train(
                intrinsic[train], omega_reference[train], coefficients[train],
                switch[train], grid=grid, **train_kwargs,
            )
            priors[~train] = regressor.predict_coefficients(
                intrinsic[~train], omega_reference[~train]
            )
        coefficients, residuals = fit_all_envelopes(
            grid, intrinsic, omega_reference, x, zeta, g, priors, n_jobs
        )
        logging.info(
            "precession regression: refinement %i, largest fit residuals %.2g (zeta) %.2g (G)",
            iteration + 1, residuals[:, 0].max(), residuals[:, 1].max(),
        )
    return coefficients, residuals


def spin_projections(xp, intrinsic, omega_reference):
    r"""The spins' projections on :math:`\vec{J}` at the reference, ``(N,
    2)``: the regressor learns those at the switch relative to them."""
    q = intrinsic[:, 0]
    nu = q / (1.0 + q) ** 2
    mass_a = nu_to_X1(nu, xp)
    spin_a = intrinsic[:, 3:6] * mass_a[:, None] ** 2
    spin_b = intrinsic[:, 6:9] * (1.0 - mass_a)[:, None] ** 2
    l_reference = orbital_angular_momentum(nu, omega_reference)
    j = spin_a + spin_b + xp.stack([0.0 * q, 0.0 * q, l_reference], axis=1)
    j_hat = j / xp.sqrt(xp.sum(j * j, axis=1))[:, None]
    return xp.stack(
        [xp.sum(spin_a * j_hat, axis=1), xp.sum(spin_b * j_hat, axis=1)], axis=1
    )


def _switch_targets(xp, intrinsic, omega_reference, switch, inverse: bool):
    """The switch targets with the projections on J taken relative to those
    at the reference (or, ``inverse``, back)."""
    projections = spin_projections(xp, intrinsic, omega_reference)
    sign = 1.0 if inverse else -1.0
    return xp.concatenate([switch[:, :12], switch[:, 12:] + sign * projections], axis=1)


def _g_baselines(grid: AngleGrid, intrinsic, omega_reference) -> np.ndarray:
    """:func:`g_baseline` of each binary (numpy), ``(N, n_coefficients)``."""
    baselines = []
    for row, omega in zip(intrinsic, omega_reference):
        frame = reference_frame(np, grid, row, float(omega))
        _, rates = carrier_table(np, grid, frame)
        baselines.append(g_baseline(np, grid, frame, rates))
    return np.array(baselines)


@dataclass
class PrecessionRegressor:
    """Precession angles from the parameters, by PCA and kernel ridge on the
    envelopes of :func:`fit_envelopes` (and kernel ridge on the spins at the
    switch, :func:`fit_switch_spins`); see the module docstring.

    Build one with :meth:`train`; :meth:`angles` gives the
    :class:`RegressedAngles` of a binary in numpy, :meth:`jax_angles` a
    batched JAX function giving them, for
    :func:`~mlgw_bns.batched_precession.precessing_mode_components`.
    """

    grid: AngleGrid
    ranges: TrainingRanges
    pca_zeta: Any  # PrincipalComponentData
    pca_g: Any
    network: Any  # KernelRidgeNetwork

    @classmethod
    def train(
        cls,
        intrinsic: np.ndarray,
        omega_reference: np.ndarray,
        coefficients: np.ndarray,
        switch: np.ndarray,
        n_components: Tuple[int, int] = (64, 64),
        kernel_gamma: float = 0.02,
        grid: AngleGrid = AngleGrid(),
        ranges: TrainingRanges = TrainingRanges(),
    ) -> "PrecessionRegressor":
        """Fit the PCAs and the regressor to the envelope ``coefficients``
        of :func:`fit_all_envelopes` (or :func:`refine_envelopes`) and the
        ``switch`` targets of :func:`training_data`.

        Parameters
        ----------
        n_components : (int, int)
            Principal components kept of the :math:`\\zeta` and :math:`G`
            envelopes.
        kernel_gamma : float
            Width of the RBF kernel, on standardized features; the ridge
            penalties are chosen per output by leave-one-out.
        """
        from .neural_network import Hyperparameters, KernelRidgeNetwork
        from .principal_component_analysis import PrincipalComponentAnalysisModel

        coefficients = coefficients.copy()
        coefficients[:, 3] -= _g_baselines(grid, intrinsic, omega_reference)
        switch = _switch_targets(np, intrinsic, omega_reference, switch, inverse=False)
        zeta_real, g_real = _pack(coefficients)
        pca_zeta = PrincipalComponentAnalysisModel(n_components[0]).fit(zeta_real)
        pca_g = PrincipalComponentAnalysisModel(n_components[1]).fit(g_real)
        reduced = np.concatenate([
            PrincipalComponentAnalysisModel.reduce_data(zeta_real, pca_zeta),
            PrincipalComponentAnalysisModel.reduce_data(g_real, pca_g),
            switch,
        ], axis=1)
        network = KernelRidgeNetwork(
            Hyperparameters.default_kernel_ridge(len(intrinsic), kernel_gamma=kernel_gamma)
        )
        network.fit(regression_features(np, intrinsic, omega_reference), reduced)
        return cls(grid=grid, ranges=ranges, pca_zeta=pca_zeta, pca_g=pca_g, network=network)

    @property
    def _split(self) -> Tuple[int, int]:
        """Where the outputs of :attr:`network` change from the zeta to the G
        principal components, and from those to the switch targets."""
        k = self.pca_zeta.eigenvalues.size
        return k, k + self.pca_g.eigenvalues.size

    def predict(self, intrinsic: np.ndarray, omega_reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The ``(N, 7, n_coefficients)`` envelope coefficients and the
        ``(N, N_SWITCH)`` switch targets (numpy)."""
        from .principal_component_analysis import PrincipalComponentAnalysisModel

        reduced = self.network.predict(regression_features(np, intrinsic, omega_reference))
        k, kk = self._split
        coefficients = _unpack(
            np,
            PrincipalComponentAnalysisModel.reconstruct_data(reduced[:, :k], self.pca_zeta),
            PrincipalComponentAnalysisModel.reconstruct_data(reduced[:, k:kk], self.pca_g),
            self.grid.n_coefficients,
        )
        coefficients[:, 3] += _g_baselines(self.grid, intrinsic, omega_reference)
        return coefficients, _switch_targets(
            np, intrinsic, omega_reference, reduced[:, kk:], inverse=True
        )

    def predict_coefficients(self, intrinsic: np.ndarray, omega_reference: np.ndarray) -> np.ndarray:
        """The ``(N, 7, n_coefficients)`` envelope coefficients (numpy)."""
        return self.predict(intrinsic, omega_reference)[0]

    def angles(self, intrinsic, omega_reference) -> RegressedAngles:
        """The :class:`RegressedAngles` of one binary, in numpy (but for the
        integration above the switch, in JAX): ``intrinsic`` :math:`= [q,
        \\Lambda_1, \\Lambda_2, \\vec{\\chi}_1, \\vec{\\chi}_2]`, the spins given
        at the orbital frequency ``omega_reference``."""
        _, jnp = _jnp()
        intrinsic = np.asarray(intrinsic, dtype=float)
        coefficients, switch = self.predict(intrinsic[None], np.array([omega_reference]))
        frame = reference_frame(np, self.grid, intrinsic, float(omega_reference))
        phases, rates = carrier_table(np, self.grid, frame)
        tail = _tail(
            self.grid, frame, jnp.asarray(coefficients[0]), jnp.asarray(phases),
            jnp.asarray(rates), jnp.asarray(switch[0]),
        )
        return RegressedAngles(
            coefficients[0], phases, rates, np.asarray(frame.rotation),
            float(frame.g_reference), *(np.asarray(a) for a in tail), self.grid,
        )

    def jax_coefficients(self) -> Callable:
        """A JAX function ``(intrinsic, omega_reference) -> (coefficients,
        switch)``: the regressed ``(N, 7, n_coefficients)`` envelope
        coefficients, without the :func:`g_baseline` of :math:`g_0`, and the
        ``(N, N_SWITCH)`` switch targets."""
        from .batched import _freeze_kernel_ridge, _kernel_ridge

        _, jnp = _jnp()
        regressor = self.network.regressor
        kernel = _freeze_kernel_ridge(regressor.X_fit_, regressor.dual_coef_, regressor.gamma, [])
        kernel = type(kernel)(*(jnp.asarray(getattr(kernel, f.name)) if f.name != "gamma"
                                else kernel.gamma for f in fields(kernel)))
        param_mean = jnp.asarray(self.network.param_scaler.mean_)
        param_scale = jnp.asarray(self.network.param_scaler.scale_)
        target_mean = jnp.asarray(self.network.target_scaler.mean_)
        target_scale = jnp.asarray(self.network.target_scaler.scale_)
        k, kk = self._split
        blocks = []
        for pca in (self.pca_zeta, self.pca_g):
            # reduced -> data: (reduced * scaling) @ eigenvectors.T + mean
            blocks.append((
                jnp.asarray(pca.principal_components_scaling[:, None] * pca.eigenvectors.T),
                jnp.asarray(pca.mean),
            ))
        n = self.grid.n_coefficients

        def coefficients(intrinsic, omega_reference):
            features = regression_features(jnp, intrinsic, omega_reference)
            (reduced,) = _kernel_ridge(jnp, (features - param_mean) / param_scale, [kernel])
            reduced = reduced * target_scale + target_mean
            zeta_real = reduced[:, :k] @ blocks[0][0] + blocks[0][1]
            g_real = reduced[:, k:kk] @ blocks[1][0] + blocks[1][1]
            return _unpack(jnp, zeta_real, g_real, n), _switch_targets(
                jnp, intrinsic, omega_reference, reduced[:, kk:], inverse=True
            )

        return coefficients

    def jax_angles(self) -> Callable:
        """A JAX function giving the :class:`RegressedAngles` of a batch.

        ``angles(intrinsic, total_mass, reference_frequency_hz,
        start_frequency_hz=None)``, with the arguments of
        :func:`~mlgw_bns.batched_precession.precession_angles` (whose
        integration this replaces; the start frequency is not needed, the
        angles cover the whole of :attr:`grid`), returns a
        :class:`RegressedAngles` whose arrays have a first axis of length
        ``N``.
        """
        jax, jnp = _jnp()
        _register_pytree()
        coefficients = self.jax_coefficients()
        grid = self.grid
        nodes = tuple(jnp.asarray(a) for a in grid.quadrature_nodes())

        def one_binary(row, omega_reference, coefficients_row, switch_row):
            frame = reference_frame(jnp, grid, row, omega_reference)
            phases, rates = carrier_table(jnp, grid, frame, nodes)
            coefficients_row = coefficients_row.at[3].add(g_baseline(jnp, grid, frame, rates, nodes))
            tail = _tail(grid, frame, coefficients_row, phases, rates, switch_row)
            return RegressedAngles(
                coefficients_row, phases, rates, frame.rotation, frame.g_reference, *tail, grid
            )

        def angles(intrinsic, total_mass, reference_frequency_hz, start_frequency_hz=None):
            intrinsic = jnp.atleast_2d(jnp.asarray(intrinsic, jnp.float64))
            n_rows = intrinsic.shape[0]
            total_mass, reference_frequency_hz = (
                jnp.broadcast_to(jnp.asarray(value, jnp.float64), (n_rows,))
                for value in (total_mass, reference_frequency_hz)
            )
            omega_reference = np.pi * reference_frequency_hz * total_mass * SUN_MASS_SECONDS
            return jax.vmap(one_binary)(
                intrinsic, omega_reference, *coefficients(intrinsic, omega_reference)
            )

        return angles

    def save(self, filename: str) -> None:
        """Pickle the regressor with joblib."""
        import joblib

        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename: str) -> "PrecessionRegressor":
        import joblib

        return joblib.load(filename)
