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

Representation of the angles
----------------------------
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

The beat of the two carriers is the nutation frequency of the two spins,
in the closed form of the multiple-scale analysis (:func:`nutation`,
:func:`nutation_frequency`; ``docs/explanation/precession_regression.md``
derives it for these precession equations): the linearized normal modes get
it wrong by up to ~50% near equal masses, where the spin-spin coupling sets
it. :attr:`AngleGrid.elliptic_beat` switches back to them.

Where two carriers nearly coincide (:math:`q \simeq 1`) the split between
their envelopes is again not determined; :func:`refine_envelopes` pulls each
binary towards what a regressor trained on the others predicts for it.

Status
------
A prototype. Trained on 16384 binaries (uniform in :class:`TrainingRanges`;
~5 minutes to integrate and fit on four CPU cores, ~15 to train; larger
training sets are kept on disk by :mod:`~mlgw_bns.precession_dataset`), against
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
With the elliptic beat (trained and validated on :math:`q \leq 1.5` only,
4096 and 256 binaries) the errors of :math:`\zeta` and the 90th percentile of
the mismatches halve (median 4.9e-4, 90th percentile 7.9e-3, against 5.8e-4
and 1.6e-2 with the linear beat); those of :math:`G` do not change, and are
now the limit there.
The representation itself gives back the integrated rotation to ~1e-6 rad
below the switch and ~1e-4 rad above it. On four CPU cores the angles take
10 ms for one binary and 0.8 ms each in a batch of 128, against 155 ms and
29 ms for the integration (mostly the 48 sequential steps above the
switch); the whole precessing waveform 25 ms and 7.5 ms, against 155 ms and
34 ms.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, fields
from functools import lru_cache
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
    elliptic_beat : bool
        Whether the beat of the two carriers is the nonlinear nutation
        frequency (:func:`nutation_frequency`) or the splitting of the
        linearized normal modes (:func:`carrier_rates`).
    averaged_precession : bool
        With :attr:`elliptic_beat`, whether one carrier turns at the rate
        at which :math:`\\hat{L}` precesses about :math:`\\vec{J}` averaged
        over the nutation (:func:`mean_precession_rate`), the other a beat
        from it, instead of the two straddling the mean of the linear
        normal modes. The frequencies in :math:`\\zeta` are that rate plus
        multiples of the beat; near equal masses, where the linear beat is
        wrong, the straddling carriers miss it and the envelopes wind.
    smoothing : float
        Penalty on the second differences of the envelopes' B-spline
        coefficients (:func:`_penalized_least_squares`): what splits
        :math:`\\zeta` between the envelopes of the two carriers where the
        envelopes could follow the beat by themselves.
    sidebands : int
        Further harmonics of the nutation given their own envelopes
        (:attr:`zeta_terms`, :attr:`g_terms`), on each side of the two
        carriers; without them those harmonics, large where the nutation is
        anharmonic, are left to the envelopes, which then wind at the beat.
    """

    omega_min: float = OMEGA_MIN
    omega_max: float = OMEGA_MAX
    n_cells: int = N_CELLS
    n_quadrature: int = N_QUADRATURE
    x_switch: float = X_SWITCH
    n_tail_steps: int = N_TAIL_STEPS
    tail_kappa: float = TAIL_KAPPA
    elliptic_beat: bool = True
    sidebands: int = 0
    smoothing: float = SMOOTHING_PENALTY
    averaged_precession: bool = False

    def __setstate__(self, state):
        # grids pickled before the elliptic beat used the linear one
        state.setdefault("elliptic_beat", False)
        state.setdefault("sidebands", 0)
        state.setdefault("smoothing", SMOOTHING_PENALTY)
        state.setdefault("averaged_precession", False)
        self.__dict__.update(state)

    @property
    def zeta_terms(self) -> Tuple[Tuple[int, int], ...]:
        r"""The exponents :math:`(a, b)` of the carrier products :math:`e^{i
        (a \Phi_1 + b \Phi_2)}` that each envelope of :math:`\zeta` multiplies:
        the two carriers, a constant, and the :attr:`sidebands` harmonics of
        the nutation beyond them, :math:`\Phi_1 + k (\Phi_1 - \Phi_2)` and
        :math:`\Phi_2 - k (\Phi_1 - \Phi_2)`."""
        terms = [(0, 0), (1, 0), (0, 1)]
        for k in range(1, self.sidebands + 1):
            terms += [(1 + k, -k), (-k, 1 + k)]
        return tuple(terms)

    @property
    def g_terms(self) -> Tuple[Tuple[int, int], ...]:
        """The exponents of the carrier products of the complex envelopes of
        :math:`G`, after its real :math:`g_0`: the two carriers and the beat,
        with :attr:`sidebands` more harmonics of the beat."""
        return ((1, 0), (0, 1)) + tuple((k, -k) for k in range(1, self.sidebands + 2))

    @property
    def n_zeta(self) -> int:
        """Envelopes of :math:`\\zeta`; :math:`g_0` is the next."""
        return len(self.zeta_terms)

    @property
    def n_envelopes(self) -> int:
        return self.n_zeta + 1 + len(self.g_terms)

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
    omega_reference: Any


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
        omega_reference=omega_reference,
    )


#: Arithmetic-geometric-mean steps of the complete elliptic integrals:
#: quadratic convergence, to double precision for :math:`m < 1 - 10^{-12}`.
N_AGM = 8

#: Bisection steps for the turning points of the nutation, which are
#: bracketed in :math:`[-1, 1]` (in units of :math:`|S_A|`), and Newton
#: steps after them.
N_BISECTION = 20
N_NEWTON = 3

#: Picard iterations of the precession-averaged evolution of :math:`J`.
N_PICARD = 6

#: Orbital frequencies the nutation frequency is computed at, spread over
#: those of the carrier quadrature; it is smooth, and interpolated (in
#: :math:`\ln` of both) between them.
N_NUTATION_NODES = 64

# the cubic through its values at z = -1, -1/3, 1/3, 1: coefficients
# (c_0, c_1, c_2, c_3) = _CUBIC_FIT @ values
_CUBIC_NODES = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0])
_CUBIC_FIT = np.linalg.inv(np.vander(_CUBIC_NODES, 4, increasing=True))


def _elliptic_integrals(xp, m):
    r"""The complete elliptic integrals :math:`K(m)` and :math:`(K - E) / (m
    K)`, the time average of :math:`\mathrm{sn}^2` over its period, by the
    arithmetic-geometric mean, without cancellation at small :math:`m`."""
    a, b = 1.0, xp.sqrt(1.0 - m)
    # c_n^2 / m, from c_0^2 = m and c_{n+1} = c_n^2 / (4 a_{n+1})
    c2_over_m = xp.ones_like(m)
    average = 0.5 * c2_over_m
    for n in range(1, N_AGM + 1):
        a, b = (a + b) / 2.0, xp.sqrt(a * b)
        c2_over_m = c2_over_m * c2_over_m * m / (16.0 * a * a)
        average = average + 2.0 ** (n - 1) * c2_over_m
    return np.pi / (2.0 * a), average


def _bisect(xp, function, derivative, lo, hi):
    """The root of ``function`` in ``[lo, hi]``, where it changes sign (or the
    end nearer to one, where it does not): :data:`N_BISECTION` bisections,
    then :data:`N_NEWTON` Newton steps kept in the final bracket."""
    f_lo = function(lo)
    for _ in range(N_BISECTION):
        middle = (lo + hi) / 2.0
        f_middle = function(middle)
        same = xp.sign(f_middle) == xp.sign(f_lo)
        lo, f_lo = xp.where(same, middle, lo), xp.where(same, f_middle, f_lo)
        hi = xp.where(same, hi, middle)
    root = (lo + hi) / 2.0
    for _ in range(N_NEWTON):
        slope = derivative(root)
        step = function(root) / xp.where(slope == 0.0, 1.0, slope)
        root = xp.clip(root - xp.where(slope == 0.0, 0.0, step), lo, hi)
    return root


def frozen_precession_constants(xp, nu, q, omega):
    r"""The coefficients of the conservative spin precession at a fixed
    orbital frequency ``omega``: :math:`(L, \alpha_A, \alpha_B, k)`.

    With :math:`Y = (\vec{S}_A / q + \vec{S}_B) \cdot \hat{L}`, the right-hand
    side of :func:`~mlgw_bns.twist_waveform._pn_precession_derivatives` is
    :math:`\dot{\vec{S}}_A = (a_A \hat{L} + h \vec{S}_B) \times \vec{S}_A`,
    :math:`a_A = w_A - 3 h Y`, and :math:`\dot{\vec{S}}_B = (a_B \hat{L} + h
    \vec{S}_A) \times \vec{S}_B`, :math:`a_B = w_B - 3 h q Y`, with :math:`h
    = v^6 / 2` and :math:`w_{A, B}` the spin-orbit frequencies, and to
    leading order :math:`L \dot{\hat{L}} = - \dot{\vec{S}}_A -
    \dot{\vec{S}}_B`. Then :math:`\alpha_{A, B} = w_{A, B} / L - h` and
    :math:`k = 3 h q / (2 L)`; see :func:`nutation`.

    The next order of :math:`\dot{\hat{L}}` (Eq. (4c) of arXiv:2005.05338)
    multiplies :math:`\dot{\vec{S}}_X` by :math:`1 + \nu v^2 c_X`, with
    :math:`c_X = -(3 + 1 / M_X) / 4`: for equal :math:`c_X` this is the
    leading order with :math:`L / (1 + \nu v^2 c)` in place of :math:`L`,
    which is what is returned as :math:`L`, at the mean of the two. This
    takes the error of the nutation frequency from ~1% to ~0.1% near
    :math:`q = 1`, where the two are equal; the remaining terms are
    quadratic in the spins. :math:`\vec{J}` is then :math:`L \hat{L} +
    \vec{S}_A + \vec{S}_B` with this :math:`L`.
    """
    v = omega ** (1.0 / 3.0)
    v5_ca, v5_cb, v7_ca, v7_cb, v9_ca, v9_cb = _spin_orbit_coefficients(nu, q, xp)
    w_a = v**5 * v5_ca + v**7 * v7_ca + v**9 * v9_ca
    w_b = v**5 * v5_cb + v**7 * v7_cb + v**9 * v9_cb
    h = 0.5 * v**6
    mass_a = nu_to_X1(nu, xp)
    mean_c = -(6.0 + 1.0 / mass_a + 1.0 / (1.0 - mass_a)) / 8.0
    l_magnitude = orbital_angular_momentum(nu, omega) / (1.0 + nu * v**2 * mean_c)
    return l_magnitude, w_a / l_magnitude - h, w_b / l_magnitude - h, 1.5 * h * q / l_magnitude


def nutation(xp, q, sa2, sb2, l_magnitude, alpha_a, alpha_b, k, f, j2, y_star, return_roots=False):
    r"""The nutation of the two spins at a fixed orbital frequency: its
    angular frequency (in time), the time average of :math:`u_A = \vec{S}_A
    \cdot \hat{L}` over it, and the elliptic parameter.

    The conservative precession (:func:`frozen_precession_constants`)
    conserves :math:`\vec{J} = L \hat{L} + \vec{S}_A + \vec{S}_B`,
    :math:`|\vec{S}_A|`, :math:`|\vec{S}_B|` and

    .. math:: F = \alpha_A u_A + \alpha_B u_B - k Y^2 ,

    (:math:`u_X = \vec{S}_X \cdot \hat{L}`; at leading PN order
    :math:`\alpha_A = \alpha_B / q` and this is the effective spin), which
    leave one degree of freedom, :math:`u_A`: with :math:`T = \hat{L} \cdot
    (\vec{S}_A \times \vec{S}_B)`, :math:`\dot{u}_A = (\alpha_B - 2 k Y) T`,
    and :math:`T^2` is the Gram determinant of :math:`\hat{L}, \vec{S}_A,
    \vec{S}_B`, a cubic in :math:`u_A` once :math:`u_B` and :math:`\vec{S}_A
    \cdot \vec{S}_B` are written with the constants (:math:`Y` linearized about
    ``y_star``, which it differs from by :math:`O(\alpha_A - \alpha_B / q)`).
    :math:`u_A` oscillates between the two roots :math:`u_\pm` of the cubic
    that bracket its maximum as :math:`u_- + (u_+ - u_-)\, \mathrm{sn}^2`,
    with period :math:`4 K(m) |S_A| / (|\tilde\alpha| \sqrt{g(u_-)})`, where
    :math:`g` is the cubic's third, linear factor and :math:`m = 1 -
    g(u_+) / g(u_-)`: the closed form of Kesden et al. (2015) and
    Chatziioannou et al. (2017), regular at :math:`q = 1` as in Gerosa et al.
    (2023), where :math:`g` is constant.

    All arguments broadcast together. ``sa2``, ``sb2`` are :math:`|\vec{S}_{A,
    B}|^2`, ``j2`` :math:`J^2`.
    """
    alpha_tilde = alpha_b - 2.0 * k * y_star
    # Y = y_0 + y_1 u_A, linearized about y_star
    y_0 = y_star - (alpha_b * y_star - k * y_star**2 - f) / alpha_tilde
    y_1 = -(alpha_a - alpha_b / q) / alpha_tilde
    sa = xp.sqrt(sa2)
    s_free = (j2 - l_magnitude**2 - sa2 - sb2) / 2.0

    def gram(z):
        u_a = sa * z
        u_b = y_0 + (y_1 - 1.0 / q) * u_a
        s_ab = s_free - l_magnitude * (u_a + u_b)
        return sa2 * sb2 - s_ab**2 - u_a**2 * sb2 + 2.0 * u_a * u_b * s_ab - u_b**2 * sa2

    values = xp.stack([gram(z) for z in _CUBIC_NODES], axis=-1)
    c_0, c_1, c_2, c_3 = (xp.sum(values * _CUBIC_FIT[i], axis=-1) for i in range(4))

    def cubic(z):
        return ((c_3 * z + c_2) * z + c_1) * z + c_0

    def slope(z):
        return (3.0 * c_3 * z + 2.0 * c_2) * z + c_1

    # the maximum, between the two physical roots: the stationary point with
    # negative curvature, written without cancellation as c_3 -> 0 (q -> 1)
    discriminant = xp.maximum(c_2**2 - 3.0 * c_3 * c_1, 0.0)
    z_max = xp.clip(c_1 / (xp.sqrt(discriminant) - c_2), -1.0, 1.0)
    # the Gram determinant is -(...)^2 <= 0 at u_A = +-|S_A|
    z_lo = _bisect(xp, cubic, slope, -xp.ones_like(z_max), z_max)
    z_hi = _bisect(xp, cubic, slope, z_max, xp.ones_like(z_max))
    # no room to nutate (a resonance, or aligned spins): both roots at the maximum
    still = cubic(z_max) <= 0.0
    z_lo, z_hi = xp.where(still, z_max, z_lo), xp.where(still, z_max, z_hi)
    # cubic = (z - z_lo)(z_hi - z) g(z), g linear
    g_lo = -c_3 * (2.0 * z_lo + z_hi) - c_2
    g_hi = -c_3 * (z_lo + 2.0 * z_hi) - c_2
    m = xp.clip(1.0 - g_hi / g_lo, 0.0, 1.0 - 1e-12)
    quarter_period, mean_sn2 = _elliptic_integrals(xp, m)
    # u_A = |S_A| z: the period is 4 K(m) |S_A| / (|alpha~| sqrt(g(z_lo)))
    frequency = (
        np.pi * xp.abs(alpha_tilde) * xp.sqrt(xp.maximum(g_lo, 0.0)) / (2.0 * quarter_period * sa)
    )
    mean_u_a = sa * (z_lo + (z_hi - z_lo) * mean_sn2)
    if return_roots:
        return frequency, mean_u_a, m, z_lo, z_hi
    return frequency, mean_u_a, m


def _tanh_sinh(half_width: float = 3.0, step: float = 1.0 / 32.0) -> Tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of the tanh-sinh quadrature on ``(0, 1)``, which
    clusters its nodes double-exponentially at the two ends."""
    t = np.arange(-half_width, half_width + step / 2, step)
    inner = 0.5 * np.pi * np.sinh(t)
    nodes = 0.5 * (1.0 + np.tanh(inner))
    weights = step * 0.25 * np.pi * np.cosh(t) / np.cosh(inner) ** 2
    return nodes, weights


#: The quadrature over half a nutation of :func:`mean_precession_rate`, in
#: :math:`2 \theta / \pi`: the peak of :math:`\Omega_z` where :math:`\hat{L}`
#: passes close to :math:`\vec{J}` is at a turning point of the nutation,
#: an end of the interval, and is resolved down to closest approaches of
#: ~1e-8 rad.
NUTATION_NODES, NUTATION_WEIGHTS = _tanh_sinh()


def mean_precession_rate(xp, q, sa2, sb2, l_magnitude, alpha_a, alpha_b, k, f, j2, y_star, z_lo, z_hi, m):
    r"""The rate :math:`\langle\Omega_z\rangle` at which :math:`\hat{L}`
    precesses about :math:`\vec{J}`, averaged over the nutation of
    :func:`nutation` (same arguments, and the turning points ``z_lo``,
    ``z_hi`` of :math:`u_A / |S_A|` and the elliptic parameter ``m`` it
    gives).

    With :math:`L \dot{\hat{L}} = -\hat{L} \times (a_A \vec{S}_A + a_B
    \vec{S}_B)` (the spin-spin terms cancel), the precession rate about
    :math:`\vec{J}` is

    .. math::
        \Omega_z = \frac{\hat{J} \cdot (\hat{L} \times \dot{\hat{L}})}{\sin^2\beta_J}
        = \frac{J N}{L (J^2 - c^2)} , \quad
        N = a_A (S_A^2 + \vec{S}_A \cdot \vec{S}_B - u_A c_S)
          + a_B (S_B^2 + \vec{S}_A \cdot \vec{S}_B - u_B c_S) ,

    with :math:`c_S = u_A + u_B`, :math:`c = L + c_S = \vec{J} \cdot
    \hat{L}`: a function of :math:`u_A` alone (for one spin, :math:`a_A J /
    L`, exactly). Over the nutation :math:`u_A = u_- + (u_+ - u_-)\,
    \mathrm{sn}^2(\psi, m)` with :math:`\psi` uniform in time, and with
    :math:`\mathrm{sn}\,\psi = \sin\theta` the time average is the
    average over :math:`0 \leq \theta \leq \pi / 2` with weight :math:`(1 -
    m \sin^2\theta)^{-1/2}`, by the tanh-sinh quadrature of
    :data:`NUTATION_NODES`: :math:`\Omega_z` peaks sharply where
    :math:`\hat{L}` passes close to :math:`\vec{J}`, which it does at a
    turning point, and the quadrature clusters its nodes there. NaN where
    there is no nutation to average over (the turning points coincide: the
    linearized cubic has no room between its roots, and its maximum may not
    be a configuration that exists) or where it is not one.
    """
    alpha_tilde = alpha_b - 2.0 * k * y_star
    y_0 = y_star - (alpha_b * y_star - k * y_star**2 - f) / alpha_tilde
    y_1 = -(alpha_a - alpha_b / q) / alpha_tilde
    h = 2.0 * k * l_magnitude / (3.0 * q)
    w_a, w_b = (alpha_a + h) * l_magnitude, (alpha_b + h) * l_magnitude
    s_free = (j2 - l_magnitude**2 - sa2 - sb2) / 2.0
    expand = lambda a: xp.asarray(a)[..., None]  # noqa: E731
    # sin^2 and cos^2 of theta, each accurate where it is small
    sin2 = np.sin(0.5 * np.pi * NUTATION_NODES) ** 2
    cos2 = np.sin(0.5 * np.pi * (1.0 - NUTATION_NODES)) ** 2
    near_hi = NUTATION_NODES > 0.5
    z = xp.where(near_hi, expand(z_hi) - expand(z_hi - z_lo) * cos2, expand(z_lo) + expand(z_hi - z_lo) * sin2)
    u_a = xp.sqrt(sa2) * z
    y = expand(y_0) + expand(y_1) * u_a
    u_b = y - u_a / q
    projection = u_a + u_b
    s_ab = expand(s_free) - expand(l_magnitude) * projection
    a_a = expand(w_a) - 3.0 * expand(h) * y
    a_b = expand(w_b) - 3.0 * expand(h) * q * y
    numerator = a_a * (sa2 + s_ab - u_a * projection) + a_b * (sb2 + s_ab - u_b * projection)
    j_l = expand(l_magnitude) + projection
    tilt = expand(j2) - j_l**2
    rate = xp.sqrt(expand(j2)) * numerator / (expand(l_magnitude) * xp.where(tilt > 0, tilt, 1.0))
    weight = NUTATION_WEIGHTS / xp.sqrt(1.0 - expand(m) + expand(m) * cos2)
    mean = xp.sum(weight * rate, axis=-1) / xp.sum(weight, axis=-1)
    exists = xp.all(tilt > 0, axis=-1) & (z_hi > z_lo)
    return xp.where(exists, mean, xp.nan)


def nutation_frequency(
    xp, frame: ReferenceFrame, omega, return_parameter: bool = False, return_precession: bool = False
):
    r"""The angular frequency, in time, of the nutation of the two spins (the
    beat of the two carriers) at the orbital frequencies ``omega``, an
    increasing array.

    :func:`nutation` at each ``omega``, with the constants carried from the
    reference along the inspiral: :math:`|\vec{S}_{A, B}|`; :math:`Y`, whose
    rate :math:`-(\alpha_A - \alpha_B / q) T` averages to zero over a
    nutation; :math:`F` at the reference values of :math:`Y` and :math:`u_A`;
    and :math:`J`, which radiation reaction changes along :math:`\hat{L}`
    only, :math:`\mathrm{d} J^2 / \mathrm{d} L = 2 L + 2 \langle u_A + u_B
    \rangle`, averaged over the nutation (Gerosa et al. 2015), and
    integrated in :math:`L` from its reference value by :data:`N_PICARD`
    Picard iterations, ``omega`` serving as the quadrature nodes. With
    ``return_parameter``, also the elliptic parameter :math:`m` there: how
    anharmonic the nutation is, close to one near a separatrix between
    precession morphologies; with ``return_precession``, also the
    precession-averaged rate of :func:`mean_precession_rate`, in time.
    """
    nu, q = frame.nu, frame.mass_ratio
    spin_a, spin_b = frame.spin_a, frame.spin_b
    sa2, sb2 = xp.sum(spin_a * spin_a), xp.sum(spin_b * spin_b)
    u_a_ref, u_b_ref = spin_a[2], spin_b[2]
    y_ref = u_a_ref / q + u_b_ref

    l_ref, *_ = frozen_precession_constants(xp, nu, q, frame.omega_reference)
    total = spin_a + spin_b
    j2_ref = total[0] ** 2 + total[1] ** 2 + (l_ref + total[2]) ** 2

    l_magnitude, alpha_a, alpha_b, k = frozen_precession_constants(xp, nu, q, omega)
    f = alpha_a * u_a_ref + alpha_b * u_b_ref - k * y_ref**2

    # L decreases along omega: integrate in -L, increasing
    minus_l = -l_magnitude

    def j2_of(projection_sum):
        # J^2 - L^2 = J_ref^2 - L_ref^2 + 2 int_{L_ref}^{L} <u_A + u_B> dL
        cumulative = xp.concatenate([
            xp.zeros(1), xp.cumsum(0.5 * (projection_sum[1:] + projection_sum[:-1]) * xp.diff(minus_l))
        ])
        at_reference = xp.interp(-l_ref, minus_l, cumulative)
        return j2_ref + l_magnitude**2 - l_ref**2 - 2.0 * (cumulative - at_reference)

    projection_sum = xp.full_like(omega, u_a_ref + u_b_ref)
    for _ in range(N_PICARD):
        j2 = j2_of(projection_sum)
        frequency, mean_u_a, parameter, z_lo, z_hi = nutation(
            xp, q, sa2, sb2, l_magnitude, alpha_a, alpha_b, k, f, j2, y_ref, return_roots=True
        )
        # <u_B> = <Y> - <u_A> / q, with <Y> on the same linearization
        alpha_tilde = alpha_b - 2.0 * k * y_ref
        mean_y = y_ref - (alpha_b * y_ref - k * y_ref**2 - f + (alpha_a - alpha_b / q) * mean_u_a) / alpha_tilde
        projection_sum = mean_u_a * (1.0 - 1.0 / q) + mean_y
    result = (frequency,)
    if return_parameter:
        result += (parameter,)
    if return_precession:
        result += (mean_precession_rate(
            xp, q, sa2, sb2, l_magnitude, alpha_a, alpha_b, k, f, j2, y_ref, z_lo, z_hi, parameter
        ),)
    return result if len(result) > 1 else frequency


def carrier_rates(
    xp, frame: ReferenceFrame, omega, domega_dx, elliptic_beat: bool = True,
    averaged_precession: bool = False,
):
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

    With ``elliptic_beat`` the difference of the two rates, the beat, is
    instead the nutation frequency of the nonlinear precession
    (:func:`nutation_frequency`), about the same mean. The linearization
    gets it wrong near equal masses, where the beat is set by the spin-spin
    coupling and depends on the relative orientation of the in-plane spins:
    by up to ~50% at :math:`q = 1` (see
    ``docs/explanation/precession_regression.md``). With
    ``averaged_precession`` the carriers are placed not about the linear
    mean but on the precession-averaged rate of :math:`\hat{L}` (see
    :attr:`AngleGrid.averaged_precession`).
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
    if elliptic_beat:
        # the beat from the nonlinear nutation, about the same mean
        middle = (fast_shifted + slow_shifted) / 2.0
        # computed at a subset of the frequencies, and interpolated linearly
        # in ln omega (the cells are fixed by the shape: gathers, no search)
        n = omega.shape[-1]
        index = np.unique(np.linspace(0, n - 1, N_NUTATION_NODES).round().astype(int))
        cell = np.clip(np.searchsorted(index, np.arange(n), side="right") - 1, 0, len(index) - 2)
        log_omega = xp.log(omega)
        log_lo, log_hi = log_omega[index[cell]], log_omega[index[cell + 1]]
        # without in-plane spins (to 1e-6 of J) there is no precession, the
        # roots of the closed form coincide and it is ill-conditioned, and the
        # carriers multiply envelopes of zero: the linear splitting will do
        if averaged_precession:
            elliptic, precession = nutation_frequency(xp, frame, omega[index], return_precession=True)
        else:
            elliptic = nutation_frequency(xp, frame, omega[index])
        linear = (fast_shifted - slow_shifted)[index]
        usable = xp.isfinite(elliptic) & (elliptic > 0.0) & ((tilt_a + tilt_b)[index] > 1e-12)
        log_beat = xp.log(xp.where(usable, elliptic, linear))
        weight = (log_omega - log_lo) / (log_hi - log_lo)
        half_beat = xp.exp(log_beat[cell] + weight * (log_beat[cell + 1] - log_beat[cell])) / 2.0
        if averaged_precession:
            # one carrier on the averaged precession, the other a beat away,
            # on the side of the linear pair (chosen at the reference)
            precession = xp.where(usable & xp.isfinite(precession), precession, middle[index])
            precession = precession[cell] + weight * (precession[cell + 1] - precession[cell])
            at_reference = xp.argmin(xp.abs(log_omega - xp.log(frame.omega_reference)))
            on_fast = (
                xp.abs(precession - fast_shifted)[at_reference]
                <= xp.abs(precession - slow_shifted)[at_reference]
            )
            fast_shifted, slow_shifted = (
                xp.where(on_fast, precession, precession + 2.0 * half_beat),
                xp.where(on_fast, precession - 2.0 * half_beat, precession),
            )
        else:
            fast_shifted, slow_shifted = middle + half_beat, middle - half_beat
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
    rates = carrier_rates(
        xp, frame, xp.asarray(omega), xp.asarray(domega_dx), grid.elliptic_beat,
        grid.averaged_precession,
    )
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


#: The envelopes without :attr:`AngleGrid.sidebands`, in the order of the
#: coefficient arrays: three of zeta, four of G (``g_0`` is real).
ENVELOPES = ("c_0", "c_1", "c_2", "g_0", "g_1", "g_2", "g_12")


def _term(xp, e_1, e_2, exponents):
    r""":math:`e_1^a e_2^b` for ``exponents`` :math:`= (a, b)`, the carriers
    being of unit modulus."""
    result = 1.0
    for carrier, power in zip((e_1, e_2), exponents):
        if power:
            result = result * (carrier if power > 0 else xp.conj(carrier)) ** abs(power)
    return result


def _evaluate(xp, grid: AngleGrid, coefficients, carriers, x):
    r"""``(zeta, G - G_ref)`` at ``x`` from the ``(n_envelopes,
    n_coefficients)`` complex envelope coefficients and the carriers there."""
    cell, weights = _bspline(xp, grid, x)
    envelopes = sum(coefficients[:, cell + r] * weights[..., r] for r in range(4))
    e_1, e_2 = carriers
    n = grid.n_zeta
    zeta = sum(envelopes[i] * _term(xp, e_1, e_2, t) for i, t in enumerate(grid.zeta_terms))
    g = envelopes[n].real + sum(
        envelopes[n + 1 + i] * _term(xp, e_1, e_2, t) for i, t in enumerate(grid.g_terms)
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

    coefficients: Any  # (n_envelopes, n_coefficients), complex
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


def j_frame_angles(rotation, x, state, derivative, reference_index):
    r""":math:`\zeta` and :math:`G - G_{\rm ref}` in the frame of :math:`\vec{J}`
    (reached by the ``rotation`` :math:`R_0` of :class:`ReferenceFrame`), from
    the integrated :math:`\hat{L}` (the ``state`` and ``derivative`` of
    :class:`~mlgw_bns.batched_precession.TabulatedAngles`, numpy).

    :math:`G` is the trapezoidal quadrature of :math:`\mathrm{Im}(\zeta^*
    \zeta') / (1 + \cos\beta_J)` on the integration nodes.
    """
    rotation = np.asarray(rotation)
    l_j = state[:, :3] @ rotation
    dl_j = derivative[:, :3] @ rotation
    zeta = l_j[:, 0] + 1j * l_j[:, 1]
    dg = (l_j[:, 0] * dl_j[:, 1] - l_j[:, 1] * dl_j[:, 0]) / (1.0 + l_j[:, 2])
    g = np.concatenate([[0.0], np.cumsum(0.5 * np.diff(x) * (dg[1:] + dg[:-1]))])
    return zeta, g - g[reference_index]


def _penalized_least_squares(
    cell, weights, factors, data, n_coefficients, prior=None, smoothing=SMOOTHING_PENALTY
):
    """Coefficients of ``sum_b factors[:, b] * spline_b(x)`` fitting ``data``.

    ``factors`` ``(n, n_blocks)`` multiply one B-spline envelope each; complex
    factors and data give complex coefficients. The normal equations are
    assembled from the four nonzero B-splines of each point, with
    ``smoothing`` and :data:`RIDGE_PENALTY`; with a ``prior``
    ``(n_blocks, n_coefficients)``, the ridge pulls towards it, with
    :data:`PRIOR_PENALTY`, instead of towards zero. ``data`` ``(n, k)``
    fits ``k`` right-hand sides at once, giving ``(k, n_blocks,
    n_coefficients)``.

    With the coefficients of the blocks interleaved (coefficient ``k`` of
    block ``b`` at ``k n_blocks + b``) the normal matrix is banded, with
    ``4 n_blocks - 1`` superdiagonals, and Hermitian positive definite: it
    is assembled cell by cell (a batched matrix product over the points of
    each cell) in banded storage and solved by a banded Cholesky
    decomposition, some hundred times fewer operations than the dense solve.
    """
    from scipy.linalg import solveh_banded

    n_points, n_blocks = factors.shape
    size = n_blocks * n_coefficients
    width = 4 * n_blocks
    upper = width - 1
    columns_of_data = data.reshape(n_points, -1)
    # (n, 4 n_blocks) values of the design matrix, local column r n_blocks + b
    values = (weights[:, :, None] * factors[:, None, :]).reshape(n_points, width)
    order = np.argsort(cell, kind="stable")
    cell, values, columns_of_data = cell[order], values[order], columns_of_data[order]
    starts = np.flatnonzero(np.diff(cell, prepend=-1))
    cells = cell[starts]
    counts = np.diff(np.append(starts, n_points))
    # the points of each cell, zero-padded to the most any cell has
    which = (np.repeat(np.arange(len(starts)), counts), np.arange(n_points) - np.repeat(starts, counts))
    padded = np.zeros((len(starts), counts.max(), width), values.dtype)
    padded[which] = values
    padded_data = np.zeros((len(starts), counts.max(), columns_of_data.shape[1]), columns_of_data.dtype)
    padded_data[which] = columns_of_data
    adjoint = np.conj(np.swapaxes(padded, 1, 2))
    gram = adjoint @ padded
    rhs_cells = adjoint @ padded_data
    rows, columns = np.triu_indices(width)
    # entry (i, j), i <= j, of the upper band is at ab[upper + i - j, j]
    band_index = (
        (upper + rows - columns)[None, :] * size + cells[:, None] * n_blocks + columns[None, :]
    ).ravel()
    complex_data = np.iscomplexobj(values) or np.iscomplexobj(data)

    def accumulate(index, weight, length):
        total = np.bincount(index, weights=weight.real, minlength=length)
        if complex_data:
            total = total + 1j * np.bincount(index, weights=weight.imag, minlength=length)
        return total

    band = accumulate(band_index, gram[:, rows, columns].ravel(), (upper + 1) * size)
    band = band.reshape(upper + 1, size)
    rhs_index = (cells[:, None] * n_blocks + np.arange(width)[None, :]).ravel()
    rhs = np.stack(
        [accumulate(rhs_index, rhs_cells[:, :, j].ravel(), size) for j in range(rhs_cells.shape[2])],
        axis=1,
    )
    scale = np.sum(band[upper].real) / size
    # smoothing D^T D + ridge, pentadiagonal, the same for each block
    second_difference = np.diff(np.eye(n_coefficients), 2, axis=0)
    penalty = smoothing * second_difference.T @ second_difference + RIDGE_PENALTY * np.eye(n_coefficients)
    if prior is not None:
        penalty += PRIOR_PENALTY * np.eye(n_coefficients)
        rhs = rhs + scale * PRIOR_PENALTY * np.ravel(np.swapaxes(prior, 0, 1))[:, None]
    for offset in range(3):
        diagonal = np.diagonal(penalty, offset)
        band[upper - offset * n_blocks, offset * n_blocks:] += scale * np.repeat(diagonal, n_blocks)
    solution = solveh_banded(band, rhs, lower=False, check_finite=False)
    solution = np.moveaxis(solution.reshape(n_coefficients, n_blocks, -1), 2, 0).swapaxes(1, 2)
    return solution if data.ndim > 1 else solution[0]


def fit_envelopes(grid: AngleGrid, frame: ReferenceFrame, x, zeta, g, prior=None, table=None):
    """The ``(n_envelopes, n_coefficients)`` envelope coefficients fitting
    ``zeta`` and ``g`` at the ``x`` within :attr:`AngleGrid.x_range` (numpy),
    and the largest residuals of the two fits; ``prior``, see
    :func:`refine_envelopes`. ``table``, the :func:`carrier_table` of the
    binary if already known (then ``frame`` is not needed)."""
    inside = x <= grid.x_range[1]
    x, zeta, g = x[inside], zeta[inside], g[inside]
    phases, rates = carrier_table(np, grid, frame) if table is None else table
    e_1, e_2 = _carriers(np, grid, phases, rates, x)
    cell, weights = _bspline(np, grid, x)
    n = grid.n_coefficients
    k = grid.n_zeta
    zeta_factors = np.stack(
        [_term(np, e_1, e_2, t) * np.ones_like(e_1) for t in grid.zeta_terms], axis=1
    )
    zeta_fit = _penalized_least_squares(
        cell, weights, zeta_factors, zeta, n, None if prior is None else prior[:k], grid.smoothing,
    )
    g_factors = np.stack(
        [np.ones_like(x)]
        + [part for t in grid.g_terms for part in (_term(np, e_1, e_2, t).real, -_term(np, e_1, e_2, t).imag)],
        axis=1,
    )
    g_prior = None if prior is None else np.stack(
        [prior[k].real] + [part for c in prior[k + 1:] for part in (c.real, c.imag)]
    )
    g_fit = _penalized_least_squares(cell, weights, g_factors, g, n, g_prior, grid.smoothing)
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


def fit_switch_spins(grid: AngleGrid, frame: ReferenceFrame, x, spin_a, spin_b, table=None) -> np.ndarray:
    r"""The :data:`N_SWITCH` regression targets of the spins at
    :attr:`AngleGrid.x_switch`, from their components in the frame of
    :math:`\vec{J}` at ``x``, ``(n, 3)`` each (numpy); ``table`` as for
    :func:`fit_envelopes`.

    The in-plane components are fitted on the carriers as :math:`\zeta` is
    (:func:`fit_envelopes`): they wind with the precession, their envelopes
    do not.
    """
    inside = x <= grid.x_range[1]
    phases, rates = carrier_table(np, grid, frame) if table is None else table
    e_1, e_2 = _carriers(np, grid, phases, rates, x[inside])
    cell, weights = _bspline(np, grid, x[inside])
    at_switch_cell, at_switch_weights = _bspline(np, grid, np.array([grid.x_switch]))
    factors = np.stack([np.ones_like(e_1), e_1, e_2], axis=1)
    in_plane = np.stack([spin[inside, 0] + 1j * spin[inside, 1] for spin in (spin_a, spin_b)], axis=1)
    targets = []
    for envelopes in _penalized_least_squares(cell, weights, factors, in_plane, grid.n_coefficients):
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


def _pack(coefficients: np.ndarray, n_zeta: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(N, n_envelopes, n)`` complex coefficients to the real ``zeta`` and
    ``G`` blocks the two PCAs see, ``(N, 2 n_zeta n)`` and ``(N, (2
    n_envelopes - 2 n_zeta - 1) n)``."""
    zeta = coefficients[:, :n_zeta]
    g = coefficients[:, n_zeta:]
    zeta_real = np.concatenate(
        [part for c in range(n_zeta) for part in (zeta[:, c].real, zeta[:, c].imag)], axis=1
    )
    g_real = np.concatenate(
        [g[:, 0].real] + [part for c in range(1, g.shape[1]) for part in (g[:, c].real, g[:, c].imag)],
        axis=1,
    )
    return zeta_real, g_real


def _unpack(xp, zeta_real, g_real, n: int):
    """Inverse of :func:`_pack`."""
    n_zeta = zeta_real.shape[1] // (2 * n)
    n_g = (g_real.shape[1] // n + 1) // 2
    zeta = [zeta_real[:, 2 * c * n:(2 * c + 1) * n] + 1j * zeta_real[:, (2 * c + 1) * n:(2 * c + 2) * n]
            for c in range(n_zeta)]
    g = [g_real[:, :n] + 0j] + [
        g_real[:, (2 * c - 1) * n:2 * c * n] + 1j * g_real[:, 2 * c * n:(2 * c + 1) * n]
        for c in range(1, n_g)
    ]
    return xp.stack(zeta + g, axis=1)


def _batched_call(function, size: int, *arrays, keep=None):
    """``function`` of the rows of ``arrays``, in batches of ``size`` (the
    last padded with copies of its first row, so that a jitted function
    compiles once), its outputs as numpy arrays concatenated along the first
    axis; only the outputs of index in ``keep``, if given."""
    parts = []
    for start in range(0, len(arrays[0]), size):
        batch = [np.asarray(a[start:start + size]) for a in arrays]
        rows = len(batch[0])
        if rows < size:
            batch = [np.concatenate([b, np.repeat(b[:1], size - rows, axis=0)]) for b in batch]
        outputs = function(*batch)
        parts.append([np.asarray(outputs[k])[:rows] for k in (range(len(outputs)) if keep is None else keep)])
    return [np.concatenate(columns) for columns in zip(*parts)]


def _batch_size(n: int, largest: int) -> int:
    """The batch for ``n`` rows: ``largest``, or the power of two above ``n``
    if smaller (one compilation per power of two)."""
    return min(largest, 1 << max(n - 1, 0).bit_length())


@lru_cache(maxsize=None)
def _jitted_tables(grid: AngleGrid) -> Callable:
    jax, jnp = _jnp()
    nodes = tuple(jnp.asarray(a) for a in grid.quadrature_nodes())

    def one(row, omega_reference):
        frame = reference_frame(jnp, grid, row, omega_reference)
        phases, rates = carrier_table(jnp, grid, frame, nodes)
        return frame.rotation, phases, rates, g_baseline(jnp, grid, frame, rates, nodes)

    return jax.jit(jax.vmap(one))


def carrier_tables(grid: AngleGrid, intrinsic, omega_reference, batch: int = 256, keep=None):
    """What the fits and the regressor need of each binary at its reference,
    in JAX batches (a hundred times faster than :func:`reference_frame` and
    :func:`carrier_table` in numpy, one binary at a time): the rotations
    :math:`R_0` ``(N, 3, 3)``, the carrier phases and rates ``(N, 2,
    n_quadrature)`` each, and the :func:`g_baseline` ``(N, n_coefficients)``;
    only those of index in ``keep``, if given."""
    return _batched_call(
        _jitted_tables(grid), _batch_size(len(intrinsic), batch), intrinsic, omega_reference, keep=keep
    )


def _fit_chunk(grid: AngleGrid, intrinsic, omega_reference, x, zeta, g, priors=None):
    """:func:`fit_envelopes` of each of a few binaries: their coefficients and
    largest residuals."""
    _, phases, rates = carrier_tables(grid, intrinsic, omega_reference, keep=(0, 1, 2))
    results = [
        fit_envelopes(grid, None, x[i], zeta[i], g[i], None if priors is None else priors[i],
                      table=(phases[i], rates[i]))
        for i in range(len(intrinsic))
    ]
    return np.array([r[0] for r in results]), np.array([r[1] for r in results])


@lru_cache(maxsize=None)
def _jitted_integration(grid: AngleGrid, n_steps: int) -> Callable:
    jax, _ = _jnp()

    def one(row, omega_reference):
        angles, state, derivative = integrate_angles(
            row[0], row[1], row[2], row[3:6], row[6:9], omega_reference / np.pi,
            grid.omega_min / np.pi, n_steps, grid.omega_switch / np.pi,
            full_state=True,
        )
        return angles.x, state, derivative

    return jax.jit(jax.vmap(one))


def _integrate_and_fit(grid: AngleGrid, intrinsic, omega_reference, n_steps: int, stride: int, batch: int):
    """:func:`generate_training_set` for a few binaries, in one process: the
    integration in a JAX batch, then the fits of each binary."""
    x, state, derivative = _batched_call(
        _jitted_integration(grid, n_steps), batch, intrinsic, omega_reference
    )
    rotation, phases, rates, baselines = carrier_tables(grid, intrinsic, omega_reference, batch)
    names = ("zeta", "g", "switch", "coefficients", "residuals")
    results = []
    for i in range(len(intrinsic)):
        table = (phases[i], rates[i])
        zeta, g = j_frame_angles(
            rotation[i], x[i], state[i, :, 6:10], derivative[i, :, 6:10], (x.shape[1] - 1) // 2
        )
        switch = fit_switch_spins(
            grid, None, x[i], state[i, :, 0:3] @ rotation[i], state[i, :, 3:6] @ rotation[i], table
        )
        zeta, g = zeta[::stride], g[::stride]
        coefficients, residuals = fit_envelopes(grid, None, x[i, ::stride], zeta, g, table=table)
        results.append((zeta, g, switch, coefficients, residuals))
    data = {name: np.array([r[k] for r in results]) for k, name in enumerate(names)}
    data["x"] = x[:, ::stride]
    data["baselines"] = baselines
    return data


def generate_training_set(
    grid: AngleGrid,
    intrinsic: np.ndarray,
    omega_reference: np.ndarray,
    n_steps: int = N_STEPS,
    stride: int = 2,
    batch: int = 32,
    n_jobs: int = -1,
) -> dict:
    r"""Integrate the precession of each binary over :attr:`AngleGrid.x_range`
    and fit its envelopes.

    The binaries are split in batches of ``batch``, each integrated (in a
    JAX batch, single-threaded) and fitted by one of ``n_jobs`` worker
    processes: ~35 ms a binary a core.

    Returns a dictionary of ``x``, :math:`\zeta` and :math:`G - G_{\rm
    ref}` (:func:`j_frame_angles`) on every ``stride``-th integration node,
    ``(N, 2 n_steps / stride + 1)`` each (the reference is the middle node;
    ``x`` is :func:`integration_nodes`), the ``(N, N_SWITCH)`` targets of
    :func:`fit_switch_spins` (``switch``), the envelope ``coefficients`` and
    largest ``residuals`` of :func:`fit_envelopes`, and the ``baselines``
    of :func:`g_baseline`.
    """
    from joblib import Parallel, delayed

    starts = range(0, len(intrinsic), batch)
    parts = []
    done = 0
    with Parallel(n_jobs=n_jobs, return_as="generator") as parallel:
        for part in parallel(
            delayed(_integrate_and_fit)(
                grid, intrinsic[s:s + batch], omega_reference[s:s + batch], n_steps, stride, batch
            )
            for s in starts
        ):
            parts.append(part)
            done += len(part["switch"])
            if len(parts) % max(len(starts) // 8, 1) == 0 or done == len(intrinsic):
                logging.info("precession regression: integrated and fitted %i/%i", done, len(intrinsic))
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def integration_nodes(grid: AngleGrid, omega_reference, n_steps: int = N_STEPS, stride: int = 2) -> np.ndarray:
    """The ``x`` of :func:`generate_training_set`, from the reference
    frequencies alone, ``(N, 2 n_steps / stride + 1)``: the nodes of
    :func:`~mlgw_bns.batched_precession.integrate_angles` from
    ``omega_min`` to the switch, as it computes them."""
    omega_reference = np.pi * (np.asarray(omega_reference, dtype=float)[:, None] / np.pi)
    omega_lo = np.minimum(np.pi * (grid.omega_min / np.pi), omega_reference)
    kappa_omega = KAPPA * omega_lo

    def x_of(omega):
        return np.log(omega) - kappa_omega / omega

    x_reference = x_of(omega_reference)
    back_step = (x_of(omega_lo) - x_reference) / n_steps
    forward_step = (x_of(np.pi * (grid.omega_switch / np.pi)) - x_reference) / n_steps
    steps = np.arange(n_steps + 1)
    x = np.concatenate(
        [(x_reference + back_step * steps)[:, :0:-1], x_reference + forward_step * steps], axis=1
    )
    return x[:, ::stride]


def training_data(
    grid: AngleGrid,
    intrinsic: np.ndarray,
    omega_reference: np.ndarray,
    n_steps: int = N_STEPS,
    batch: int = 32,
    stride: int = 2,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""The ``x``, :math:`\zeta`, :math:`G - G_{\rm ref}` and switch targets
    of :func:`generate_training_set`."""
    data = generate_training_set(grid, intrinsic, omega_reference, n_steps, stride, batch, n_jobs)
    return data["x"], data["zeta"], data["g"], data["switch"]


def fit_all_envelopes(
    grid: AngleGrid, intrinsic, omega_reference, x, zeta, g, priors=None, n_jobs: int = -1,
    batch: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    r""":func:`fit_envelopes` for each binary of :func:`training_data`, in
    parallel (batches of ``batch`` per worker): the ``(N, n_envelopes,
    n_coefficients)`` coefficients and the ``(N, 2)`` largest residuals of
    the fits of :math:`\zeta` and :math:`G`."""
    from joblib import Parallel, delayed

    parts = Parallel(n_jobs=n_jobs)(
        delayed(_fit_chunk)(
            grid, intrinsic[s:s + batch], omega_reference[s:s + batch], x[s:s + batch],
            zeta[s:s + batch], g[s:s + batch], None if priors is None else priors[s:s + batch],
        )
        for s in range(0, len(intrinsic), batch)
    )
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


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
    """:func:`g_baseline` of each binary, ``(N, n_coefficients)`` (numpy,
    computed in JAX batches)."""
    return carrier_tables(grid, intrinsic, omega_reference, keep=(3,))[0]


@dataclass
class PrecessionRegressor:
    """Precession angles from the parameters, by PCA and kernel ridge (or a
    :class:`~mlgw_bns.jax_mlp.JaxMLP`) on the envelopes of
    :func:`fit_envelopes` (and on the spins at the switch,
    :func:`fit_switch_spins`); see the module docstring.

    Build one with :meth:`train`; :meth:`angles` gives the
    :class:`RegressedAngles` of a binary in numpy, :meth:`jax_angles` a
    batched JAX function giving them, for
    :func:`~mlgw_bns.batched_precession.precessing_mode_components`.
    """

    grid: AngleGrid
    ranges: TrainingRanges
    pca_zeta: Any  # PrincipalComponentData
    pca_g: Any
    network: Any  # KernelRidgeNetwork or JaxMLP

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
        mlp: Optional[Any] = None,
        mlp_loss: str = "natural",
        baselines: Optional[np.ndarray] = None,
        mlp_checkpoint: Optional[str] = None,
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
        mlp : MLPConfig, optional
            Regress with a :class:`~mlgw_bns.jax_mlp.JaxMLP` so configured
            instead of kernel ridge.
        mlp_loss : str
            ``"natural"`` weighs the perceptron's squared errors as those of
            the envelope coefficients and switch targets they reconstruct
            (each principal component by its variance in the units of the
            data); ``"uniform"`` weighs every standardized output alike, as
            kernel ridge, which fits each on its own, effectively does.
        baselines : array, optional
            The :func:`g_baseline` of each binary, if known
            (:func:`generate_training_set` gives them).
        mlp_checkpoint : str, optional
            Where the perceptron keeps its training state, to resume from
            (:meth:`~mlgw_bns.jax_mlp.JaxMLP.fit`).
        """
        if baselines is None:
            baselines = _g_baselines(grid, intrinsic, omega_reference)
        return cls.train_on_chunks(
            lambda: [(intrinsic, omega_reference, coefficients, switch, baselines)],
            n_components=n_components, kernel_gamma=kernel_gamma, grid=grid, ranges=ranges,
            mlp=mlp, mlp_loss=mlp_loss, mlp_checkpoint=mlp_checkpoint,
        )

    @classmethod
    def train_on_chunks(
        cls,
        chunks: Callable,
        n_components: Tuple[int, int] = (64, 64),
        kernel_gamma: float = 0.02,
        grid: AngleGrid = AngleGrid(),
        ranges: TrainingRanges = TrainingRanges(),
        mlp: Optional[Any] = None,
        mlp_loss: str = "natural",
        mlp_checkpoint: Optional[str] = None,
    ) -> "PrecessionRegressor":
        """:meth:`train` on a training set given in pieces, too large to hold:
        ``chunks()`` gives an iterable of ``(intrinsic, omega_reference,
        coefficients, switch, baselines)`` of disjoint sets of binaries, and
        is called twice. The first pass accumulates the covariances of the
        envelopes (:class:`~mlgw_bns.principal_component_analysis.CovarianceAccumulator`),
        the second projects them on the principal components; only those
        projections, the features and the switch targets of all the binaries
        are held at once."""
        from .data_management import PrincipalComponentData
        from .neural_network import Hyperparameters, KernelRidgeNetwork
        from .principal_component_analysis import CovarianceAccumulator

        def targets(chunk):
            intrinsic, omega_reference, coefficients, switch, baselines = chunk
            coefficients = coefficients.copy()
            coefficients[:, grid.n_zeta] -= baselines
            return (
                regression_features(np, intrinsic, omega_reference),
                *_pack(coefficients, grid.n_zeta),
                _switch_targets(np, intrinsic, omega_reference, switch, inverse=False),
            )

        start = time.perf_counter()
        accumulators = (CovarianceAccumulator(), CovarianceAccumulator())
        for chunk in chunks():
            _, zeta_real, g_real, _ = targets(chunk)
            accumulators[0].add(zeta_real)
            accumulators[1].add(g_real)
        bases = [a.principal_components(k) for a, k in zip(accumulators, n_components)]
        features, projections, switches = [], [], []
        for chunk in chunks():
            chunk_features, zeta_real, g_real, chunk_switch = targets(chunk)
            features.append(chunk_features)
            switches.append(chunk_switch)
            projections.append([(block - mean) @ vectors for block, (vectors, _, mean) in zip((zeta_real, g_real), bases)])
        pcas, reduced = [], []
        for j, (vectors, values, mean) in enumerate(bases):
            projected = np.concatenate([p[j] for p in projections])
            scaling = np.max(np.abs(projected), axis=0)
            pcas.append(PrincipalComponentData(vectors, values, mean, scaling))
            reduced.append(projected / scaling)
        pca_zeta, pca_g = pcas
        switch = np.concatenate(switches)
        reduced = np.concatenate(reduced + [switch], axis=1)
        features = np.concatenate(features)
        logging.info(
            "precession regression: principal components of %i binaries in %.0f s",
            len(features), time.perf_counter() - start,
        )
        if mlp is None:
            network = KernelRidgeNetwork(
                Hyperparameters.default_kernel_ridge(len(features), kernel_gamma=kernel_gamma)
            )
            network.fit(features, reduced)
        else:
            from .jax_mlp import JaxMLP

            weights = None
            if mlp_loss == "natural":
                units = np.concatenate([
                    pca_zeta.principal_components_scaling, pca_g.principal_components_scaling,
                    np.ones(switch.shape[1]),
                ])
                weights = np.var(reduced, axis=0) * units**2
            elif mlp_loss != "uniform":
                raise ValueError(f"unknown mlp_loss {mlp_loss!r}")
            network = JaxMLP(mlp).fit(features, reduced, weights, checkpoint=mlp_checkpoint)
        return cls(grid=grid, ranges=ranges, pca_zeta=pca_zeta, pca_g=pca_g, network=network)

    @property
    def _split(self) -> Tuple[int, int]:
        """Where the outputs of :attr:`network` change from the zeta to the G
        principal components, and from those to the switch targets."""
        k = self.pca_zeta.eigenvalues.size
        return k, k + self.pca_g.eigenvalues.size

    def predict(self, intrinsic: np.ndarray, omega_reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The ``(N, n_envelopes, n_coefficients)`` envelope coefficients and the
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
        coefficients[:, self.grid.n_zeta] += _g_baselines(self.grid, intrinsic, omega_reference)
        return coefficients, _switch_targets(
            np, intrinsic, omega_reference, reduced[:, kk:], inverse=True
        )

    def project(self, intrinsic: np.ndarray, omega_reference: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        """The envelope ``coefficients`` of these binaries after the PCA
        compression alone, as a perfect regression would predict them: the
        floor the principal components set."""
        from .principal_component_analysis import PrincipalComponentAnalysisModel

        baselines = _g_baselines(self.grid, intrinsic, omega_reference)
        coefficients = coefficients.copy()
        coefficients[:, self.grid.n_zeta] -= baselines
        blocks = [
            PrincipalComponentAnalysisModel.reconstruct_data(
                PrincipalComponentAnalysisModel.reduce_data(block, pca), pca
            )
            for block, pca in zip(_pack(coefficients, self.grid.n_zeta), (self.pca_zeta, self.pca_g))
        ]
        projected = _unpack(np, *blocks, self.grid.n_coefficients)
        projected[:, self.grid.n_zeta] += baselines
        return projected

    def predict_coefficients(self, intrinsic: np.ndarray, omega_reference: np.ndarray) -> np.ndarray:
        """The ``(N, n_envelopes, n_coefficients)`` envelope coefficients (numpy)."""
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
        switch)``: the regressed ``(N, n_envelopes, n_coefficients)`` envelope
        coefficients, without the :func:`g_baseline` of :math:`g_0`, and the
        ``(N, N_SWITCH)`` switch targets."""
        from .batched import _freeze_kernel_ridge, _kernel_ridge
        from .jax_mlp import JaxMLP

        _, jnp = _jnp()
        if isinstance(self.network, JaxMLP):
            reduced_of = self.network.jax_function()
        else:
            regressor = self.network.regressor
            kernel = _freeze_kernel_ridge(regressor.X_fit_, regressor.dual_coef_, regressor.gamma, [])
            kernel = type(kernel)(*(jnp.asarray(getattr(kernel, f.name)) if f.name != "gamma"
                                    else kernel.gamma for f in fields(kernel)))
            param_mean = jnp.asarray(self.network.param_scaler.mean_)
            param_scale = jnp.asarray(self.network.param_scaler.scale_)
            target_mean = jnp.asarray(self.network.target_scaler.mean_)
            target_scale = jnp.asarray(self.network.target_scaler.scale_)

            def reduced_of(features):
                (reduced,) = _kernel_ridge(jnp, (features - param_mean) / param_scale, [kernel])
                return reduced * target_scale + target_mean

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
            reduced = reduced_of(regression_features(jnp, intrinsic, omega_reference))
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
            coefficients_row = coefficients_row.at[grid.n_zeta].add(g_baseline(jnp, grid, frame, rates, nodes))
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
