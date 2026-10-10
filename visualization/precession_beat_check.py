r"""Where does the two-spin beat of the regressed precession angles go wrong near q = 1?

:mod:`mlgw_bns.precession_regression` carries the in-plane part of
:math:`\hat{L}` on two carriers whose rates are the normal-mode frequencies
of the *linearized* two-spin precession
(:func:`~mlgw_bns.precession_regression.carrier_rates`); their difference is
the beat of the two spins, which near equal masses it gets wrong by up to
~20%. This compares, along integrated binaries
(:func:`~mlgw_bns.batched_precession.integrate_angles`), three beat rates
:math:`\mathrm{d}\Phi_{\rm beat} / \mathrm{d}\Omega`:

* ``true``: half a cycle between successive extrema of :math:`\vec{S}_A
  \cdot \hat{L}`, which oscillates at the nutation (beat) frequency for all
  mass ratios (unlike :math:`|\vec{S}|`, constant at :math:`q = 1`);
* ``linear``: the splitting of the two carrier rates (``elliptic_beat=False``);
* ``elliptic``: the closed-form nutation frequency that replaces it
  (:func:`~mlgw_bns.precession_regression.nutation_frequency`), from the
  reference values alone;
* ``frozen``: the nutation period of the conservative dynamics at fixed
  :math:`\Omega`, from the integrated state there (the same right-hand side
  with :math:`\dot\Omega = 0`). This is what the multiple-scale analysis
  gives in closed form, through the roots of a cubic and a complete elliptic
  integral (Kesden et al. 2015; Chatziioannou et al. 2017; regular at
  :math:`q = 1` in the weighted-spin-difference form of Gerosa et al. 2023,
  arXiv:2304.04801): if it matches ``true`` where ``linear`` does not, the
  closed form is the fix. It also isolates the error of ``elliptic`` that
  comes from carrying the constants (above all :math:`J`) from the
  reference.

Run with::

    python visualization/precession_beat_check.py [--n 6] [--seed 0]
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.integrate import solve_ivp

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from mlgw_bns.batched_precession import integrate_angles  # noqa: E402
from mlgw_bns.precession_regression import (  # noqa: E402
    AngleGrid,
    carrier_rates,
    reference_frame,
)
from mlgw_bns.twist_waveform import _pn_precession_derivatives, nu_to_X1  # noqa: E402


def binary(q, chi_perp, rng, chi_z=(-0.2, 0.2), lam=(400.0, 400.0)):
    """``intrinsic`` row with in-plane spins of magnitude ``chi_perp`` in random directions."""
    row = np.empty(9)
    row[0], row[1], row[2] = q, *lam
    for start, cz in zip((3, 6), chi_z):
        angle = rng.uniform(0, 2 * np.pi)
        row[start:start + 3] = chi_perp * np.cos(angle), chi_perp * np.sin(angle), cz
    return row


def extrema_rates(omega, signal):
    r"""``(Omega_mid, dPhi/dOmega, Omega_extrema)``: pi over the Omega between
    successive extrema."""
    d = np.diff(signal)
    turning = np.nonzero(np.sign(d[1:]) != np.sign(d[:-1]))[0] + 1
    # parabolic refinement of each extremum
    refined = []
    for i in turning:
        if 0 < i < len(signal) - 1:
            y0, y1, y2 = signal[i - 1], signal[i], signal[i + 1]
            den = y0 - 2 * y1 + y2
            shift = 0.5 * (y0 - y2) / den if den != 0 else 0.0
            refined.append(np.interp(i + shift, np.arange(len(omega)), omega))
    refined = np.array(refined)
    return 0.5 * (refined[1:] + refined[:-1]), np.pi / np.diff(refined), refined


def frozen_rate(nu, q, a10, omega, s_a, s_b, l_hat, guess):
    r"""Nutation angular frequency (per unit time) of the dynamics at fixed ``omega``,
    and :math:`\dot\Omega` there; ``guess``, a rough angular frequency, sets
    the span (a few cycles)."""

    def rhs(_, y):
        d_sa, d_sb, d_l, _, _, _ = _pn_precession_derivatives(
            nu, q, omega, y[0:3], y[3:6], y[6:9], a10
        )
        return np.concatenate([d_sa, d_sb, d_l])

    y0 = np.concatenate([s_a, s_b, l_hat])
    _, _, _, _, omega_dot, _ = _pn_precession_derivatives(nu, q, omega, s_a, s_b, l_hat, a10)

    def projection_rate(t, y):
        d = rhs(t, y)
        return np.dot(d[0:3], y[6:9]) + np.dot(y[0:3], d[6:9])

    span = 4 * 2 * np.pi / abs(guess)
    solution = solve_ivp(
        rhs, (0, span), y0, method="DOP853", rtol=1e-11, atol=1e-14,
        events=projection_rate, dense_output=False,
    )
    times = solution.t_events[0]
    if len(times) < 3:
        return np.nan, omega_dot
    # extrema alternate: a full period is every other one
    return 2 * np.pi / np.mean(times[2:] - times[:-2]), omega_dot


def beat_phases(grid, row, omega_reference, n_frozen):
    r"""Integrate the binary ``row`` and compare the beat phase each model
    accumulates between successive extrema of :math:`\vec{S}_A \cdot \hat{L}`
    with the true :math:`\pi`: ``(mid, phases, frozen_inside)``, the
    :math:`\Omega` of each half-cycle, the relative error of each model's
    phase over it, and where the frozen rate is interpolated rather than
    extrapolated."""
    q = row[0]
    angles, state, _ = integrate_angles(
        row[0], row[1], row[2], jnp.asarray(row[3:6]), jnp.asarray(row[6:9]),
        omega_reference / np.pi, grid.omega_min / np.pi, 4096,
        final_frequency_22=grid.omega_switch / np.pi, full_state=True,
    )
    state = np.asarray(state)
    omega = state[:, 10]
    s_a, s_b, l_hat = state[:, 0:3], state[:, 3:6], state[:, 6:9]
    mid, true, edges = extrema_rates(omega, np.sum(s_a * l_hat, axis=1))

    frame = reference_frame(np, grid, row, omega_reference)
    nu = q / (1 + q) ** 2
    a10 = float(frame.a10_tidal)
    picks = np.unique(np.geomspace(1, len(mid), n_frozen).astype(int) - 1)
    frozen = np.empty(len(picks))
    for j, i in enumerate(picks):
        k = np.searchsorted(omega, mid[i])
        _, _, _, _, omega_dot, _ = _pn_precession_derivatives(
            nu, q, mid[i], s_a[k], s_b[k], l_hat[k], a10
        )
        f_rate, omega_dot = frozen_rate(
            nu, q, a10, mid[i], s_a[k], s_b[k], l_hat[k], true[i] * omega_dot
        )
        frozen[j] = f_rate / omega_dot

    # the beat phase each model accumulates between successive true
    # extrema, which is pi: the rate is not constant over a half-cycle,
    # so comparing rates at its midpoint is biased where it is long
    fine = np.geomspace(edges[0], edges[-1], 20000)
    ok = np.isfinite(frozen)
    models = {
        name: np.subtract(*carrier_rates(np, frame, fine, np.ones_like(fine), elliptic))
        for name, elliptic in (("linear", False), ("elliptic", True))
    }
    models["frozen"] = np.exp(np.interp(np.log(fine), np.log(mid[picks][ok]), np.log(frozen[ok])))
    phases = {}
    for name, rate in models.items():
        cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(fine))])
        phases[name] = np.diff(np.interp(edges, fine, cumulative)) / np.pi - 1
    inside = (mid >= mid[picks][ok][0]) & (mid <= mid[picks][ok][-1]) if ok.any() else np.zeros_like(mid, bool)
    return mid, phases, inside


def validation_binaries(args, grid):
    """The ``--top`` binaries of ``--validation`` with the most anharmonic
    nutation, and ``--controls`` of the least."""
    from precession_regression_study import load, nutation_parameters

    data = load(args.validation)
    max_m = nutation_parameters(grid, data["intrinsic"], data["omega_reference"]).max(axis=1)
    order = np.argsort(max_m)[::-1]
    chosen = np.concatenate([order[: args.top], order[::-1][: args.controls]])
    return data["intrinsic"][chosen], data["omega_reference"][chosen], max_m[chosen]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=6, help="Omega samples per binary for the frozen period")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chi-perp", type=float, nargs="+", default=[0.1, 0.4])
    parser.add_argument("--q", type=float, nargs="+", default=[1.02, 1.1, 1.25, 1.5, 2.0])
    parser.add_argument("--validation", nargs="+",
                        help="instead, binaries of these files of precession_regression_study.py")
    parser.add_argument("--top", type=int, default=12, help="... with the largest max m")
    parser.add_argument("--controls", type=int, default=4, help="... and with the smallest")
    args = parser.parse_args()

    grid = AngleGrid()
    if args.validation:
        rows, omegas, max_m = validation_binaries(args, grid)
        print("beat phase accumulated over the band minus the true one, in cycles "
              "(frozen: between its nodes only)")
        print(f"{'q':>5} {'chi_1z':>6} {'chi_2z':>6} {'max m':>6} {'cycles':>7} {'linear':>8} {'elliptic':>9} {'frozen':>8}")
        for row, omega_reference, m in zip(rows, omegas, max_m):
            mid, phases, inside = beat_phases(grid, row, omega_reference, args.n)
            total = {name: np.sum(phase) / 2 for name, phase in phases.items()}
            total["frozen"] = np.sum(phases["frozen"][inside]) / 2
            print(f"{row[0]:5.2f} {row[5]:+6.2f} {row[8]:+6.2f} {m:6.3f} {len(mid) / 2:7.1f}"
                  f" {total['linear']:+8.2f} {total['elliptic']:+9.2f} {total['frozen']:+8.2f}")
        return

    omega_reference = 1e-3
    rng = np.random.default_rng(args.seed)
    print("relative error of the beat phase over each half-cycle, averaged in bands of Omega")
    print(f"{'q':>5} {'chi_p':>5} {'M Omega':>9} {'n_half':>7} {'linear':>9} {'elliptic':>9} {'frozen':>9}")
    for chi_perp in args.chi_perp:
        for q in args.q:
            row = binary(q, chi_perp, rng)
            mid, phases, inside = beat_phases(grid, row, omega_reference, args.n)
            # report in bands of Omega, the frozen one only between its nodes
            bands = np.geomspace(mid[0], mid[-1] * (1 + 1e-12), args.n + 1)
            for lo, hi in zip(bands[:-1], bands[1:]):
                sel = (mid >= lo) & (mid < hi)
                if not sel.any():
                    continue
                lin = np.mean(phases["linear"][sel])
                ell = np.mean(phases["elliptic"][sel])
                frz = np.mean(phases["frozen"][sel & inside]) if (sel & inside).any() else np.nan
                print(f"{q:5.2f} {chi_perp:5.2f} {np.sqrt(lo * hi):9.2e} {sel.sum():7d}"
                      f" {lin:+9.4f} {ell:+9.4f} {frz:+9.4f}")
            print()


if __name__ == "__main__":
    main()
