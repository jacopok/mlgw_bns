r"""The precession angles of two validation binaries of a regressor, one it
predicts well and one badly: integrated, the analytic backbone of the
regression, and the regressor's prediction, and the deviations of the last
two from the integration.

The backbone is what the representation of
:mod:`mlgw_bns.precession_regression` knows in closed form, with nothing
regressed: the carriers, the secular growth of :math:`G`
(:func:`~mlgw_bns.precession_regression.g_baseline`) and, for :math:`\zeta`,
the leading order :math:`L \hat{L}_\perp = -(s_A + s_B)` with the in-plane
spins of the reference each on its carrier (spin A, the heavier, on the slow
one): :math:`c_0 = 0`, :math:`c_{\rm fast} = -s_B / L`, :math:`c_{\rm slow}
= -s_A / L`. It is defined below the switch only; above it the regressor
integrates from the state it predicts there.

    python precession_angles_cases.py \
        precession_scale/run/runs/mlp_n1048576_best139_1048576.joblib
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from mlgw_bns.batched_precession import integrate_angles
from mlgw_bns.precession_regression import (
    PrecessionRegressor,
    _carriers,
    _evaluate,
    _rotation_from,
    carrier_table,
    euler_angles_of,
    g_baseline,
    reference_frame,
)
from mlgw_bns.taylorf2 import SUN_MASS_SECONDS
from mlgw_bns.twist_waveform import orbital_angular_momentum

#: Total mass giving the frequency axis; the angles depend on M Omega only.
TOTAL_MASS = 2.8


def backbone_angles(grid, intrinsic, omega_reference, omega):
    """The Euler angles of the analytic backbone at the orbital frequencies
    ``omega`` (below the switch)."""
    frame = reference_frame(np, grid, intrinsic, omega_reference)
    phases, rates = carrier_table(np, grid, frame)
    in_plane = [(frame.rotation.T @ spin)[:2] for spin in (frame.spin_b, frame.spin_a)]
    l_nodes = orbital_angular_momentum(frame.nu, grid.omega_of(grid.coefficient_x))
    coefficients = np.zeros((grid.n_envelopes, grid.n_coefficients), complex)
    for term, s in zip([(1, 0), (0, 1)], in_plane):  # fast: spin B, slow: spin A
        coefficients[grid.zeta_terms.index(term)] = -(s[0] + 1j * s[1]) / l_nodes
    coefficients[grid.n_zeta] = g_baseline(np, grid, frame, rates)
    x = grid.x_of(omega)
    zeta, g = _evaluate(np, grid, coefficients, _carriers(np, grid, phases, rates, x), x)
    return euler_angles_of(np, _rotation_from(np, frame.rotation, zeta, g + frame.g_reference))


def all_angles(regressor, intrinsic, omega_reference, omega):
    grid = regressor.grid
    integrated = integrate_angles(
        intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3:6], intrinsic[6:9],
        omega_reference / np.pi, grid.omega_min / np.pi,
    )
    below = grid.x_of(omega) <= grid.x_switch
    backbone = [np.where(below, a, np.nan) for a in backbone_angles(grid, intrinsic, omega_reference, omega)]
    return {
        "integrated": [np.asarray(a) for a in integrated.at_momega(omega)],
        "backbone": backbone,
        "regressor": list(regressor.angles(intrinsic, omega_reference).at_momega(omega)),
    }


def wrapped(a):
    return np.angle(np.exp(1j * a))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("regressor")
    parser.add_argument("--validation", default="precession_scale/run/validation/exact.npz")
    parser.add_argument("--good-quantile", type=float, default=0.5)
    parser.add_argument("--bad-quantile", type=float, default=1.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    regressor = PrecessionRegressor.load(args.regressor)
    name = args.regressor.rsplit("/", 1)[-1].removesuffix(".joblib")
    results = np.load(args.regressor.replace(name + ".joblib", f"validate_{name}.npz"))
    mismatches, intrinsic = results["mismatches"], results["intrinsic"]
    omega_reference = np.load(args.validation)["omega_reference"][: len(mismatches)]
    order = np.argsort(mismatches)
    cases = [
        ("good", order[int(args.good_quantile * (len(order) - 1))]),
        ("bad", order[int(args.bad_quantile * (len(order) - 1))]),
    ]

    grid = regressor.grid
    to_hz = 1.0 / (np.pi * TOTAL_MASS * SUN_MASS_SECONDS)
    figure, axes = plt.subplots(
        6, len(cases), figsize=(7 * len(cases), 15), sharex=True, squeeze=False,
        gridspec_kw={"height_ratios": [2, 2, 2, 1.3, 1.3, 1.3]},
    )
    styles = {
        "integrated": dict(color="k", lw=2.2),
        "backbone": dict(color="C1", lw=1.0),
        "regressor": dict(color="C0", lw=1.0, ls="--"),
    }
    labels = [r"$\alpha$", r"$\beta$", r"$\gamma$"]
    for column, (case, i) in zip(axes.T, cases):
        omega = np.geomspace(grid.omega_min, grid.omega_max, 4000)
        angles = all_angles(regressor, intrinsic[i], omega_reference[i], omega)
        f = omega * to_hz
        exact = angles["integrated"]
        for k in range(3):
            for source, values in angles.items():
                shown = values[k] if k == 1 else np.unwrap(np.nan_to_num(values[k]))
                if source == "backbone":
                    shown = np.where(np.isnan(values[k]), np.nan, shown)
                column[k].plot(f, shown, label=source, **styles[source])
                if source != "integrated":
                    column[3 + k].plot(f, np.abs(wrapped(values[k] - exact[k])), **styles[source])
            column[k].set_ylabel(labels[k])
            column[3 + k].set(ylabel=rf"$|\Delta${labels[k]}$|$ [rad]", yscale="log", ylim=(1e-7, 4))
        q, l1, l2 = intrinsic[i, :3]
        column[0].set_title(
            f"{case}: mismatch {mismatches[i]:.1e} (validation binary {i})\n"
            f"q {q:.2f}, Λ {l1:.0f}/{l2:.0f}, χ1 {np.round(intrinsic[i, 3:6], 2)}, "
            f"χ2 {np.round(intrinsic[i, 6:9], 2)}, f_ref {omega_reference[i] * to_hz:.1f} Hz",
            fontsize=9, loc="left",
        )
        for ax in column:
            ax.axvline(omega_reference[i] * to_hz, color="0.5", lw=0.8, ls=":")
            ax.axvspan(grid.omega_switch * to_hz, grid.omega_max * to_hz, color="0.93", zorder=0)
            ax.axvspan(f[0], 20.0, color="0.93", zorder=0)
            ax.set_xscale("log")
        column[0].legend(fontsize=8)
        column[-1].set_xlabel(f"(2, 2) frequency [Hz] at M = {TOTAL_MASS}")
    out = args.out or f"precession_angles_cases_{name}.png"
    figure.tight_layout()
    figure.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main()
