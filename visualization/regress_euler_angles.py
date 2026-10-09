r"""Prototype: regress the PN spin-precession Euler angles instead of integrating them.

The JAX precessing model spends 77--94 % of its time in
:func:`~mlgw_bns.batched_precession.integrate_angles` (see
``profile_precession_dynamics.py``). Here the integration is replaced by what
the aligned-spin surrogate does for its phases: a PCA of the tabulated
solution plus a kernel-ridge regression from the parameters to the
coefficients.

What is regressed, and why that form:

* the state :math:`[\hat{L}, \alpha - \gamma]` of
  :class:`~mlgw_bns.batched_precession.TabulatedAngles`, not the angles: it is
  smooth through the reference frequency, where :math:`\hat{L} = \hat{z}` and
  :math:`\beta = |\cdot|` has a kink;
* in a *canonical frame*: rotating both spins about :math:`\hat{z}` rotates
  :math:`L_x + i L_y` by the same angle and leaves :math:`\alpha - \gamma`
  alone (checked below), so the first spin is put in the x-z plane and the
  result is rotated back. That removes one of the nine inputs and makes the
  azimuth exact rather than learned;
* on a grid :math:`w \in [0, 1]` uniform in the integration variable
  :math:`x(\Omega)` between a fixed lowest frequency and the binary's own
  :math:`1.1\,\Omega_{\rm mrg}` (an analytic function of the parameters), so
  that the end of the table is not a moving kink.

Inputs: :math:`q, \Lambda_1, \Lambda_2, \chi_{1\perp}, \chi_{1z}, \chi_{2x},
\chi_{2y}, \chi_{2z}` (canonical frame) and :math:`\Omega_{\rm ref} = \pi M
f_{\rm ref}`, the only way the total mass enters the dynamics.

Run with::

    python visualization/regress_euler_angles.py [--n-train 2000] [--n-test 200]
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
from mlgw_bns.neural_network import Hyperparameters, KernelRidgeNetwork
from mlgw_bns.principal_component_analysis import PrincipalComponentAnalysisModel

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from mlgw_bns.batched_precession import KAPPA, integrate_angles  # noqa: E402
from mlgw_bns.taylorf2 import SUN_MASS_SECONDS  # noqa: E402

F_REF_HZ = 20.0
LARGEST_M = 4
MASS_RANGE = (2.5, 4.0)
MAX_IN_PLANE = 0.4


def lowest_omega(f_min_hz: float) -> float:
    """The global lowest orbital frequency of the table: the lowest start any
    binary of the training range needs to twist its (4, 4) multipole down to
    ``f_min_hz`` (the smallest total mass). The precession phase grows as
    :math:`1/\\Omega`, so this sets how hard the regression is."""
    return np.pi * 2.0 * f_min_hz * MASS_RANGE[0] * SUN_MASS_SECONDS / LARGEST_M


def draw_binaries(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {
        "q": rng.uniform(1.0, 3.0, n),
        "lambda_1": rng.uniform(5.0, 5000.0, n),
        "lambda_2": rng.uniform(5.0, 5000.0, n),
        "total_mass": rng.uniform(*MASS_RANGE, n),
    }
    for i in (1, 2):
        perp = rng.uniform(0.0, MAX_IN_PLANE, n)
        azimuth = rng.uniform(0.0, 2 * np.pi, n)
        out[f"chi_{i}"] = np.stack(
            [perp * np.cos(azimuth), perp * np.sin(azimuth), rng.uniform(-0.5, 0.5, n)], axis=1
        )
    return out


def canonical_features(b: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Regression inputs, and the azimuth ``phi_1`` of the first spin that
    the canonical frame is rotated by."""
    phi_1 = np.arctan2(b["chi_1"][:, 1], b["chi_1"][:, 0])
    c2 = (b["chi_2"][:, 0] + 1j * b["chi_2"][:, 1]) * np.exp(-1j * phi_1)
    omega_ref = np.pi * F_REF_HZ * b["total_mass"] * SUN_MASS_SECONDS
    features = np.stack(
        [
            b["q"], b["lambda_1"], b["lambda_2"],
            np.hypot(b["chi_1"][:, 0], b["chi_1"][:, 1]), b["chi_1"][:, 2],
            c2.real, c2.imag, b["chi_2"][:, 2],
            omega_ref,
        ],
        axis=1,
    )
    return features, phi_1


def canonical_binaries(features: np.ndarray) -> dict[str, np.ndarray]:
    """The inverse of :func:`canonical_features` (``phi_1 = 0``)."""
    zeros = np.zeros(len(features))
    return {
        "q": features[:, 0], "lambda_1": features[:, 1], "lambda_2": features[:, 2],
        "chi_1": np.stack([features[:, 3], zeros, features[:, 4]], axis=1),
        "chi_2": np.stack([features[:, 5], features[:, 6], features[:, 7]], axis=1),
        "total_mass": features[:, 8] / (np.pi * F_REF_HZ * SUN_MASS_SECONDS),
    }


def grid_state(
    q, lambda_1, lambda_2, chi_1, chi_2, omega_ref, omega_lo: float, omega_cut: float,
    n_steps: int, n_grid: int,
):
    r"""The variables :math:`[\vec{S}_A, \vec{S}_B, \hat{L}, \alpha - \gamma]` of
    one binary on the ``w`` grid, the orbital frequency of the nodes, and
    :math:`\alpha - \gamma` at the reference frequency (whose branch is arbitrary)."""
    angles, state, derivative = integrate_angles(
        q, lambda_1, lambda_2, chi_1, chi_2, omega_ref / np.pi, omega_lo / np.pi, n_steps,
        full_state=True,
    )
    tab = replace(angles, state=state[:, :10], derivative=derivative[:, :10])
    x_lo, x_hi = tab._x(omega_lo), tab._x(jnp.minimum(tab.omega_hi, omega_cut))
    x = x_lo + jnp.linspace(0.0, 1.0, n_grid) * (x_hi - x_lo)
    # invert x(omega) = ln omega - kappa omega_lo / omega by Newton steps in ln omega
    omega = jnp.exp(x)
    for _ in range(60):
        f = jnp.log(omega) - tab.kappa_omega / omega - x
        omega = omega * jnp.exp(-f / (1.0 + tab.kappa_omega / omega))
    reference = tab.state[(tab.x.shape[0] - 1) // 2, 9]
    return tab.state_at_momega(omega), omega, reference


def integrate(
    b: dict[str, np.ndarray], omega_lo: float, omega_cut: float, n_steps: int, n_grid: int,
    batch: int = 64,
):
    """:func:`grid_state` for every binary of ``b`` (``chi_*`` as drawn, so
    not necessarily in the canonical frame), in batches."""
    omega_ref = np.pi * F_REF_HZ * b["total_mass"] * SUN_MASS_SECONDS
    fn = jax.jit(jax.vmap(lambda *a: grid_state(*a, omega_lo, omega_cut, n_steps, n_grid)))
    chunks = []
    for start in range(0, len(omega_ref), batch):
        sl = slice(start, start + batch)
        chunks.append(fn(*(jnp.asarray(a[sl]) for a in (
            b["q"], b["lambda_1"], b["lambda_2"], b["chi_1"], b["chi_2"], omega_ref
        ))))
    return tuple(np.concatenate([np.asarray(c[i]) for c in chunks]) for i in range(3))


def orbital_angular_momentum(nu: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """:math:`|\\vec{L}| / M^2` at 2PN, as :func:`_pn_precession_derivatives` has it."""
    v = omega ** (1 / 3)
    nu = nu[:, None]
    return nu / v * (1 + v**2 * (1.5 + nu / 6) + v**4 * (3.375 - 2.375 * nu + nu**2 / 24))


def j_frame(state: np.ndarray, omega: np.ndarray, nu: np.ndarray):
    r""":math:`\hat{J}`, and the basis :math:`\hat{e}_1 \propto \hat{z} -
    (\hat{z}\cdot\hat{J})\hat{J}`, :math:`\hat{e}_2 = \hat{J} \times \hat{e}_1`."""
    spin = state[..., 0:3] + state[..., 3:6]
    l_vec = state[..., 6:9]
    total = orbital_angular_momentum(nu, omega)[..., None] * l_vec + spin
    j_hat = total / np.linalg.norm(total, axis=-1, keepdims=True)
    e_1 = np.array([0.0, 0.0, 1.0]) - j_hat[..., 2:3] * j_hat
    e_1 /= np.linalg.norm(e_1, axis=-1, keepdims=True)
    return j_hat, e_1, np.cross(j_hat, e_1)


def reference_index(omega: np.ndarray, omega_ref: np.ndarray) -> np.ndarray:
    """The node nearest the reference frequency, for each binary."""
    return np.argmin(np.abs(omega - omega_ref[:, None]), axis=1)


def at_reference(values: np.ndarray, omega: np.ndarray, omega_ref: np.ndarray) -> np.ndarray:
    """``values`` (``(N, n_grid)``) linearly interpolated at ``omega_ref``."""
    k = np.clip(reference_index(omega, omega_ref), 1, omega.shape[1] - 2)
    rows = np.arange(len(values))
    lo = np.where(omega[rows, k] <= omega_ref, k, k - 1)
    t = (omega_ref - omega[rows, lo]) / (omega[rows, lo + 1] - omega[rows, lo])
    return (1 - t) * values[rows, lo] + t * values[rows, lo + 1]


def to_targets(
    state: np.ndarray, omega: np.ndarray, reference: np.ndarray, nu: np.ndarray,
    omega_ref: np.ndarray,
):
    r"""The vector that is regressed. :math:`\hat{L}` winds around the total
    angular momentum :math:`\hat{J}` (tens of radians across the band), so
    it is written as :math:`\hat{L} = \cos\theta\, \hat{J} + \sin\theta\,
    (\cos\Phi\, \hat{e}_1 + \sin\Phi\, \hat{e}_2)` and the four smooth
    functions :math:`J_x, J_y, \theta, \Phi` are regressed. :math:`\Phi` is
    unwrapped and zero at the reference, where :math:`\hat{L} = \hat{z}`
    lies on :math:`\hat{e}_1`; :math:`\delta = (\alpha - \gamma) - (\alpha -
    \gamma)_{\rm ref}`; the last two numbers are the cosine and sine of the
    reference value (known modulo :math:`2\pi`, which is all the twist uses)."""
    j_hat, e_1, e_2 = j_frame(state, omega, nu)
    l_vec = state[..., 6:9]
    cos_theta = np.sum(l_vec * j_hat, axis=-1)
    in_plane = np.stack([np.sum(l_vec * e_1, axis=-1), np.sum(l_vec * e_2, axis=-1)], axis=-1)
    theta = np.arctan2(np.linalg.norm(in_plane, axis=-1), cos_theta)
    phi = np.unwrap(np.arctan2(in_plane[..., 1], in_plane[..., 0]), axis=1)
    phi -= 2 * np.pi * np.round(at_reference(phi, omega, omega_ref) / (2 * np.pi))[:, None]
    n = len(state)
    return np.concatenate(
        [
            j_hat[..., :2].reshape(n, -1), theta, phi, state[..., 9] - reference[:, None],
            np.cos(reference)[:, None], np.sin(reference)[:, None],
        ],
        axis=1,
    )


def from_targets(targets: np.ndarray, omega: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Inverse of :func:`to_targets`, up to the branch: ``(N, n_grid, 4)``, the
    state :math:`[\\hat{L}, \\alpha - \\gamma]`. Needs the nodes' orbital
    frequencies only to build the :math:`\\hat{J}` frame's :math:`\\hat{e}_i`
    (which depend on :math:`\\hat{J}` alone, so not really)."""
    n, n_grid = len(targets), omega.shape[1]
    j_xy = targets[:, : 2 * n_grid].reshape(n, n_grid, 2)
    theta, phi, delta = (targets[:, (2 + i) * n_grid : (3 + i) * n_grid] for i in range(3))
    j_hat = np.concatenate([j_xy, np.sqrt(1 - np.sum(j_xy**2, axis=-1, keepdims=True))], axis=-1)
    e_1 = np.array([0.0, 0.0, 1.0]) - j_hat[..., 2:3] * j_hat
    e_1 /= np.linalg.norm(e_1, axis=-1, keepdims=True)
    e_2 = np.cross(j_hat, e_1)
    l_vec = (
        np.cos(theta)[..., None] * j_hat
        + np.sin(theta)[..., None]
        * (np.cos(phi)[..., None] * e_1 + np.sin(phi)[..., None] * e_2)
    )
    reference = np.arctan2(targets[:, -1], targets[:, -2])
    return np.concatenate([l_vec, (delta + reference[:, None])[..., None]], axis=-1)


def angle_errors(predicted: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per node, in radians: the angle between the two :math:`\\hat{L}` and the
    error of :math:`\\alpha - \\gamma` modulo :math:`2\\pi`; ``truth`` is the
    full ten-column state."""
    unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    cosine = np.clip(np.sum(unit(predicted[..., :3]) * unit(truth[..., 6:9]), axis=-1), -1, 1)
    d = predicted[..., 3] - truth[..., 9]
    return np.arccos(cosine), np.abs(np.angle(np.exp(1j * d)))


def report(label: str, predicted: np.ndarray, truth: np.ndarray) -> None:
    l_err, g_err = angle_errors(predicted, truth)
    q = lambda e: " ".join(f"{np.percentile(e.max(axis=1), p):9.2e}" for p in (50, 90, 99, 100))
    print(f"{label:<28} max over grid of L-angle  [50/90/99/100 %ile of binaries]: {q(l_err)}")
    print(f"{'':<28} max over grid of (a-g)      [50/90/99/100 %ile of binaries]: {q(g_err)}")


def get_data(args, n: int, seed: int, name: str):
    """Canonical features and the integrated state for ``n`` random binaries, cached."""
    path = args.out_dir / (
        f"regress_euler_angles_{name}_{n}_{seed}_{args.n_steps}_{args.n_grid}_{args.f_min:g}Hz_cut{args.omega_cut:g}.npz"
    )
    if path.exists():
        data = np.load(path)
        return data["features"], data["state"], data["omega"], data["reference"]
    b = draw_binaries(n, seed)
    features, _ = canonical_features(b)
    t0 = time.perf_counter()
    state, omega, reference = integrate(
        canonical_binaries(features), lowest_omega(args.f_min), args.omega_cut, args.n_steps, args.n_grid
    )
    logging.info("integrated %d binaries in %.1f s", n, time.perf_counter() - t0)
    np.savez(path, features=features, state=state, omega=omega, reference=reference)
    return features, state, omega, reference


def target_blocks(n_grid: int) -> dict[str, slice]:
    """The slices of :func:`to_targets`' vector, each of which gets its own
    PCA and regressor."""
    n = n_grid
    return {
        "J": slice(0, 2 * n), "theta": slice(2 * n, 3 * n), "Phi": slice(3 * n, 4 * n),
        "delta": slice(4 * n, 5 * n), "ref": slice(5 * n, 5 * n + 2),
    }


def eta(features: np.ndarray) -> np.ndarray:
    return features[:, 0] / (1 + features[:, 0]) ** 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--f-min", type=float, default=20.0,
                        help="lowest GW frequency [Hz] the (4, 4) multipole is twisted at")
    parser.add_argument("--omega-cut", type=float, default=0.1,
                        help="upper end of the table, M Omega_orb (or 1.1 Omega_mrg if lower)")
    parser.add_argument("--n-grid", type=int, default=513)
    parser.add_argument("--stage", default="fit", choices=["symmetry", "fit"])
    parser.add_argument("--components", type=json.loads,
                        default={"J": 8, "theta": 20, "Phi": 40, "delta": 20, "ref": 2})
    parser.add_argument("--alpha", type=float, nargs="+", default=[1e-8],
                        help="fixed ridge penalties (0: leave-one-out selection)")
    parser.add_argument("--gamma", type=float, nargs="+", default=[0.01, 0.1])
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.stage == "fit":
        x_train, state_train, omega_train, ref_train = get_data(args, args.n_train, 10, "train")
        x_test, state_test, omega_test, ref_test = get_data(args, args.n_test, 11, "test")
        y_train = to_targets(state_train, omega_train, ref_train, eta(x_train), x_train[:, 8])
        y_test = to_targets(state_test, omega_test, ref_test, eta(x_test), x_test[:, 8])
        report("targets round trip", from_targets(y_test, omega_test, eta(x_test)), state_test)
        blocks = target_blocks(args.n_grid)
        for gamma, alpha in itertools.product(args.gamma, args.alpha):
            prediction = np.empty_like(y_test)
            roundtrip = np.empty_like(y_test)
            for name, sl in blocks.items():
                n_components = min(args.components[name], y_train[:, sl].shape[1])
                pca = PrincipalComponentAnalysisModel(n_components).fit(y_train[:, sl])
                reduced = PrincipalComponentAnalysisModel.reduce_data(y_train[:, sl], pca)
                roundtrip[:, sl] = PrincipalComponentAnalysisModel.reconstruct_data(
                    PrincipalComponentAnalysisModel.reduce_data(y_test[:, sl], pca), pca
                )
                network = KernelRidgeNetwork(
                    Hyperparameters.default_kernel_ridge(
                        len(x_train), kernel_gamma=gamma,
                        kernel_alpha=None if alpha == 0 else alpha,
                        kernel_alpha_selection="loo" if alpha == 0 else "fixed",
                    )
                )
                network.fit(x_train, reduced)
                prediction[:, sl] = PrincipalComponentAnalysisModel.reconstruct_data(
                    network.predict(x_test), pca
                )
                rms = lambda e: np.sqrt(np.mean((e - y_test[:, sl]) ** 2))
                print(f"  gamma={gamma:<5} alpha={alpha:<6} {name:<6} K={n_components:<3} rms error: PCA only "
                      f"{rms(roundtrip[:, sl]):9.2e}, + KRR {rms(prediction[:, sl]):9.2e}"
                      f"   (rms of the target {np.sqrt(np.mean((y_test[:, sl] - y_train[:, sl].mean(0)) ** 2)):.2e})")
            report(f"gamma={gamma} alpha={alpha}: PCA only", from_targets(roundtrip, omega_test, eta(x_test)), state_test)
            report(f"gamma={gamma} alpha={alpha}: PCA+KRR", from_targets(prediction, omega_test, eta(x_test)), state_test)

    if args.stage == "symmetry":
        # rotating both spins about z: L_x + i L_y rotates, alpha - gamma does not
        b = draw_binaries(4, 1)
        features, phi_1 = canonical_features(b)
        canonical, *_ = integrate(
            canonical_binaries(features), lowest_omega(args.f_min), args.omega_cut, args.n_steps, args.n_grid
        )
        rotated, *_ = integrate(b, lowest_omega(args.f_min), args.omega_cut, args.n_steps, args.n_grid)
        lxy = rotated[..., 6] + 1j * rotated[..., 7]
        lxy_c = (canonical[..., 6] + 1j * canonical[..., 7]) * np.exp(1j * phi_1)[:, None]
        print("max |Lxy - e^{i phi} Lxy_canonical| =", np.abs(lxy - lxy_c).max())
        print("max |Lz diff|                       =", np.abs(rotated[..., 8] - canonical[..., 8]).max())
        print("max |(alpha-gamma) diff|            =", np.abs(rotated[..., 9] - canonical[..., 9]).max())
        print("alpha-gamma range:", canonical[..., 9].min(), canonical[..., 9].max())


if __name__ == "__main__":
    main()
