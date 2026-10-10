r"""What the tail of the regressed-precession mismatches correlates with.

Ranks the validation binaries of :mod:`precession_regression_study` by
waveform mismatch and compares them with candidate indicators: the elliptic
parameter :math:`m` of the nutation (how close the binary is to a
separatrix between precession morphologies, where the period diverges and
the two carriers no longer hold the nutation), and how close it is to the
"up-down" configuration (heavier spin along :math:`\hat{L}`, lighter
against), which near equal masses is unstable to precession over the whole
band (Gerosa et al. 2015, arXiv:1506.09116)::

    python visualization/precession_tail_diagnostic.py \
        --regressor visualization/precession_beat_ab/elliptic.joblib \
        --data visualization/precession_beat_ab/validation.npz --n 256
"""

from __future__ import annotations

import argparse

import numpy as np
from precession_regression_study import (
    chi_p,
    function_errors,
    load,
    nutation_parameters,
    waveform_mismatches,
)
from scipy.stats import spearmanr

import mlgw_bns.precession_regression as pr


def up_down_onset(intrinsic) -> np.ndarray:
    r""":math:`r_{\rm ud+} / M` of Gerosa et al. (2015), below which the
    up-down configuration is unstable, for the aligned components of the
    spins; zero where the binary is not up-down (heavier spin along
    :math:`\hat{L}`, lighter against). Their :math:`q \leq 1` is our
    :math:`1 / q`."""
    q = 1.0 / intrinsic[:, 0]
    up, down = intrinsic[:, 5], -intrinsic[:, 8]
    is_up_down = (up > 0) & (down > 0)
    onset = (np.sqrt(np.abs(up)) + np.sqrt(q * np.abs(down))) ** 4 / np.maximum((1 - q) ** 2, 1e-12)
    return np.where(is_up_down, onset, 0.0)


def envelope_content(grid: pr.AngleGrid, data, n: int) -> np.ndarray:
    r"""What is left in the *fitted* (not regressed) envelopes :math:`c_1,
    c_2` of :math:`\zeta`, for the first ``n`` binaries of ``data``: the
    number of beat cycles over the band, how many times each envelope's
    phase winds, signed (a carrier phase error: the same sign for the two
    if the carriers' mean is off, opposite if their beat is), and the fraction of its power that
    oscillates at the beat or faster (the nutation harmonics beyond the two
    carriers); ``(n, 5)``."""
    rows = []
    for i in range(n):
        frame = pr.reference_frame(np, grid, data["intrinsic"][i], data["omega_reference"][i])
        phases, rates = pr.carrier_table(np, grid, frame)
        x = data["x"][i]
        phase = pr._hermite(np, *grid.x_range, phases, rates, x)
        beat = phase[0] - phase[1]
        cell, weights = pr._bspline(np, grid, x)
        envelopes = sum(data["coefficients"][i][:, cell + r] * weights[..., r] for r in range(4))
        n_beat = abs(beat[-1] - beat[0]) / (2 * np.pi)
        row = [n_beat]
        windings, fractions = [], []
        for envelope in envelopes[1:3]:
            windings.append(np.diff(np.unwrap(np.angle(envelope))).sum() / (2 * np.pi))
            # resampled uniformly in the beat phase: power at >= 1/2 cycle per beat cycle
            order = np.argsort(beat)
            uniform = np.linspace(beat[order][0], beat[order][-1], 4096)
            resampled = np.interp(uniform, beat[order], envelope.real[order]) + 1j * np.interp(
                uniform, beat[order], envelope.imag[order]
            )
            power = np.abs(np.fft.fft(resampled - resampled.mean())) ** 2
            cycles = np.abs(np.fft.fftfreq(len(uniform), d=(uniform[1] - uniform[0]) / (2 * np.pi)))
            fractions.append(power[cycles >= 0.5].sum() / max(power.sum(), 1e-300))
        rows.append(row + windings + fractions)
    return np.array(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regressor")
    parser.add_argument("--data", required=True)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--worst", type=int, default=15)
    parser.add_argument("--out", help="save the mismatches, errors and parameters m here (.npz)")
    parser.add_argument(
        "--envelopes", action="store_true",
        help="only analyse what the fitted envelopes hold, binned by max m (needs no regressor)",
    )
    args = parser.parse_args()

    data = load([args.data])
    if args.envelopes:
        n = args.n
        content = envelope_content(data["grid"], data, n)
        max_m = nutation_parameters(
            data["grid"], data["intrinsic"][:n], data["omega_reference"][:n]
        ).max(axis=1)
        print("medians by max m: beat cycles, windings of c_1, c_2, power at the beat or faster in c_1, c_2")
        edges = [0.0, 0.3, 0.6, 0.8, 0.9, 1.0]
        for lo, hi in zip(edges[:-1], edges[1:]):
            cell = (max_m >= lo) & (max_m < hi)
            if cell.any():
                values = np.median(np.abs(content[cell]), axis=0)
                winding = cell & (np.abs(content[:, 1]) > 0.25) & (np.abs(content[:, 2]) > 0.25)
                same = np.mean(np.sign(content[winding, 1]) == np.sign(content[winding, 2])) if winding.any() else np.nan
                print(f"  m {lo}-{hi} ({cell.sum():3d}): {values[0]:6.1f}  {values[1]:5.2f} {values[2]:5.2f}  "
                      f"{values[3]:.1e} {values[4]:.1e}  same-sign windings {same:.0%} of {winding.sum()}")
        for k, name in enumerate(["beat cycles", "winding c_1", "winding c_2", "beat power c_1", "beat power c_2"]):
            print(f"  rank correlation of max m with {name}: {spearmanr(max_m, np.abs(content[:, k]))[0]:+.2f}")
        return

    regressor = pr.PrecessionRegressor.load(args.regressor)
    n = args.n
    intrinsic, omega_reference = data["intrinsic"][:n], data["omega_reference"][:n]

    mismatches, params, _ = waveform_mismatches(regressor, data, n)
    errors = function_errors(regressor, data, n)
    m = nutation_parameters(regressor.grid, intrinsic, omega_reference)
    if args.out:
        np.savez(args.out, mismatches=mismatches, errors=errors, m=m, intrinsic=intrinsic)
    onset = up_down_onset(intrinsic)
    # separation at the start of the band, r/M ~ (M Omega)^(-2/3)
    r_start = regressor.grid.omega_min ** (-2.0 / 3.0)

    indicators = {
        "q - 1": intrinsic[:, 0] - 1.0,
        "max m": m.max(axis=1),
        "-log(1 - max m)": -np.log1p(-m.max(axis=1)),
        "up-down (r_ud+ > r_start)": (onset > r_start).astype(float),
        "chi_p": chi_p(intrinsic),
        "|chi_1z - chi_2z|": np.abs(intrinsic[:, 5] - intrinsic[:, 8]),
    }
    print(f"{n} binaries, mismatch median {np.median(mismatches):.1e}, 90% "
          f"{np.percentile(mismatches, 90):.1e}, max {mismatches.max():.1e}")
    print("Spearman rank correlation with the mismatch:")
    for name, values in indicators.items():
        rho, p = spearmanr(values, mismatches)
        print(f"  {name:28s} {rho:+.2f} (p {p:.1e})")

    max_m = m.max(axis=1)
    print("Spearman rank correlation with the largest errors below / above x = -10:")
    for name, values in [("max m", max_m), ("q - 1", intrinsic[:, 0] - 1.0)]:
        rhos = [spearmanr(values, errors[:, k])[0] for k in range(4)]
        print(f"  {name:8s} zeta {rhos[0]:+.2f} / {rhos[1]:+.2f}, G {rhos[2]:+.2f} / {rhos[3]:+.2f}")

    print("median mismatch (number of binaries) by q and max m:")
    m_edges = [0.0, 0.3, 0.6, 0.8, 1.0]
    for q_lo, q_hi in [(1.0, 1.2), (1.2, 1.5)]:
        in_q = (intrinsic[:, 0] >= q_lo) & (intrinsic[:, 0] < q_hi)
        rho = spearmanr(max_m[in_q], mismatches[in_q])[0]
        cells = []
        for lo, hi in zip(m_edges[:-1], m_edges[1:]):
            cell = in_q & (max_m >= lo) & (max_m < hi)
            cells.append(f"{np.median(mismatches[cell]):.1e} ({cell.sum()})" if cell.any() else "-")
        print(f"  q in [{q_lo}, {q_hi}): " + ", ".join(
            f"m {lo}-{hi}: {c}" for lo, hi, c in zip(m_edges[:-1], m_edges[1:], cells)
        ) + f"; rank correlation with max m {rho:+.2f}")

    near_equal = intrinsic[:, 0] < 1.2
    up_down = onset > r_start
    for name, mask in [
        ("q < 1.2, up-down", near_equal & up_down),
        ("q < 1.2, not up-down", near_equal & ~up_down),
        ("q >= 1.2", ~near_equal),
    ]:
        if mask.any():
            print(f"  {name:22s} {mask.sum():4d} binaries: median {np.median(mismatches[mask]):.1e}, "
                  f"> 1e-2: {np.mean(mismatches[mask] > 1e-2):.0%}")

    print(f"worst {args.worst}: mismatch, q, chi_1z, chi_2z, in-plane 1, 2, max m, up-down")
    for i in np.argsort(mismatches)[::-1][: args.worst]:
        row = intrinsic[i]
        print(f"  {mismatches[i]:.1e}  {row[0]:.3f}  {row[5]:+.2f}  {row[8]:+.2f}  "
              f"{np.hypot(row[3], row[4]):.2f}  {np.hypot(row[6], row[7]):.2f}  "
              f"{m[i].max():.3f}  {'yes' if up_down[i] else 'no'}")


if __name__ == "__main__":
    main()
