"""Per-binary unoptimised mismatch and merger-reference offsets of two models.

Reproduces the sixteen binaries of
``tests/test_model.py::test_default_model_merger_reference_matches_teobresums``
and, for the shipped default model and a second model (by default the
interim one extracted to ``visualization/interim_default_hom``), prints

- the unoptimised summed-waveform mismatch against TEOBResumS;
- the time and phase offsets of the (2,2) against TEOBResumS, from a
  linear fit of the (2,2) phase difference above 1 kHz;
- the time and phase offsets that maximise the summed-waveform overlap.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from test_model import reduced_range_parameter_generator  # noqa: E402

from mlgw_bns.higher_order_modes import Mode  # noqa: E402
from mlgw_bns.model import DEFAULT_MODES, Model  # noqa: E402
from mlgw_bns.mode_model import ParametersWithExtrinsic  # noqa: E402
from mlgw_bns.model_validation import ValidateModel  # noqa: E402


def binaries(model, n=16):
    generator = reduced_range_parameter_generator(model, seed=7)
    for _ in range(n):
        intrinsic = next(generator)
        yield ParametersWithExtrinsic(
            mass_ratio=intrinsic.mass_ratio,
            lambda_1=intrinsic.lambda_1,
            lambda_2=intrinsic.lambda_2,
            chi_1=intrinsic.chi_1,
            chi_2=intrinsic.chi_2,
            distance_mpc=100.0,
            inclination=1.0,
            total_mass=2.8,
        )


def analyse(model, frequencies, psd, truths, n):
    weight = np.gradient(frequencies) / psd
    time_grid = np.linspace(-2e-3, 2e-3, 4001)
    rows = []
    for params, true_modes in zip(binaries(model, n), truths):
        pred_modes = model.predict_modes_dict(frequencies, params)
        true = sum(true_modes.values())
        pred = sum(pred_modes.values())
        support = np.abs(true) > 0

        def inner(a, b):
            return np.sum((np.conj(a) * b * weight)[support])

        norm = np.sqrt(inner(true, true).real * inner(pred, pred).real)
        unopt = 1 - inner(true, pred).real / norm

        # (2,2) phase difference, fitted with a line near the top of the band
        top = support & (frequencies > 1000.0)
        dphi22 = np.unwrap(np.angle(pred_modes[(2, 2)][top] / true_modes[(2, 2)][top]))
        slope, intercept = np.polyfit(frequencies[top], dphi22, 1)
        dt22 = slope / (2 * np.pi)

        # time shift maximising the summed overlap (phase maximised analytically)
        c = np.conj(true) * pred * weight
        overlaps = np.array(
            [np.sum(c[support] * np.exp(2j * np.pi * frequencies[support] * t)) for t in time_grid]
        )
        best = np.argmax(np.abs(overlaps))
        max_mm = 1 - np.abs(overlaps[best]) / norm

        rows.append((params, unopt, max_mm, dt22, intercept, time_grid[best]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--other",
        default=str(Path(__file__).resolve().parent / "interim_default_hom" / "default_hom"),
        help="base filename of the model to compare against the default one",
    )
    parser.add_argument("-n", type=int, default=16, help="number of binaries")
    args = parser.parse_args()

    new = Model.default_for_testing()
    other = Model(modes=list(DEFAULT_MODES), filename=args.other)
    other.load()

    validator = ValidateModel(new.mode_models[Mode(2, 2)])
    frequencies = validator.frequencies
    truths = [new.get_teob_modes_dict(frequencies, p) for p in binaries(new, args.n)]

    results = {
        "new": analyse(new, frequencies, validator.psd_values, truths, args.n),
        "other": analyse(other, frequencies, validator.psd_values, truths, args.n),
    }

    header = f"{'i':>2} {'q':>5} {'L1':>6} {'L2':>6} {'chi1':>6} {'chi2':>6} | " + " | ".join(
        f"{name:>5}: {'unopt':>8} {'max_t':>8} {'dt22[us]':>9} {'dphi22':>7} {'t_best[us]':>10}"
        for name in results
    )
    print(header)
    for i, (row_new, row_other) in enumerate(zip(results["new"], results["other"])):
        p = row_new[0]
        line = f"{i:>2} {p.mass_ratio:5.3f} {p.lambda_1:6.0f} {p.lambda_2:6.0f} {p.chi_1:6.3f} {p.chi_2:6.3f} | "
        line += " | ".join(
            f"{'':>5}  {r[1]:8.1e} {r[2]:8.1e} {r[3] * 1e6:9.2f} {r[4]:7.3f} {r[5] * 1e6:10.2f}"
            for r in (row_new, row_other)
        )
        print(line)
    for name, rows in results.items():
        unopt = np.array([r[1] for r in rows])
        dt = np.abs([r[3] for r in rows]) * 1e6
        print(
            f"{name}: unoptimised median {np.median(unopt):.2e}, 90% {np.quantile(unopt, 0.9):.2e}, "
            f"max {np.max(unopt):.2e}, #>8e-2 {np.sum(unopt > 8e-2)}; "
            f"|dt22| median {np.median(dt):.1f} us, max {np.max(dt):.1f} us"
        )


if __name__ == "__main__":
    main()
