"""Mismatch table for the power-weighting A/B, on one shared validation set.

``visualization/validate_model.py`` also draws residual and predictor
figures, which cost ~1040 extra EOB waveforms per model and say nothing
about the regressor fit. This runs only the two mismatch distributions
that decide whether power weighting ships, for several models in one go.

Both arms draw their validation parameters from the same seeds, so the
comparison is paired: the same binaries, the same EOB ground truth.

Run with::

    python compare_power_weighting.py                       # pw_off vs pw_on
    python compare_power_weighting.py --models default_hom;pw_on -n 200
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "visualization"))

import validate_model as vm  # noqa: E402


def summarise(name: str, n: int) -> dict:
    """Per-mode and full-waveform mismatch medians for one model."""
    vm.N_MISMATCH_WAVEFORMS = n
    vm.N_FULL_WAVEFORM_MISMATCHES = n

    model = vm.load_model(name)
    by_mode = vm.per_mode_mismatches(model)
    full, full_no_opt, power = vm.full_waveform_mismatches(model)

    result = {"full": full, "full_no_opt": full_no_opt, "power": power}
    for mode in vm.MODES:
        optimised, predicted = by_mode[mode]
        result[mode] = (optimised, predicted)
    return result


def ratio(new: float, old: float) -> str:
    if not np.isfinite(old) or old == 0 or not np.isfinite(new) or new == 0:
        return "    n/a"
    factor = old / new
    return f"{factor:6.2f}x" if factor >= 1 else f"1/{1 / factor:<5.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=str, default="pw_off;pw_on")
    parser.add_argument("-n", "--n-mismatches", type=int, default=300)
    args = parser.parse_args()

    names = args.models.split(";")
    results = {}
    for name in names:
        print(f"\n=== {name} ===", flush=True)
        results[name] = summarise(name, args.n_mismatches)

    baseline = names[0]

    def row(label, values):
        cells = "".join(f"{v:>13.3e}" for v in values)
        gain = ratio(values[-1], values[0]) if len(values) > 1 else ""
        print(f"  {label:<28}{cells}   {gain}")

    print("\n" + "=" * 78)
    print(f"  median mismatch over {args.n_mismatches} waveforms "
          f"(last column: {names[0]} -> {names[-1]})")
    print("=" * 78)
    header = "".join(f"{n:>13}" for n in names)
    print(f"  {'':<28}{header}")

    for mode in vm.MODES:
        share = np.median(results[names[0]]["power"].get(mode, np.array([np.nan])))
        print(f"\n  ({mode.l},{mode.m})   power share {share:.3e}")
        row("optimised", [np.median(results[n][mode][0]) for n in names])
        row("predicted-shift", [np.median(results[n][mode][1]) for n in names])
        row("predicted-shift 90th pct",
            [np.percentile(results[n][mode][1], 90) for n in names])

    print("\n  full waveform")
    row("optimised (median)", [np.median(results[n]["full"]) for n in names])
    row("optimised (worst)", [np.max(results[n]["full"]) for n in names])
    row("not optimised (median)",
        [np.median(results[n]["full_no_opt"]) for n in names])
    print()


if __name__ == "__main__":
    main()
