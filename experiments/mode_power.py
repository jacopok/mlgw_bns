r"""How much each mode is worth, and therefore how much its error costs.

The per-mode sweeps in :mod:`experiments.run_sweep` score every mode on
its own, against its own EOB ground truth. That is the right way to
compare *pipelines*, but it says nothing about how much a mode's error
matters once the modes are summed: a mismatch of 0.4 in a mode carrying
:math:`10^{-6}` of the signal power costs less than :math:`10^{-8}` in
the (2,2).

Writing the total as :math:`h = \sum_k h_k` with an error
:math:`\delta h_k` in each mode, and taking the modes to be
approximately orthogonal under the noise-weighted inner product --- they
oscillate at different rates, so this is good here --- the fractional
loss of signal-to-noise is

.. math::

    \mathcal{M} \simeq \sum_k w_k \mathcal{M}_k, \qquad
    w_k = \frac{\langle h_k, h_k \rangle}{\sum_j \langle h_j, h_j \rangle}

so a mode's own mismatch enters weighted by its share of the power. This
measures those shares over the same validation binaries the sweeps use.
All modes share ``VALIDATION_SEED``, so their cached validation sets are
the same parameters in the same order and compare row by row.

The weights also carry a factor :math:`|{}_{-2}Y_{\ell m}(\iota)|^2` that
depends on the inclination; angle-averaging over the sky makes every one
of those integrate to the same constant, so what is reported here --- the
intrinsic power ratio --- is the angle-averaged answer, and the
inclination only redistributes it.

Run with: python -m experiments.mode_power
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from mlgw_bns.dataset_generation import Dataset
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model_validation import ValidateModel

from . import cache as cache_module

MODES = [Mode(2, 2), Mode(2, 1), Mode(3, 3)]


def mode_power(mode: Mode, n_train: int) -> tuple[np.ndarray, np.ndarray]:
    r"""PSD-weighted power :math:`\int |A|^2 / S_n \, df` per validation binary.

    Returns the powers and the mass ratios. The overall constant --- the
    factor of 4, the distance, the harmonic normalisation --- is dropped,
    since every use below is a ratio between modes.
    """
    cache = cache_module.load(mode, n_train)
    indices = cache_module.downsampling_indices(cache)

    amplitudes = (
        cache["validation_pn_amplitudes"] * cache["validation_amplitude_residuals"]
    )

    dataset = Dataset(
        initial_frequency_hz=cache_module.INITIAL_FREQUENCY_HZ,
        srate_hz=cache_module.SRATE_HZ,
    )
    frequencies_hz = dataset.natural_units_to_hz(
        dataset.frequencies[indices.amplitude_indices]
    )

    validator = ValidateModel(cache_module.make_mode_model(mode))
    # The nodes reach a hair below the first tabulated PSD frequency;
    # clipping there rather than extrapolating is harmless, since the ET
    # PSD is enormous at 3.6 Hz and the band contributes nothing.
    psd = validator.psd_at_frequencies(
        np.clip(frequencies_hz, validator.frequencies[0], validator.frequencies[-1])
    )

    power = np.trapezoid(amplitudes**2 / psd, frequencies_hz, axis=1)
    return power, cache["validation_parameters"][:, 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=8192)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    powers = {}
    mass_ratios = None
    for mode in MODES:
        powers[mode], mass_ratios = mode_power(mode, args.n_train)

    assert mass_ratios is not None
    q = mass_ratios
    total = sum(powers[mode] for mode in MODES)

    print(f"{len(q)} validation binaries, ET PSD, 5 Hz to merger")
    print()
    print("Share of the noise-weighted power, w_k (angle-averaged):")
    print(
        f"{'mode':>7}  {'median':>10}  {'q < 1.15':>10}  {'q > 1.5':>10}  {'max':>10}"
    )
    for mode in MODES:
        w = powers[mode] / total
        print(
            f"  ({mode.l},{mode.m})  {np.median(w):10.3e}  "
            f"{np.median(w[q < 1.15]):10.3e}  {np.median(w[q > 1.5]):10.3e}  "
            f"{w.max():10.3e}"
        )

    print()
    print("Cost of the measured per-mode mismatches, w_k * M_k, per binary.")
    print("`M_k` is read back from the sweep results, so this is what each")
    print("mode's error is actually worth in a summed waveform.")
    print()
    print(
        f"{'mode':>7}  {'configuration':<24}  {'median cost':>12}  "
        f"{'worst cost':>12}"
    )
    costs: dict[str, np.ndarray] = {}
    for mode in MODES:
        w = powers[mode] / total
        for label, mismatches in _sweep_mismatches(mode, args.n_train).items():
            if len(mismatches) != len(w):
                continue
            cost = w * mismatches
            costs[label] = costs.get(label, np.zeros_like(cost)) + cost
            print(
                f"  ({mode.l},{mode.m})  {label:<24}  {np.median(cost):12.3e}  "
                f"{cost.max():12.3e}"
            )

    print()
    print("Summed over the three modes --- the full-waveform mismatch:")
    print(f"{'configuration':<26}  {'median':>12}  {'p90':>12}  {'worst':>12}")
    for label, cost in costs.items():
        print(
            f"  {label:<24}  {np.median(cost):12.3e}  "
            f"{np.percentile(cost, 90):12.3e}  {cost.max():12.3e}"
        )


#: The rows of the per-mode sweeps worth converting into a cost, and the
#: short names to print them under.
_WANTED = {
    "regressor_kwargs=batch_size:5": "shipped",
    "baseline": "batch-size fixed",
}


def _sweep_mismatches(mode: Mode, n_train: int) -> dict[str, np.ndarray]:
    """Per-binary mismatches for a few configurations, from the sweep output.

    Returns an empty dict when the relevant sweep has not been run, so
    that the power table above is still printed.
    """
    import json
    from pathlib import Path

    results = Path(__file__).resolve().parent / "results"
    if mode == Mode(2, 2):
        path = results / f"production_baseline_l2_m2_n{n_train}.json"
    else:
        path = results / f"per_mode_l{mode.l}_m{mode.m}_n{n_train}.json"
    if not path.exists():
        return {}

    rows = json.loads(path.read_text())
    out = {}
    for row in rows:
        label = row["label"]
        if label in _WANTED:
            out[_WANTED[label]] = np.asarray(row["mismatches"])
    # whichever *trained* configuration scored best; the rows labelled
    # "[PCA floor]" skip the regression entirely and are not achievable.
    trained = [row for row in rows if "[PCA floor]" not in row["label"]]
    if trained:
        best = min(trained, key=lambda row: row["median"])
        out["best found"] = np.asarray(best["mismatches"])
    return out


if __name__ == "__main__":
    main()
