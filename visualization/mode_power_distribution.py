r"""Distribution of each mode's share of the total power.

Surrogate-only (no EOB calls at all): for a sample of waveforms drawn
with random intrinsic parameters, inclination and coalescence phase,
computes each mode's PSD-weighted power as a fraction of the total
summed power, via :meth:`Model.predict_modes_dict`. The point is to see
which modes matter at all across the parameter space, not to validate
the surrogate against anything --- there is no notion of "truth" here.

Two scenarios are compared side by side:

1. the full trained parameter space;
2. restricted to :math:`q < 1.2` and :math:`\Lambda < 1000` for both
   stars --- comparable-mass, low-deformability binaries, where the
   post-Newtonian expectation is that odd-``m`` modes (suppressed by
   the mass asymmetry) matter much less.

Run with: python visualization/mode_power_distribution.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_model import (  # noqa: E402  (sys.path must be set up first)
    DISTANCE_MPC,
    MODEL_FILENAME,
    TOTAL_MASS,
    load_model,
)

from mlgw_bns.data_management import ParameterRanges
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.mode_model import ParametersWithExtrinsic
from mlgw_bns.model import Model
from mlgw_bns.model_validation import ValidateModel

OUTPUT_PREFIX = "mode_power_distribution"

N_WAVEFORMS = 10000
SEED = 4242
ORIENTATION_SEED = 6060

#: The "comparable mass, low deformability" restricted scenario.
RESTRICTED_Q_MAX = 1.2
RESTRICTED_LAMBDA_MAX = 1000.0


def _restricted_parameter_ranges(full_ranges: ParameterRanges) -> ParameterRanges:
    """``full_ranges`` narrowed to :math:`q < 1.2`, :math:`\\Lambda_{1,2} < 1000`."""
    return ParameterRanges(
        mass_range=full_ranges.mass_range,
        q_range=(full_ranges.q_range[0], min(full_ranges.q_range[1], RESTRICTED_Q_MAX)),
        lambda1_range=(
            full_ranges.lambda1_range[0],
            min(full_ranges.lambda1_range[1], RESTRICTED_LAMBDA_MAX),
        ),
        lambda2_range=(
            full_ranges.lambda2_range[0],
            min(full_ranges.lambda2_range[1], RESTRICTED_LAMBDA_MAX),
        ),
        chi1_range=full_ranges.chi1_range,
        chi2_range=full_ranges.chi2_range,
    )


def _mode_power(complex_waveform: np.ndarray, frequencies: np.ndarray, psd_values: np.ndarray) -> float:
    r"""PSD-weighted power :math:`\int |h(f)|^2 / S(f)\, df`, over ``h``'s own support.

    Different modes start at different frequencies (an :math:`(\ell, m)`
    mode is zero below :math:`(m/2) f_0`), so each is integrated only
    where it is actually nonzero rather than over a common band.
    """
    support = np.abs(complex_waveform) > 0
    if support.sum() < 2:
        return 0.0
    return float(
        np.trapezoid(
            np.abs(complex_waveform[support]) ** 2 / psd_values[support],
            x=frequencies[support],
        )
    )


def sample_mode_power_fractions(
    model: Model,
    parameter_ranges: ParameterRanges,
    n_waveforms: int = N_WAVEFORMS,
    seed: int = SEED,
    orientation_seed: int = ORIENTATION_SEED,
) -> dict:
    r"""Per-mode power fraction of the total, over random draws.

    Each draw samples intrinsic parameters from ``parameter_ranges``
    (via the model's own ``parameter_generator_class``, so e.g. the
    default :class:`~mlgw_bns.dataset_generation.UniformParameterGenerator`
    for ``hom7_big``) plus an isotropic inclination
    (:math:`\cos\iota \sim \mathrm{Uniform}(-1, 1)`) and a coalescence
    phase uniform on :math:`[0, 2\pi)`, and calls
    :meth:`Model.predict_modes_dict` once --- no EOB involved at all.

    Each mode's power is reported as a fraction of the *total* summed
    power (:math:`\sum_{\ell m} P_{\ell m}`), not relative to (2,2)
    specifically --- with (2,2) overwhelmingly dominant this changes
    the numbers very little, but it's the more standard quantity (also
    what ``validate_model.py``'s ``power_fractions`` reports).

    Returns
    -------
    dict[Mode, np.ndarray]
        One array of power fractions per mode in ``model.modes``, over
        the successful draws (a draw is skipped if the total power is
        zero, or if any mode's prediction is non-finite).
    """
    dataset = model.dataset
    generator = dataset.parameter_generator_class(
        parameter_ranges=parameter_ranges, dataset=dataset, seed=seed
    )
    orientation_rng = np.random.default_rng(orientation_seed)

    reference_model = model.mode_models[Mode(2, 2)]
    validator = ValidateModel(reference_model)
    frequencies = validator.frequencies
    psd_values = validator.psd_values

    fractions = {mode: [] for mode in model.modes}

    with tqdm(total=n_waveforms, unit="waveform") as pbar:
        while pbar.n < n_waveforms:
            intrinsic = next(generator)
            iota = np.arccos(orientation_rng.uniform(-1.0, 1.0))
            phi_c = orientation_rng.uniform(0.0, 2 * np.pi)
            params = ParametersWithExtrinsic(
                mass_ratio=intrinsic.mass_ratio,
                lambda_1=intrinsic.lambda_1,
                lambda_2=intrinsic.lambda_2,
                chi_1=intrinsic.chi_1,
                chi_2=intrinsic.chi_2,
                distance_mpc=DISTANCE_MPC,
                inclination=iota,
                reference_phase=phi_c,
                total_mass=TOTAL_MASS,
            )
            try:
                modes_dict = model.predict_modes_dict(frequencies, params)
            except Exception:  # pragma: no cover - occasional NN/PN edge cases
                continue
            if not all(np.all(np.isfinite(v)) for v in modes_dict.values()):
                continue

            powers = {
                mode: _mode_power(modes_dict[(mode.l, mode.m)], frequencies, psd_values)
                for mode in model.modes
            }
            total_power = sum(powers.values())
            if total_power <= 0:
                continue

            for mode in model.modes:
                fractions[mode].append(powers[mode] / total_power)
            pbar.update(1)

    return {mode: np.array(values) for mode, values in fractions.items()}


def plot_mode_power_distribution(results_by_scenario: dict, modes: list) -> None:
    r"""Grouped boxplot of power fraction vs. :math:`m`, coloured by :math:`\ell`.

    ``results_by_scenario`` maps a scenario label to the output of
    :func:`sample_mode_power_fractions`; one panel per scenario, sharing
    a log-scaled y-axis. Within each panel, modes are grouped on the
    x-axis by their azimuthal number :math:`m`; modes sharing an
    :math:`m` (e.g. (2,2) and (3,2)) are offset side by side and
    coloured by :math:`\ell`. Boxes span the 25-75 interquartile range;
    whiskers extend to the 5th/95th percentiles.
    """
    modes_by_m: dict = {}
    for mode in modes:
        modes_by_m.setdefault(mode.m, []).append(mode)
    for m in modes_by_m:
        modes_by_m[m].sort(key=lambda mode: mode.l)

    m_values = sorted(modes_by_m.keys())
    l_values = sorted({mode.l for mode in modes})
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    l_colors = {l: color_cycle[i % len(color_cycle)] for i, l in enumerate(l_values)}

    n_scenarios = len(results_by_scenario)
    fig, axes = plt.subplots(
        1, n_scenarios, figsize=(5.5 * n_scenarios + 1.5, 6.5), sharey=True
    )
    if n_scenarios == 1:
        axes = [axes]

    # Boxes belonging to the same `m` are packed close together (small
    # offsets, narrow width) so they read as one group; the group itself
    # sits far enough from its neighbours (integer spacing) and is set
    # off by a dashed vertical line at each half-integer boundary, so a
    # box can't be mistaken for belonging to the next `m` over.
    box_kwargs = dict(whis=[5, 95], showfliers=False, widths=0.1, patch_artist=True)

    for ax, (scenario_label, fractions) in zip(axes, results_by_scenario.items()):
        for m in m_values:
            modes_here = modes_by_m[m]
            n = len(modes_here)
            offsets = np.linspace(-0.11, 0.11, n) if n > 1 else [0.0]
            for mode, offset in zip(modes_here, offsets):
                data = fractions[mode]
                data = data[data > 0]
                if len(data) < 2:
                    continue
                bp = ax.boxplot([data], positions=[m + offset], **box_kwargs)
                for patch in bp["boxes"]:
                    patch.set_facecolor(l_colors[mode.l])
                    patch.set_alpha(0.6)
                for median in bp["medians"]:
                    median.set_color("black")

        for boundary in np.arange(min(m_values), max(m_values)) + 0.5:
            ax.axvline(boundary, color="gray", lw=0.6, linestyle=":", zorder=0)

        ax.set_yscale("log")
        ax.set_xticks(m_values)
        ax.set_xticklabels([f"${m}$" for m in m_values])
        ax.set_xlim(min(m_values) - 0.5, max(m_values) + 0.5)
        ax.set_xlabel("$m$")
        ax.set_title(scenario_label)
        ax.grid(True, which="both", axis="y", lw=0.3)

    axes[0].set_ylabel(
        r"Power fraction, $(h_{\ell m} | h_{\ell m}) \,\big/\, "
        r"\sum_{\ell' m'} (h_{\ell' m'} | h_{\ell' m'})$"
    )

    legend_handles = [
        plt.Line2D(
            [0], [0], marker="s", linestyle="", markersize=12,
            markerfacecolor=l_colors[l], markeredgecolor="black", alpha=0.6,
            label=rf"$\ell = {l}$",
        )
        for l in l_values
    ]
    fig.legend(handles=legend_handles, loc="upper right")
    fig.suptitle(
        rf"Per-mode power fraction of the total, {N_WAVEFORMS} surrogate "
        "draws (random intrinsic parameters, inclination, coalescence phase)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    outfile = f"{OUTPUT_PREFIX}.png"
    fig.savefig(outfile, dpi=150)
    print(f"Saved plot to {outfile}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=MODEL_FILENAME, help="base filename of the model to use"
    )
    parser.add_argument(
        "--prefix", default=None, help="prefix for the output figure; defaults to the model's name"
    )
    parser.add_argument("--n-waveforms", type=int, default=N_WAVEFORMS)
    args = parser.parse_args()

    if args.n_waveforms != N_WAVEFORMS:
        N_WAVEFORMS = args.n_waveforms

    OUTPUT_PREFIX = (
        args.prefix
        if args.prefix is not None
        else f"{os.path.basename(args.model.rstrip('/')) or OUTPUT_PREFIX}_mode_power_distribution"
    )

    model = load_model(args.model)
    full_ranges = model.dataset.parameter_ranges
    restricted_ranges = _restricted_parameter_ranges(full_ranges)

    print(f"Sampling {N_WAVEFORMS} waveforms over the full parameter space...")
    full_fractions = sample_mode_power_fractions(model, full_ranges, N_WAVEFORMS, SEED, ORIENTATION_SEED)

    print(
        f"Sampling {N_WAVEFORMS} waveforms restricted to "
        f"q < {RESTRICTED_Q_MAX}, Lambda < {RESTRICTED_LAMBDA_MAX}..."
    )
    restricted_fractions = sample_mode_power_fractions(
        model, restricted_ranges, N_WAVEFORMS, SEED + 1, ORIENTATION_SEED + 1
    )

    plot_mode_power_distribution(
        {
            "Full parameter space": full_fractions,
            rf"$q < {RESTRICTED_Q_MAX}$, $\Lambda < {RESTRICTED_LAMBDA_MAX:.0f}$": restricted_fractions,
        },
        model.modes,
    )
