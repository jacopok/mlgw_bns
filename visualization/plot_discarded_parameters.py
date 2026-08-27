"""Corner plot marking which regions of parameter space have a
non-positive EOB amplitude somewhere in the band.

For the (2,1) and (3,3) modes this used to make `generate_residuals`
return `None`, and the parameter set was skipped during training, since a
sign flip in the amplitude causes a pi phase jump the NN cannot learn.
`generate_residuals` no longer discards these --- it is not None-safe
against them --- so this script instead calls
`effective_one_body_waveform` directly and flags a parameter set whenever
its EOB amplitude dips to or below zero anywhere in the band, i.e. what
would previously have been discarded. It then plots the result as a
corner plot over the five intrinsic parameters (q, Lambda_1, Lambda_2,
chi_1, chi_2).

Run with: python visualization/plot_discarded_parameters.py
"""

import numpy as np
import matplotlib.pyplot as plt

from mlgw_bns.dataset_generation import Dataset
from mlgw_bns.higher_order_modes import Mode, teob_mode_generator_factory

MODES = [Mode(2, 1), Mode(3, 3)]
N_WAVEFORMS = 2000

PARAM_LABELS = [r"$q$", r"$\Lambda_1$", r"$\Lambda_2$", r"$\chi_1$", r"$\chi_2$"]

dataset = Dataset(initial_frequency_hz=20.0, srate_hz=4096.0)

# Same distribution used when generating the training dataset for a new
# ModeModel: uniform draws over `dataset.parameter_ranges`.
parameter_generator = dataset.make_parameter_generator()
params_list = [next(parameter_generator) for _ in range(N_WAVEFORMS)]
samples = np.array([params.array for params in params_list])

f_hz = np.arange(20.0, 2048.0, 0.5)
f_natural = dataset.hz_to_natural_units(f_hz)


def corner_plot(samples: np.ndarray, discarded: np.ndarray, title: str, outfile: str) -> None:
    n_params = samples.shape[1]
    fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))

    kept = ~discarded

    for row in range(n_params):
        for col in range(n_params):
            ax = axes[row, col]

            if col > row:
                ax.axis("off")
                continue

            if row == col:
                bins = np.histogram_bin_edges(samples[:, col], bins=30)
                ax.hist(samples[kept, col], bins=bins, color="0.6", alpha=0.7, label="kept")
                ax.hist(samples[discarded, col], bins=bins, color="crimson", alpha=0.8, label="would be discarded")
                ax.set_yticks([])
            else:
                ax.scatter(samples[kept, col], samples[kept, row], s=4, color="0.6", alpha=0.5, linewidths=0)
                ax.scatter(
                    samples[discarded, col],
                    samples[discarded, row],
                    s=8,
                    color="crimson",
                    alpha=0.9,
                    linewidths=0,
                )

            if row == n_params - 1:
                ax.set_xlabel(PARAM_LABELS[col])
            else:
                ax.set_xticklabels([])
            if col == 0 and row != 0:
                ax.set_ylabel(PARAM_LABELS[row])
            elif col == 0 and row == 0:
                pass
            else:
                ax.set_yticklabels([])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 1].legend(handles, labels, loc="center", fontsize=9, frameon=False)

    n_discarded = int(discarded.sum())
    fig.suptitle(f"{title}\n{n_discarded}/{len(discarded)} parameter sets discarded")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Saved plot to {outfile} ({n_discarded}/{len(discarded)} discarded)")


for mode in MODES:
    generator = teob_mode_generator_factory(mode)

    discarded = np.zeros(N_WAVEFORMS, dtype=bool)
    for i, params in enumerate(params_list):
        _, amplitude_eob, _ = generator.effective_one_body_waveform(params, f_natural)
        discarded[i] = bool(np.any(amplitude_eob <= 0))

    corner_plot(
        samples,
        discarded,
        title=rf"Would-be-discarded parameter sets for $(\ell, m) = ({mode.l}, {mode.m})$",
        outfile=f"discarded_params_l{mode.l}_m{mode.m}.png",
    )

# plt.show()
