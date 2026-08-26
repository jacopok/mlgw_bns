"""Visualize the EOB/PN residuals for the pathological parameter sets that
`WaveformGenerator.generate_residuals` discards.

For the (2,1) and (3,3) modes, `generate_residuals` returns `None` whenever
the EOB amplitude goes non-positive somewhere in the band, since the
resulting pi phase jump is not something the NN can learn. This script
finds those discarded parameter sets and plots, for each of them, the raw
EOB amplitude (to show where it crosses zero), the amplitude residual
log(|A_eob| / |A_pn|), and the phase residual phi_eob - phi_pn -- i.e.
exactly what `generate_residuals` would have produced, had it not
discarded them.

Run with: python visualization/plot_discarded_residuals.py
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from mlgw_bns.dataset_generation import Dataset
from mlgw_bns.higher_order_modes import Mode, teob_mode_generator_factory

MODES = [Mode(2, 1), Mode(3, 3)]
N_WAVEFORMS = 2000
MAX_SHOWN = 25  # cap how many pathological waveforms to overlay per mode

dataset = Dataset(initial_frequency_hz=20.0, srate_hz=4096.0)

# Same distribution used when generating the training dataset for a new
# ModeModel: uniform draws over `dataset.parameter_ranges`.
parameter_generator = dataset.make_parameter_generator()
params_list = [next(parameter_generator) for _ in range(N_WAVEFORMS)]

f_hz = np.arange(20.0, 2048.0, 0.5)
f_natural = dataset.hz_to_natural_units(f_hz)

cmap = matplotlib.colormaps["viridis"]
q_min, q_max = dataset.parameter_ranges.q_range

for mode in MODES:
    generator = teob_mode_generator_factory(mode)

    # Same discard condition as `WaveformGenerator.generate_residuals`,
    # but keeping the pathological cases instead of dropping them.
    discarded = []
    for params in params_list:
        f_spa, amplitude_eob, phase_eob = generator.effective_one_body_waveform(params, f_natural)
        if np.any(amplitude_eob <= 0):
            discarded.append((params, f_spa, amplitude_eob, phase_eob))

    print(f"{mode}: {len(discarded)}/{N_WAVEFORMS} discarded")

    shown = discarded[:MAX_SHOWN]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    for params, f_spa, amplitude_eob, phase_eob in shown:
        color = cmap((params.mass_ratio - q_min) / (q_max - q_min))

        amplitude_pn = generator.post_newtonian_amplitude(params, f_spa)
        phase_pn = generator.post_newtonian_phase(params, f_spa)

        axes[0].plot(f_spa, amplitude_eob, color=color, alpha=0.7, linewidth=0.8)
        axes[1].plot(
            f_spa,
            np.log(np.abs(amplitude_eob) / np.abs(amplitude_pn)),
            color=color,
            alpha=0.7,
            linewidth=0.8,
        )
        axes[2].plot(f_spa, phase_eob - phase_pn, color=color, alpha=0.7, linewidth=0.8)

    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel(r"$A_{\rm EOB}$")
    axes[1].set_ylabel(r"$\log(A_{\rm EOB} / A_{\rm PN})$")
    axes[2].set_ylabel(r"$\phi_{\rm EOB} - \phi_{\rm PN}$ [rad]")
    axes[2].set_xlabel(r"$Mf$")

    for ax in axes:
        ax.grid(True)
        ax.set_xscale("log")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=q_min, vmax=q_max))
    fig.colorbar(sm, ax=axes, label="Mass ratio $q$", pad=0.02)

    fig.suptitle(
        rf"Pathological (discarded) waveforms for $(\ell, m) = ({mode.l}, {mode.m})$"
        f"\n{len(discarded)}/{N_WAVEFORMS} discarded, showing {len(shown)}"
    )

    outfile = f"discarded_residuals_l{mode.l}_m{mode.m}.png"
    fig.savefig(outfile, dpi=150)
    print(f"Saved plot to {outfile}")

plt.show()
