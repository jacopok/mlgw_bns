"""Generate many TEOBResumS waveforms for the (2,1), (2,2), (3,3), (4,4) modes,
with parameters drawn from the same distribution used to build the training
dataset for a new model, and plot amplitude and phase for all of them.

Run with: python visualization/plot_teob_modes.py
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from mlgw_bns.dataset_generation import Dataset
from mlgw_bns.higher_order_modes import Mode, teob_mode_generator_factory

MODES = [Mode(2, 1), Mode(2, 2), Mode(3, 3), Mode(4, 4)]
N_WAVEFORMS = 20

dataset = Dataset(initial_frequency_hz=20.0, srate_hz=4096.0)

# Same distribution used when generating the training dataset for a new
# Model: uniform draws over `dataset.parameter_ranges`.
parameter_generator = dataset.make_parameter_generator()
params_list = [next(parameter_generator) for _ in range(N_WAVEFORMS)]

f_hz = np.arange(20.0, 2048.0, 0.1)
f_natural = dataset.hz_to_natural_units(f_hz)

fig, axes = plt.subplots(2, len(MODES), figsize=(16, 6), sharex=True)

cmap = matplotlib.colormaps["viridis"]
q_min, q_max = dataset.parameter_ranges.q_range
colors = [cmap((p.mass_ratio - q_min) / (q_max - q_min)) for p in params_list]

for i, mode in enumerate(MODES):
    generator = teob_mode_generator_factory(mode)

    for params, color in tqdm(zip(params_list, colors)):
        f_spa, amplitude, phase = generator.effective_one_body_waveform(params, f_natural)

        axes[0, i].plot(f_spa, amplitude, color=color, alpha=0.5, linewidth=0.8)
        axes[0, i].plot(f_spa, -amplitude, color=color, alpha=0.5, linewidth=3.)
        axes[1, i].plot(f_spa, phase, color=color, alpha=0.5, linewidth=0.8)

    axes[0, i].set_yscale("log")
    axes[0, i].set_title(rf"$(\ell, m) = ({mode.l}, {mode.m})$")
    axes[1, i].set_xlabel(r"$Mf$")

axes[0, 0].set_ylabel("Amplitude")
axes[1, 0].set_ylabel("Phase [rad]")

for ax_row in axes:
    for ax in ax_row:
        ax.grid(True)
        ax.set_xscale('log')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=q_min, vmax=q_max))
fig.colorbar(sm, ax=axes, label="Mass ratio $q$", pad=0.01)

fig.suptitle(
    f"TEOBResumS modes, {N_WAVEFORMS} waveforms drawn from the "
    "training parameter distribution"
)

outfile = "teob_modes_21_22_33_44.png"
fig.savefig(outfile, dpi=150)
print(f"Saved plot to {outfile}")
plt.show()
