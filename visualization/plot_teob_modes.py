"""Generate many TEOBResumS waveforms for the (2,1), (2,2), (3,3), (4,4) modes,
with parameters drawn from the same distribution used to build the training
dataset for a new model, and plot amplitude and phase for all of them,
against the post-Newtonian amplitude the surrogate uses as its baseline.

The mass ratio is restricted to a narrow band just above equal mass in
order to show the (2,1) failure mode. That mode's amplitude carries a
factor which vanishes for equal masses, so near :math:`q = 1` it is small
enough that the spin-dependent contribution can cancel it at some
frequency: the EOB amplitude crosses zero and comes back with the
opposite sign, which is a physical :math:`\\pi` phase flip. Sampling the
full :math:`q \\in [1, 3]` produces these in only ~4% of draws, which is
too few to see; over `Q_RANGE` they are common.

The post-Newtonian baseline does not reproduce the crossing, so wherever
the EOB amplitude changes sign the ratio the model is trained on ---
:math:`A_{\\rm EOB} / A_{\\rm PN}` --- changes sign too, and the surrogate
has to represent a feature whose location moves with the parameters.

Run with: python visualization/plot_teob_modes.py
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from mlgw_bns.data_management import ParameterRanges
from mlgw_bns.dataset_generation import Dataset
from mlgw_bns.higher_order_modes import Mode, teob_mode_generator_factory

MODES = [
    Mode(2, 1),
    # Mode(2, 2),
    # Mode(3, 3),
    Mode(4, 4)
    ]
N_WAVEFORMS = 60

#: Just above equal mass, where the (2,1) sign changes live. Widen this
#: back to (1.0, 3.0) to recover the previous, unrestricted behaviour.
Q_RANGE = (1.0, 1.3)

dataset = Dataset(
    initial_frequency_hz=20.0,
    srate_hz=4096.0,
    parameter_ranges=ParameterRanges(q_range=Q_RANGE),
)

# Same distribution used when generating the training dataset for a new
# Model: uniform draws over `dataset.parameter_ranges`.
parameter_generator = dataset.make_parameter_generator(seed=2)
params_list = [next(parameter_generator) for _ in range(N_WAVEFORMS)]

f_hz = np.arange(20.0, 2048.0, 0.1)
f_natural = dataset.hz_to_natural_units(f_hz)

cmap = matplotlib.colormaps["viridis"]
q_min, q_max = dataset.parameter_ranges.q_range
colors = [cmap((p.mass_ratio - q_min) / (q_max - q_min)) for p in params_list]

# Collect first, plot second: the amplitude scale has to accommodate both
# signs, so it can only be chosen once all the curves are known.
curves: dict[Mode, list] = {}
for mode in MODES:
    generator = teob_mode_generator_factory(mode)
    curves[mode] = []

    for params, color in tqdm(zip(params_list, colors), desc=f"({mode.l},{mode.m})"):
        f_spa, amplitude, phase = generator.effective_one_body_waveform(
            params, f_natural
        )
        pn_amplitude = generator.post_newtonian_amplitude(params, f_spa)
        curves[mode].append((f_spa, amplitude, pn_amplitude, phase, color))

flipping = {
    mode: [np.any(amp < 0) for _, amp, _, _, _ in rows]
    for mode, rows in curves.items()
}

# Symmetric-log amplitudes: the amplitude spans decades, but the point of
# the plot is the sign, which a log axis cannot show. `linthresh` is set
# well below the typical amplitude so the crossings stay legible.
all_amplitudes = np.concatenate(
    [np.abs(amp) for rows in curves.values() for _, amp, _, _, _ in rows]
)
linthresh = np.percentile(all_amplitudes[all_amplitudes > 0], 1)

fig, axes = plt.subplots(2, len(MODES), figsize=(8 * len(MODES), 8), sharex=True,
                         squeeze=False)

for i, mode in enumerate(MODES):
    for (f_spa, amplitude, pn_amplitude, phase, color), flips in zip(
        curves[mode], flipping[mode]
    ):
        # The sign-changing waveforms are the subject of the figure, so
        # they are drawn solid and opaque against everything else.
        width, alpha = (1.6, 0.95) if flips else (0.8, 0.35)

        axes[0, i].plot(f_spa, amplitude, color=color, alpha=alpha, linewidth=width)
        axes[0, i].plot(
            f_spa, pn_amplitude, color=color, alpha=alpha * 0.8,
            linewidth=width * 0.8, linestyle="--",
        )
        axes[1, i].plot(f_spa, phase, color=color, alpha=alpha, linewidth=width)

    n_flip = sum(flipping[mode])
    axes[0, i].set_title(
        rf"$(\ell, m) = ({mode.l}, {mode.m})$"
        + f"\n{n_flip}/{len(curves[mode])} change sign"
    )
    axes[0, i].set_yscale("symlog", linthresh=linthresh)
    axes[0, i].axhline(0.0, color="black", linewidth=1.0)
    axes[1, i].set_xlabel(r"$Mf$")

axes[0, 0].set_ylabel("Amplitude (symlog)")
axes[1, 0].set_ylabel("Phase [rad]")

axes[0, 0].legend(
    handles=[
        Line2D([], [], color="black", linestyle="-", label="TEOBResumS"),
        Line2D([], [], color="black", linestyle="--", label="post-Newtonian"),
        Line2D([], [], color="black", linestyle="-", linewidth=1.6,
               label="sign change (opaque)"),
    ],
    loc="lower left",
    fontsize="small",
)

for ax_row in axes:
    for ax in ax_row:
        ax.grid(True)
        ax.set_xscale('log')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=q_min, vmax=q_max))
fig.colorbar(sm, ax=axes, label="Mass ratio $q$", pad=0.01)

fig.suptitle(
    f"TEOBResumS modes and their post-Newtonian baseline, {N_WAVEFORMS} waveforms "
    rf"with $q \in [{q_min}, {q_max}]$"
)

outfile = "teob_modes_21_22_33_44.png"
fig.savefig(outfile, dpi=150)
print(f"Saved plot to {outfile}")
for mode in MODES:
    print(f"  ({mode.l},{mode.m}): {sum(flipping[mode])}/{len(curves[mode])} "
          "waveforms change sign")
plt.show()
