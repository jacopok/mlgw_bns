"""Generate many TEOBResumS/PN training residuals for the (2,1), (2,2),
(3,3), (4,4) modes, with parameters drawn from the same distribution used
to build the training dataset for a new model, and plot the amplitude and
phase residuals for all of them.

These are exactly the quantities a per-mode `Model` is trained to
reproduce: `amplitude_residual = log(|A_eob| / |A_pn|)` and
`phase_residual = phi_eob - phi_pn`, as computed by
`WaveformGenerator.generate_residuals`.

The third row shows the phase residual with the linear-in-frequency term
`2 pi (f - f_0) Delta_t(theta)` removed, using the shared time-shift
predictor trained for the `default_hom` model in the top-level folder.
This is the same subtraction `Model.generate` applies (via
`mlgw_bns.model.remove_linear_trend`) before the PCA and the network see
the residuals, so the third row --- not the second --- is what a model
actually has to learn.

Run with: python visualization/plot_teob_pn_residuals.py
"""

from pathlib import Path

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from mlgw_bns.dataset_generation import Dataset
from mlgw_bns.higher_order_modes import Mode, teob_mode_generator_factory
from mlgw_bns.neural_network import load_timeshifts_predictor_from_file

MODES = [Mode(2, 1), Mode(2, 2), Mode(3, 3), Mode(4, 4)]
N_WAVEFORMS = 100

#: Shared cross-mode time-shift predictor of the `default_hom` model, saved
#: by `ModesModel.save` next to the per-mode checkpoints in the repository
#: root. It maps [q, lambda_1, lambda_2, chi_1, chi_2] to a time shift in
#: seconds, at the reference total mass `Dataset.total_mass`.
TIMESHIFTS_FILE = Path(__file__).resolve().parent.parent / "default_hom_timeshifts.pkl"

dataset = Dataset(initial_frequency_hz=20.0, srate_hz=4096.0)

if not TIMESHIFTS_FILE.exists():
    raise FileNotFoundError(
        f"No time-shift predictor at {TIMESHIFTS_FILE}; "
        "run make_default_dataset.py first."
    )
timeshifts_predictor = load_timeshifts_predictor_from_file(str(TIMESHIFTS_FILE))

# Same distribution used when generating the training dataset for a new
# Model: uniform draws over `dataset.parameter_ranges`.
parameter_generator = dataset.make_parameter_generator()
params_list = [next(parameter_generator) for _ in range(N_WAVEFORMS)]

f_hz = np.arange(20., 2048.0, 0.1)
f_natural = dataset.hz_to_natural_units(f_hz)

# The predictor was trained on time shifts computed from frequencies in Hz,
# so the term it feeds must be built in Hz too, even though the plots are
# against the natural-units frequency.
time_shifts = timeshifts_predictor.predict(
    np.array([params.array for params in params_list])
)

fig, axes = plt.subplots(3, len(MODES), figsize=(16, 9), sharex=True)

cmap = matplotlib.colormaps["viridis"]
q_min, q_max = dataset.parameter_ranges.q_range
colors = [cmap((p.mass_ratio - q_min) / (q_max - q_min)) for p in params_list]

for i, mode in enumerate(MODES):
    generator = teob_mode_generator_factory(mode)

    for params, color, time_shift in tqdm(zip(params_list, colors, time_shifts)):
        # `generate_residuals` returns None when the EOB amplitude is
        # discarded (e.g. amp <= 0 somewhere, for the (2,1)/(3,3) modes).
        residuals = generator.generate_residuals(params, f_natural)
        if residuals is None:
            continue
        amplitude_residual, phase_residual = residuals

        # Same subtraction as `mlgw_bns.model.remove_linear_trend`: take out
        # the time-shift term and re-anchor the residual to zero at the first
        # frequency, since a constant phase offset is not learned either.
        phase_residual_flattened = (
            phase_residual
            - 2 * np.pi * (f_hz - f_hz[0]) * time_shift
            - phase_residual[0]
        )

        axes[0, i].plot(f_natural, amplitude_residual, color=color, alpha=0.5, linewidth=0.8)
        axes[1, i].plot(f_natural, phase_residual, color=color, alpha=0.5, linewidth=0.8)
        axes[2, i].plot(f_natural, phase_residual_flattened, color=color, alpha=0.5, linewidth=0.8)

    axes[0, i].set_title(rf"$(\ell, m) = ({mode.l}, {mode.m})$")
    axes[2, i].set_xlabel(r"$Mf$")

axes[0, 0].set_ylabel(r"$A_{\rm EOB} / A_{\rm PN}$")
axes[1, 0].set_ylabel(r"$\phi_{\rm EOB} - \phi_{\rm PN}$ [rad]")
axes[2, 0].set_ylabel(
    r"$\phi_{\rm EOB} - \phi_{\rm PN} - 2 \pi (f - f_0) \Delta t$ [rad]"
)

for ax_row in axes:
    for ax in ax_row:
        ax.grid(True)
        ax.set_xscale("log")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=q_min, vmax=q_max))
fig.colorbar(sm, ax=axes, label="Mass ratio $q$", pad=0.01)

fig.suptitle(
    f"TEOBResumS/PN training residuals, {N_WAVEFORMS} waveforms drawn "
    "from the training parameter distribution"
)

outfile = "teob_pn_residuals_21_22_33_44.png"
fig.savefig(outfile, dpi=150)
print(f"Saved plot to {outfile}")
# plt.show()
