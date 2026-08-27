"""Generate many TEOBResumS/PN training residuals for the (2,1), (2,2),
(3,3), (4,4) modes, with parameters drawn from the same distribution used
to build the training dataset for a new model, and plot the amplitude and
phase residuals for all of them.

These are exactly the quantities a per-mode `ModeModel` is trained to
reproduce: `amplitude_residual = A_eob / A_pn` and
`phase_residual = phi_eob - phi_pn`, as computed by
`WaveformGenerator.generate_residuals`. The ratio is signed, not a log:
the EOB mode amplitude crosses zero within the band for a few per cent
of the (2,1) waveforms.

The first two rows contrast the two amplitude parametrizations. The
first divides each waveform by its own PN amplitude, which is what the
originally shipped models were trained on; the second divides every
waveform by the PN amplitude of one fixed reference --- the centre of
the parameter ranges, zero spins --- which is what `reference_amplitude`
does. The (2,1) and (3,3) PN amplitudes have a deep minimum at a
parameter-dependent frequency, and dividing by it there throws the ratio
to twenty or sixty; those are the spikes the first row shows and the
second does not.

The fourth row shows the phase residual with the linear-in-frequency term
`2 pi (f - f_0) Delta_t(theta)` removed, using the shared time-shift
predictor trained for the `default_hom` model in the top-level folder.
This is the same subtraction `ModeModel.generate` applies (via
`mlgw_bns.mode_model.remove_linear_trend`) before the PCA and the network see
the residuals, so the fourth row --- not the third --- is what a model
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
#: by `Model.save` next to the per-mode checkpoints in the repository
#: root. It maps [q, lambda_1, lambda_2, chi_1, chi_2] to a time shift in
#: seconds, at the reference total mass `Dataset.total_mass`.
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
TIMESHIFTS_FILE = REPOSITORY_ROOT / "default_hom_timeshifts.pkl"
if not TIMESHIFTS_FILE.exists():
    # Fall back to the copy shipped inside the package, so that the plot can
    # be made without having trained a model locally first.
    TIMESHIFTS_FILE = REPOSITORY_ROOT / "mlgw_bns" / "data" / "default_hom_timeshifts.pkl"

dataset = Dataset(initial_frequency_hz=20.0, srate_hz=4096.0)

#: Same dataset, but flagged so that it can hand over the fixed reference
#: parameters whose PN amplitude divides the EOB one when
#: `reference_amplitude` is on.
reference_dataset = Dataset(
    initial_frequency_hz=20.0, srate_hz=4096.0, reference_amplitude=True
)
reference_parameters = reference_dataset.amplitude_reference_parameters

if not TIMESHIFTS_FILE.exists():
    raise FileNotFoundError(
        f"No time-shift predictor at {TIMESHIFTS_FILE}; "
        "run make_default_dataset.py first."
    )
timeshifts_predictor = load_timeshifts_predictor_from_file(str(TIMESHIFTS_FILE))

# Same distribution used when generating the training dataset for a new
# ModeModel: uniform draws over `dataset.parameter_ranges`.
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

fig, axes = plt.subplots(4, len(MODES), figsize=(16, 12), sharex=True)

cmap = matplotlib.colormaps["viridis"]
q_min, q_max = dataset.parameter_ranges.q_range
colors = [cmap((p.mass_ratio - q_min) / (q_max - q_min)) for p in params_list]

for i, mode in enumerate(MODES):
    generator = teob_mode_generator_factory(mode)
    reference_pn_amplitude = generator.post_newtonian_amplitude(
        reference_parameters, f_natural
    )

    for params, color, time_shift in tqdm(zip(params_list, colors, time_shifts)):
        # The signature is Optional, so the guard stays; in practice the EOB
        # amplitude is kept even where it is negative, which is why the
        # (2,1) curves below cross zero rather than stopping.
        residuals = generator.generate_residuals(params, f_natural)
        if residuals is None:
            continue
        amplitude_residual, phase_residual = residuals

        # What `reference_amplitude` would model instead. Calling
        # `generate_residuals` again with `amplitude_reference` would give the
        # same thing but regenerate the EOB waveform; since only the divisor
        # differs, it is cheaper to swap it here.
        reference_amplitude_residual = (
            amplitude_residual
            * generator.post_newtonian_amplitude(params, f_natural)
            / reference_pn_amplitude
        )

        # Same subtraction as `mlgw_bns.mode_model.remove_linear_trend`: take out
        # the time-shift term and re-anchor the residual to zero at the first
        # frequency, since a constant phase offset is not learned either.
        phase_residual_flattened = (
            phase_residual
            - 2 * np.pi * (f_hz - f_hz[0]) * time_shift
            - phase_residual[0]
        )

        axes[0, i].plot(f_natural, amplitude_residual, color=color, alpha=0.5, linewidth=0.8)
        axes[1, i].plot(
            f_natural, reference_amplitude_residual, color=color, alpha=0.5, linewidth=0.8
        )
        axes[2, i].plot(f_natural, phase_residual, color=color, alpha=0.5, linewidth=0.8)
        axes[3, i].plot(f_natural, phase_residual_flattened, color=color, alpha=0.5, linewidth=0.8)

    axes[0, i].set_title(rf"$(\ell, m) = ({mode.l}, {mode.m})$")
    axes[3, i].set_xlabel(r"$Mf$")

    # The reference ratio falls by orders of magnitude across the band and
    # still changes sign, so it needs a symmetric log scale; the threshold is
    # set from the data so that the linear region is the noise near zero.
    # largest = max(abs(line.get_ydata()).max() for line in axes[1, i].lines)
    # axes[1, i].set_yscale("symlog", linthresh=largest * 1e-4)

axes[0, 0].set_ylabel(r"$A_{\rm EOB}(\theta) / A_{\rm PN}(\theta)$")
axes[1, 0].set_ylabel(r"$A_{\rm EOB}(\theta) / A_{\rm PN}(\theta_{\rm ref})$")
axes[2, 0].set_ylabel(r"$\phi_{\rm EOB} - \phi_{\rm PN}$ [rad]")
axes[3, 0].set_ylabel(
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
