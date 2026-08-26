r"""Validate a trained :class:`Model` (as produced by
``make_default_dataset.py``), both mode-by-mode and for the full
multi-mode waveform reconstruction.

Three things are produced:

1. **mlgw-EOB residuals**, per mode: the amplitude ratio
   ``A_mlgw / A_EOB`` and the phase difference
   ``phi_mlgw - phi_EOB``, for many parameter sets drawn from the
   training distribution. This is the most direct picture of what the
   surrogate gets wrong, and where in frequency it does so.
2. **Per-mode mismatches**, via :class:`ValidateModel`, which
   marginalises over a global time shift and phase. These are reported
   alongside each mode's *share of the PSD-weighted power* in the summed
   waveform, because a mismatch is a relative measure and so says nothing
   on its own about how much a mode matters. The (2,1) mode in
   particular carries :math:`\sim 10^{-5}` of the power and can post a
   mismatch of order unity while the full waveform is accurate to
   :math:`10^{-6}` --- without the weight beside it, that reads as the
   worst thing in the model rather than the least important.
3. **Full-waveform mismatches**, comparing the multi-mode reconstruction
   (:meth:`Model.predict_modes_dict`) against the EOB ground truth
   (:meth:`Model.get_teob_modes_dict`), marginalising over both a
   time shift and a reference azimuthal phase.

The one place this deviates from :mod:`mlgw_bns.model_validation` is the
time-shift predictor: :class:`ValidateModel` reaches for each individual
``ModeModel.timeshifts_predictor``, whereas a :class:`Model` trains a
single shared predictor from the (2,2) mode and applies it to every mode.
:class:`SharedTimeshiftValidateModel` below overrides just that lookup.

Run with: python visualization/validate_model.py
"""

import logging
from typing import Optional

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.mode_model import ParametersWithExtrinsic
from mlgw_bns.model_validation import ValidateModel
from mlgw_bns.model import Model

logging.basicConfig(level=logging.WARNING)

MODEL_FILENAME = "../default_hom"
MODES = [Mode(2, 2), Mode(2, 1), Mode(3, 3), Mode(4, 4)]

N_RESIDUAL_WAVEFORMS = 40
N_MISMATCH_WAVEFORMS = 1000
N_FULL_WAVEFORM_MISMATCHES = 1000

SEED = 17

# Extrinsic parameters used for the full-waveform reconstruction; the
# intrinsic ones are drawn from the training distribution.
DISTANCE_MPC = 100.0
INCLINATION = 1.0
TOTAL_MASS = 2.8


class SharedTimeshiftValidateModel(ValidateModel):
    """:class:`ValidateModel` using a :class:`Model`'s shared predictor.

    A :class:`Model` trains one time-shift predictor (from the (2,2)
    mode) and reuses it for every mode, so validating a single mode must
    use that shared predictor rather than the per-mode one that
    :meth:`ValidateModel.time_shifts_predictor` would return.

    Parameters
    ----------
    model : ModeModel
        The per-mode model to validate.
    shared_predictor : TimeshiftsNN or TimeshiftsGPR
        The owning :class:`Model`'s ``time_shifts_predictor``.
    **kwargs
        Forwarded to :class:`ValidateModel`.
    """

    def __init__(self, model, shared_predictor, **kwargs):
        super().__init__(model, **kwargs)
        self._shared_predictor = shared_predictor

    def time_shifts_predictor(self):
        if self._shared_predictor is None:
            raise ValueError(
                "The Model has no shared time-shift predictor; "
                "train or load one before validating."
            )
        return self._shared_predictor


def load_model() -> Model:
    """Load the trained :class:`Model` from disk."""
    model = Model(
        modes=MODES,
        filename=MODEL_FILENAME,
    )
    model.load()
    if not model.nn_available:
        raise RuntimeError(
            f"No trained network found for {MODEL_FILENAME!r}; "
            "run make_default_dataset.py first."
        )
    return model


def mlgw_eob_residuals(validator: ValidateModel, n_waveforms: int):
    """Return the mlgw-vs-EOB amplitude and phase residuals for one mode.

    Both waveform sets are taken in the model's own downsampled
    amplitude/phase representation, with the learned time-shift
    correction applied to the prediction (exactly as
    :meth:`ValidateModel.validation_mismatches` does), so that what is
    left is the surrogate's own reconstruction error.

    Returns
    -------
    frequencies_hz : np.ndarray
        Frequencies of the phase/amplitude sample points, in Hz.
    amplitude_residuals : np.ndarray
        ``A_mlgw / A_EOB``, shape ``(n_waveforms, n_amp_points)``.
        A perfect reconstruction gives 1, not 0: the model now works
        with the amplitude ratio rather than its logarithm, so that
        sign changes in the EOB amplitude are representable.
    phase_residuals : np.ndarray
        ``phi_mlgw - phi_EOB``, shape ``(n_waveforms, n_phase_points)``.
    parameter_set : ParameterSet
        The (filtered) parameters these residuals correspond to.
    """
    parameter_set = validator.param_set(n_waveforms, SEED)

    true_waveforms, parameter_set = validator.true_waveforms(parameter_set)
    phase0_eob = np.copy(true_waveforms.phases)
    true_waveforms.phases -= phase0_eob[:, 0].reshape(-1, 1)

    predicted_waveforms = validator.predicted_waveforms(parameter_set)
    validator._apply_predicted_time_shifts(predicted_waveforms, parameter_set)

    downsampling = validator.model.downsampling_indices
    frequencies_hz = validator.model.dataset.frequencies_hz

    amplitude_residuals = (
        predicted_waveforms.amplitudes / true_waveforms.amplitudes
    )
    phase_residuals = predicted_waveforms.phases - true_waveforms.phases

    return (
        frequencies_hz[downsampling.amplitude_indices],
        frequencies_hz[downsampling.phase_indices],
        amplitude_residuals,
        phase_residuals,
        parameter_set,
    )


def plot_residuals(model: Model) -> dict:
    """Plot the per-mode mlgw-EOB residuals; return them keyed by mode.

    Three rows: the amplitude residual, the raw phase residual, and the
    phase residual with its best-fit linear-in-frequency term removed.
    The last one matters because a residual that is purely linear in
    frequency is only a time-shift error, which a mismatch calculation
    marginalises away; whatever is left after detrending is genuine
    phase-shape error that no alignment can absorb.
    """
    fig, axes = plt.subplots(
        3, len(MODES), figsize=(6 * len(MODES), 10), squeeze=False
    )

    cmap = matplotlib.colormaps["viridis"]
    q_min, q_max = model.dataset.parameter_ranges.q_range

    residuals_by_mode = {}

    for i, mode in enumerate(MODES):
        validator = SharedTimeshiftValidateModel(
            model.mode_models[mode], model.time_shifts_predictor
        )
        amp_f, phi_f, amp_res, phi_res, param_set = mlgw_eob_residuals(
            validator, N_RESIDUAL_WAVEFORMS
        )
        residuals_by_mode[mode] = (amp_f, phi_f, amp_res, phi_res)

        for j in range(len(amp_res)):
            q = param_set.parameter_array[j, 0]
            color = cmap((q - q_min) / (q_max - q_min))
            axes[0, i].plot(amp_f, amp_res[j], color=color, alpha=0.6, linewidth=0.8)
            axes[1, i].plot(phi_f, phi_res[j], color=color, alpha=0.6, linewidth=0.8)

            slope, intercept = np.polyfit(phi_f, phi_res[j], 1)
            detrended = phi_res[j] - (slope * phi_f + intercept)
            axes[2, i].plot(phi_f, detrended, color=color, alpha=0.6, linewidth=0.8)

        axes[0, i].set_title(rf"$(\ell, m) = ({mode.l}, {mode.m})$")
        axes[2, i].set_xlabel("$f$ [Hz]")

    axes[0, 0].set_ylabel(r"$A_{\rm mlgw} / A_{\rm EOB}$")
    axes[1, 0].set_ylabel(r"$\phi_{\rm mlgw} - \phi_{\rm EOB}$ [rad]")
    axes[2, 0].set_ylabel("phase residual,\nlinear trend removed [rad]")

    # The amplitude row holds a ratio, so its "no error" line sits at 1;
    # the two phase rows are differences, so theirs sit at 0.
    for reference, ax_row in zip([1.0, 0.0, 0.0], axes):
        for ax in ax_row:
            ax.grid(True)
            ax.set_xscale("log")
            ax.axhline(reference, color="black", linewidth=0.8, linestyle="--")

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=q_min, vmax=q_max)
    )
    fig.colorbar(sm, ax=axes, label="Mass ratio $q$", pad=0.02)

    fig.suptitle(
        f"mlgw-EOB reconstruction residuals, {N_RESIDUAL_WAVEFORMS} waveforms "
        "from the training distribution"
    )

    outfile = "validation_residuals.png"
    fig.savefig(outfile, dpi=150)
    print(f"Saved plot to {outfile}")

    return residuals_by_mode


def per_mode_mismatches(model: Model) -> dict:
    """Compute the single-mode mismatch distribution for every mode."""
    mismatches_by_mode = {}

    for mode in MODES:
        validator = SharedTimeshiftValidateModel(
            model.mode_models[mode], model.time_shifts_predictor
        )
        mismatches = validator.validation_mismatches(
            N_MISMATCH_WAVEFORMS, seed=SEED, include_time_shifts=True
        )
        mismatches_by_mode[mode] = np.array(mismatches)
        print(
            f"  ({mode.l},{mode.m}): median {np.median(mismatches):.3e}, "
            f"worst {np.max(mismatches):.3e}"
        )

    return mismatches_by_mode


def full_waveform_mismatches(model: Model) -> tuple:
    r"""Compute the multi-mode full-waveform mismatch distribution.

    Compares :meth:`Model.predict_modes_dict` against the EOB ground
    truth from :meth:`Model.get_teob_modes_dict`, restricted to the
    band where the EOB waveform is actually defined (it is zero-padded
    below its starting frequency).

    Also accumulates, per mode, the fraction of the total PSD-weighted
    power that mode carries, :math:`(h_{\ell m}|h_{\ell m}) / (h|h)`
    with :math:`h = \sum_{\ell m} h_{\ell m}`. The truth waveforms are
    already being generated here, so this costs nothing extra.

    Returns
    -------
    mismatches : np.ndarray
        Full-waveform mismatches.
    power_fractions : dict[Mode, np.ndarray]
        Per-mode power fractions, one entry per waveform.
    """
    reference_model = model.mode_models[Mode(2, 2)]
    validator = ValidateModel(reference_model)
    frequencies = validator.frequencies

    parameter_generator = model.dataset.make_parameter_generator(SEED)

    def inner_product(a: np.ndarray, mask: np.ndarray) -> float:
        """PSD-weighted power of a complex waveform over the support."""
        return float(
            np.abs(
                np.trapezoid(
                    np.conj(a[mask]) * a[mask] / validator.psd_values[mask],
                    x=frequencies[mask],
                )
            )
        )

    mismatches = []
    power_fractions: dict = {mode: [] for mode in MODES}
    for _ in range(N_FULL_WAVEFORM_MISMATCHES):
        intrinsic = next(parameter_generator)
        params = ParametersWithExtrinsic(
            mass_ratio=intrinsic.mass_ratio,
            lambda_1=intrinsic.lambda_1,
            lambda_2=intrinsic.lambda_2,
            chi_1=intrinsic.chi_1,
            chi_2=intrinsic.chi_2,
            distance_mpc=DISTANCE_MPC,
            inclination=INCLINATION,
            total_mass=TOTAL_MASS,
        )

        predicted = model.predict_modes_dict(frequencies, params)
        true = model.get_teob_modes_dict(frequencies, params)

        # The EOB modes are zero below the frequency where the waveform
        # actually starts; restrict to the common support.
        support = np.ones(len(frequencies), dtype=bool)
        for mode_array in true.values():
            support &= np.abs(mode_array) > 0
        if support.sum() < 2:
            continue

        mismatches.append(
            validator.full_waveform_mismatch(
                {k: v[support] for k, v in true.items()},
                {k: v[support] for k, v in predicted.items()},
                frequencies=frequencies[support],
            )
        )

        total_power = inner_product(sum(true.values()), support)
        for mode in MODES:
            key = (mode.l, mode.m)
            if key in true and total_power > 0:
                power_fractions[mode].append(
                    inner_product(true[key], support) / total_power
                )

    mismatches = np.array(mismatches)
    power_fractions = {
        mode: np.array(values) for mode, values in power_fractions.items()
    }
    print(
        f"  full waveform: median {np.median(mismatches):.3e}, "
        f"worst {np.max(mismatches):.3e}"
    )
    return mismatches, power_fractions


def report_weighted_mismatches(
    mismatches_by_mode: dict, power_fractions: dict, full_mismatches: np.ndarray
) -> None:
    """Print per-mode mismatches next to each mode's share of the power.

    The last column is the product of the two: a crude but useful figure
    of merit, since to first order a mode's mismatch contributes to the
    full-waveform error in proportion to how much of the signal it is.
    """
    print()
    print(f"  {'mode':>6}  {'mismatch (med)':>15}  {'power share':>13}  "
          f"{'product':>10}")
    for mode in MODES:
        mismatch = np.median(mismatches_by_mode[mode])
        fractions = power_fractions.get(mode, np.array([]))
        if not len(fractions):
            print(f"  ({mode.l},{mode.m})  {mismatch:15.3e}  {'n/a':>13}  {'n/a':>10}")
            continue
        share = np.median(fractions)
        print(f"  ({mode.l},{mode.m})  {mismatch:15.3e}  {share:13.3e}  "
              f"{mismatch * share:10.3e}")
    print(f"  {'full':>6}  {np.median(full_mismatches):15.3e}  "
          f"{1.0:13.3e}  {np.median(full_mismatches):10.3e}")
    print()


def plot_mismatches(
    mismatches_by_mode: dict,
    full_mismatches: np.ndarray,
    power_fractions: Optional[dict] = None,
) -> None:
    r"""Plot the per-mode and full-waveform mismatch distributions.

    Each distribution is shown as a KDE, evaluated in :math:`\log_{10}`
    of the mismatch (since the values span several orders of magnitude
    and the axis is log-scaled) and then mapped back onto that axis.
    Each mode's legend entry carries its share of the PSD-weighted power,
    so that a broad mismatch distribution can be read against how much
    that mode actually contributes.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    all_values = list(mismatches_by_mode.values()) + [full_mismatches]
    finite = np.concatenate([v[v > 0] for v in all_values if len(v)])
    log_grid = np.linspace(np.log10(finite.min()), np.log10(finite.max()), 400)
    grid = 10**log_grid

    def plot_kde(values: np.ndarray, **kwargs) -> None:
        positive = values[values > 0]
        if len(positive) < 2:
            return
        log_values = np.log10(positive)
        density = gaussian_kde(log_values)(log_grid)
        ax.plot(grid, density, **kwargs)

    for mode, mismatches in mismatches_by_mode.items():
        label = rf"$(\ell, m) = ({mode.l}, {mode.m})$"
        if power_fractions and len(power_fractions.get(mode, [])):
            label += f"  [{np.median(power_fractions[mode]):.3%} of power]"
        plot_kde(mismatches, linewidth=1.8, label=label)

    plot_kde(
        full_mismatches,
        linewidth=2.2,
        linestyle="--",
        color="black",
        label="full waveform",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Mismatch")
    ax.set_title(
        "per-mode mismatches are relative: read them against the power share",
        fontsize="small",
    )
    ax.set_ylabel(r"Density [per $\log_{10}$ mismatch]")
    ax.grid(True)
    ax.legend()
    fig.suptitle("Per-mode and full-waveform mismatch distributions (KDE)")
    fig.tight_layout()

    outfile = "validation_mismatches.png"
    fig.savefig(outfile, dpi=150)
    print(f"Saved plot to {outfile}")


if __name__ == "__main__":
    model = load_model()

    print("Computing mlgw-EOB residuals...")
    plot_residuals(model)

    print("Computing per-mode mismatches...")
    mismatches_by_mode = per_mode_mismatches(model)

    print("Computing full-waveform mismatches and per-mode power shares...")
    full_mismatches, power_fractions = full_waveform_mismatches(model)

    report_weighted_mismatches(mismatches_by_mode, power_fractions, full_mismatches)
    plot_mismatches(mismatches_by_mode, full_mismatches, power_fractions)
