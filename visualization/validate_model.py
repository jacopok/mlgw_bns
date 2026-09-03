r"""Validate a trained :class:`Model` (as produced by
``make_default_dataset.py``), both mode-by-mode and for the full
multi-mode waveform reconstruction.

Three things are produced:

1. **mlgw-EOB residuals**, per mode: the fractional amplitude error
   ``2 (A_mlgw - A_EOB) / (|A_mlgw| + |A_EOB|)`` (bounded through the
   odd-m amplitude nodes), and two views of the phase error --- one with
   the surrogate's *predicted* merger time shift applied (constant removed),
   one with the best-fit linear-in-frequency term removed by least
   squares. The EOB phase is kept with its native value at ``f_0``. If
   the first phase row is much larger than the second, the time/phase
   predictors are not doing their job; normally the two agree.
2. **Per-mode mismatches**, via :class:`ValidateModel`, in the same two
   configurations: residual time and phase marginalised (``optimised``),
   and the predicted time shift applied with only the phase marginalised
   (``pred-shift``). These are reported alongside each mode's *share of
   the PSD-weighted power* in the summed waveform, because a mismatch is
   a relative measure and so says nothing on its own about how much a
   mode matters. The (2,1) mode in particular carries
   :math:`\sim 10^{-5}` of the power and can post a mismatch of order
   unity while the full waveform is accurate to :math:`10^{-6}` ---
   without the weight beside it, that reads as the worst thing in the
   model rather than the least important.
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
import os
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

#: Default model to validate, relative to this directory. Override with
#: ``--model``; the figures are then named after it, so that several
#: models can be validated side by side without overwriting each other.
MODEL_FILENAME = "../default_hom"
OUTPUT_PREFIX = "validation"

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


def load_model(filename: str = None) -> Model:
    """Load the trained :class:`Model` from disk."""
    model = Model(
        modes=MODES,
        filename=MODEL_FILENAME if filename is None else filename,
    )
    model.load()
    if not model.nn_available:
        raise RuntimeError(
            f"No trained network found for {MODEL_FILENAME!r}; "
            "run make_default_dataset.py first."
        )
    return model


def mlgw_eob_residuals(validator: ValidateModel, n_waveforms: int):
    r"""Return the mlgw-vs-EOB amplitude and phase residuals for one mode.

    Both waveform sets are taken in the model's own downsampled
    amplitude/phase representation. The EOB phase is kept with its native
    value at :math:`f_0` (not re-zeroed), so the two phase residuals below
    are on the same footing as the residuals plot's two rows:

    * ``phase_residuals_regressed`` --- ``phi_mlgw`` with the *predicted*
      merger time shift added, minus ``phi_EOB``, with the overall
      constant removed (a global reference phase is marginalised in every
      mismatch). What is left is the frequency-dependent error the
      surrogate actually contributes once its own predictors have run;
      if this is large, the time/phase predictors are not doing their job.
    * ``phase_residuals_detrended`` --- ``phi_mlgw - phi_EOB`` with its
      best-fit linear-in-frequency term removed by least squares. A purely
      linear residual is only a time-shift error, which a mismatch
      marginalises away; whatever is left is genuine phase-shape error.

    The two agree closely when the time-shift predictor is accurate.

    Returns
    -------
    amplitude_frequencies_hz, phase_frequencies_hz : np.ndarray
        Frequencies of the amplitude / phase sample points, in Hz.
    amplitude_residuals : np.ndarray
        ``2 (A_mlgw - A_EOB) / (|A_mlgw| + |A_EOB|)``, shape
        ``(n_waveforms, n_amp_points)``. This is the fractional amplitude
        error for small errors, but stays bounded to ``[-2, 2]`` through
        the amplitude nodes of the odd-m modes, where ``A_EOB`` passes
        through zero and a plain ratio would diverge.
    phase_residuals_regressed : np.ndarray
        Shape ``(n_waveforms, n_phase_points)``, see above.
    phase_residuals_detrended : np.ndarray
        Shape ``(n_waveforms, n_phase_points)``, see above.
    parameter_set : ParameterSet
        The (filtered) parameters these residuals correspond to.
    """
    parameter_set = validator.param_set(n_waveforms, SEED)

    true_waveforms, parameter_set = validator.true_waveforms(parameter_set)
    predicted_waveforms = validator.predicted_waveforms(parameter_set)

    downsampling = validator.model.downsampling_indices
    frequencies_hz = validator.model.dataset.frequencies_hz
    phase_freqs = frequencies_hz[downsampling.phase_indices]

    time_shifts = (
        validator.time_shifts_predictor()
        .predict(parameter_set.parameter_array)
        .reshape(-1, 1)
    )

    amplitude_residuals = 2 * (
        predicted_waveforms.amplitudes - true_waveforms.amplitudes
    ) / (
        np.abs(predicted_waveforms.amplitudes)
        + np.abs(true_waveforms.amplitudes)
    )

    phase_residuals = predicted_waveforms.phases - true_waveforms.phases

    phase_residuals_regressed = phase_residuals + (
        2 * np.pi * (phase_freqs - phase_freqs[0]) * time_shifts
    )
    phase_residuals_regressed -= np.median(
        phase_residuals_regressed, axis=1, keepdims=True
    )

    phase_residuals_detrended = np.empty_like(phase_residuals)
    for j in range(len(phase_residuals)):
        slope, intercept = np.polyfit(phase_freqs, phase_residuals[j], 1)
        phase_residuals_detrended[j] = phase_residuals[j] - (
            slope * phase_freqs + intercept
        )

    return (
        frequencies_hz[downsampling.amplitude_indices],
        phase_freqs,
        amplitude_residuals,
        phase_residuals_regressed,
        phase_residuals_detrended,
        parameter_set,
    )


def plot_residuals(model: Model) -> dict:
    """Plot the per-mode mlgw-EOB residuals; return them keyed by mode.

    Three rows:

    1. ``2 (A_mlgw - A_EOB) / (|A_mlgw| + |A_EOB|)`` --- the fractional
       amplitude error, bounded to ``[-2, 2]`` through the odd-m modes'
       amplitude nodes;
    2. ``(phi_mlgw + predicted time shift) - phi_EOB``, overall constant
       removed --- the frequency-dependent phase error left once the
       surrogate's own time/phase predictors have run. Large values here
       mean the predictors are not doing their job;
    3. ``phi_mlgw - phi_EOB`` with its best-fit linear-in-frequency term
       removed by least squares --- genuine phase-shape error that no
       time-and-phase alignment can absorb.

    Rows 2 and 3 measure the same thing by two routes (predicted vs
    optimised alignment) and agree closely when the predictors are good.
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
        amp_f, phi_f, amp_res, phi_reg, phi_det, param_set = mlgw_eob_residuals(
            validator, N_RESIDUAL_WAVEFORMS
        )
        residuals_by_mode[mode] = (amp_f, phi_f, amp_res, phi_reg, phi_det)

        for j in range(len(amp_res)):
            q = param_set.parameter_array[j, 0]
            color = cmap((q - q_min) / (q_max - q_min))
            axes[0, i].plot(amp_f, amp_res[j], color=color, alpha=0.6, linewidth=0.8)
            axes[1, i].plot(phi_f, phi_reg[j], color=color, alpha=0.6, linewidth=0.8)
            axes[2, i].plot(phi_f, phi_det[j], color=color, alpha=0.6, linewidth=0.8)

        axes[0, i].set_title(rf"$(\ell, m) = ({mode.l}, {mode.m})$")
        axes[0, i].set_ylim(-2.1, 2.1)
        axes[2, i].set_xlabel("$f$ [Hz]")

    axes[0, 0].set_ylabel(
        r"$2 (A_{\rm mlgw} - A_{\rm EOB}) / (|A_{\rm mlgw}| + |A_{\rm EOB}|)$"
    )
    axes[1, 0].set_ylabel(
        r"$(\phi_{\rm mlgw} + \Delta t_{\rm pred}) - \phi_{\rm EOB}$,"
        "\nconstant removed [rad]"
    )
    axes[2, 0].set_ylabel(
        r"$\phi_{\rm mlgw} - \phi_{\rm EOB}$,"
        "\nbest-fit linear term removed [rad]"
    )

    # Every row is now a difference, so its "no error" line sits at 0.
    for ax_row in axes:
        for ax in ax_row:
            ax.grid(True)
            ax.set_xscale("log")
            ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=q_min, vmax=q_max)
    )
    fig.colorbar(sm, ax=axes, label="Mass ratio $q$", pad=0.02)

    fig.suptitle(
        f"mlgw-EOB reconstruction residuals, {N_RESIDUAL_WAVEFORMS} waveforms "
        "from the training distribution"
    )

    outfile = f"{OUTPUT_PREFIX}_residuals.png"
    fig.savefig(outfile, dpi=150)
    print(f"Saved plot to {outfile}")

    return residuals_by_mode


def predicted_shift_mismatches(validator: ValidateModel, n_waveforms: int):
    r"""Single-mode mismatch with the *predicted* merger time shift applied.

    Unlike :meth:`ValidateModel.validation_mismatches`, no residual time
    shift is optimised: the surrogate's own time-shift prediction is
    added and only the global reference phase is marginalised (via the
    :math:`|\cdot|` in the overlap). This is the mismatch counterpart of
    the residuals plot's middle row --- it says how good the model is
    once its predictors have run, with nothing optimised afterwards.
    """
    parameter_set = validator.param_set(n_waveforms, SEED)
    true_waveforms, parameter_set = validator.true_waveforms(parameter_set)
    predicted_waveforms = validator.predicted_waveforms(parameter_set)

    phase_freqs = validator.model.dataset.frequencies_hz[
        validator.model.downsampling_indices.phase_indices
    ]
    time_shifts = (
        validator.time_shifts_predictor()
        .predict(parameter_set.parameter_array)
        .reshape(-1, 1)
    )
    predicted_waveforms.phases = predicted_waveforms.phases + (
        2 * np.pi * (phase_freqs - phase_freqs[0]) * time_shifts
    )

    true_cartesian, predicted_cartesian = validator.waveforms(
        true_waveforms, predicted_waveforms
    )
    weight = np.gradient(validator.frequencies) / validator.psd_values

    def inner(a, b):
        return np.abs(np.sum(np.conj(a) * b * weight, axis=-1))

    overlap = inner(true_cartesian, predicted_cartesian) / np.sqrt(
        inner(true_cartesian, true_cartesian)
        * inner(predicted_cartesian, predicted_cartesian)
    )
    return 1.0 - overlap


def per_mode_mismatches(model: Model) -> dict:
    """Single-mode mismatch distributions for every mode, two configurations.

    Returns ``{mode: (optimised, regressed)}`` where ``optimised`` has a
    residual time shift and phase marginalised (the canonical per-mode
    mismatch) and ``regressed`` instead applies the surrogate's predicted
    time shift and marginalises only the phase --- the same two routes as
    rows 3 and 2 of the residuals plot.
    """
    mismatches_by_mode = {}

    for mode in MODES:
        validator = SharedTimeshiftValidateModel(
            model.mode_models[mode], model.time_shifts_predictor
        )
        optimised = np.array(
            validator.validation_mismatches(
                N_MISMATCH_WAVEFORMS, seed=SEED, include_time_shifts=True
            )
        )
        regressed = np.array(
            predicted_shift_mismatches(validator, N_MISMATCH_WAVEFORMS)
        )
        mismatches_by_mode[mode] = (optimised, regressed)
        print(
            f"  ({mode.l},{mode.m}): optimised median {np.median(optimised):.3e}, "
            f"predicted-shift median {np.median(regressed):.3e}"
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

    ``optimised`` is the canonical per-mode mismatch (residual time and
    phase marginalised); ``pred-shift`` applies the surrogate's predicted
    time shift instead. The ``product`` column is ``optimised`` times the
    power share: to first order a mode's mismatch contributes to the
    full-waveform error in proportion to how much of the signal it is.
    """
    print()
    print(f"  {'mode':>6}  {'optimised (med)':>15}  {'pred-shift (med)':>17}  "
          f"{'power share':>13}  {'product':>10}")
    for mode in MODES:
        optimised, regressed = mismatches_by_mode[mode]
        opt_med, reg_med = np.median(optimised), np.median(regressed)
        fractions = power_fractions.get(mode, np.array([]))
        if not len(fractions):
            print(f"  ({mode.l},{mode.m})  {opt_med:15.3e}  {reg_med:17.3e}  "
                  f"{'n/a':>13}  {'n/a':>10}")
            continue
        share = np.median(fractions)
        print(f"  ({mode.l},{mode.m})  {opt_med:15.3e}  {reg_med:17.3e}  "
              f"{share:13.3e}  {opt_med * share:10.3e}")
    print(f"  {'full':>6}  {np.median(full_mismatches):15.3e}  {'':>17}  "
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

    all_values = [
        arr
        for pair in mismatches_by_mode.values()
        for arr in pair
    ] + [full_mismatches]
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

    for (mode, (optimised, regressed)), color in zip(
        mismatches_by_mode.items(), plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ):
        label = rf"$(\ell, m) = ({mode.l}, {mode.m})$"
        if power_fractions and len(power_fractions.get(mode, [])):
            label += f"  [{np.median(power_fractions[mode]):.3%} of power]"
        plot_kde(optimised, linewidth=2.0, color=color, label=label)
        plot_kde(regressed, linewidth=1.4, linestyle="--", color=color)

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
        "solid: residual time+phase marginalised    dashed: predicted time "
        "shift applied, phase marginalised",
        fontsize="small",
    )
    ax.set_ylabel(r"Density [per $\log_{10}$ mismatch]")
    ax.grid(True)
    ax.legend()
    fig.suptitle("Per-mode and full-waveform mismatch distributions (KDE)")
    fig.tight_layout()

    outfile = f"{OUTPUT_PREFIX}_mismatches.png"
    fig.savefig(outfile, dpi=150)
    print(f"Saved plot to {outfile}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=MODEL_FILENAME,
        help="base filename of the model to validate",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="prefix for the output figures; defaults to the model's name",
    )
    parser.add_argument(
        "--n-mismatches",
        type=int,
        default=None,
        help=(
            "waveforms per mismatch distribution; the default of "
            f"{N_MISMATCH_WAVEFORMS} gives smooth tails but is slow, and a "
            "couple of hundred is enough to compare two models' medians"
        ),
    )
    args = parser.parse_args()

    if args.n_mismatches is not None:
        N_MISMATCH_WAVEFORMS = args.n_mismatches
        N_FULL_WAVEFORM_MISMATCHES = args.n_mismatches

    OUTPUT_PREFIX = (
        args.prefix
        if args.prefix is not None
        else os.path.basename(args.model.rstrip("/")) or OUTPUT_PREFIX
    )

    model = load_model(args.model)

    print("Computing mlgw-EOB residuals...")
    plot_residuals(model)

    print("Computing per-mode mismatches...")
    mismatches_by_mode = per_mode_mismatches(model)

    print("Computing full-waveform mismatches and per-mode power shares...")
    full_mismatches, power_fractions = full_waveform_mismatches(model)

    report_weighted_mismatches(mismatches_by_mode, power_fractions, full_mismatches)
    plot_mismatches(mismatches_by_mode, full_mismatches, power_fractions)
