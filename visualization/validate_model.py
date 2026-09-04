r"""Validate a trained :class:`Model` (as produced by
``make_default_dataset.py``), both mode-by-mode and for the full
multi-mode waveform reconstruction.

Three things are produced:

1. **mlgw-EOB residuals**, per mode: the fractional amplitude error
   ``2 (A_mlgw - A_EOB) / (|A_mlgw| + |A_EOB|)`` (bounded through the
   odd-m amplitude nodes), and two views of the phase error --- one with
   the surrogate's *predicted* time-and-phase alignment applied (nothing
   removed afterwards), one with the best-fit linear-in-frequency term
   removed by least squares. The EOB phase is kept with its native value
   at ``f_0``. If the first phase row is much larger than the second, the
   time/phase predictors are not doing their job --- a failed
   reference-phase prediction shows up there as a near-constant offset
   (all-positive or all-negative across the band); normally the two agree.
2. **Per-mode mismatches**, via :class:`ValidateModel`, in the same two
   configurations: residual time and phase optimised (``optimised``),
   and the surrogate's predicted time-and-phase alignment applied with
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

from mlgw_bns.data_management import FDWaveforms, Residuals
from mlgw_bns.dataset_generation import ParameterSet
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
N_MISMATCH_WAVEFORMS = 100
N_FULL_WAVEFORM_MISMATCHES = 100
N_PREDICTOR_VALIDATION_WAVEFORMS = 1000

SEED = 17
#: Seed for the time-shift/mode-phases predictor validation: deliberately
#: different from ``SEED`` (and from whatever seed ``Model.generate`` used
#: for its reference pre-pass) so the comparison is against fresh data,
#: not a resampling of whatever the predictors were fit on.
PREDICTOR_VALIDATION_SEED = 4004

#: Defaults mirroring ``Model.generate``'s reference pre-pass (see
#: ``Model._train_reference_predictors``): the coarse grid the time-shift
#: and mode-phases predictors were actually fit on. Not persisted on the
#: model itself, so validating against "the same grid" means assuming the
#: caller used these defaults (true for ``make_default_dataset.py``).
REFERENCE_GRID_POINTS = 64
REFERENCE_FMAX_HZ = 512.0

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


def batched_true_waveforms(model: Model, mode: Mode, parameter_set, min_valid: int = 2):
    r"""EOB ground truth for one mode, generated with *all* modes requested.

    ``ValidateModel.true_waveforms`` (and :meth:`Model.get_teob_modes_dict`)
    generate ground truth one mode at a time, via
    :meth:`~mlgw_bns.higher_order_modes.TEOBResumSModeGenerator.effective_one_body_waveform`.
    But :meth:`~mlgw_bns.higher_order_modes.TEOBResumSModeGenerator.all_modes_amplitude_phase`
    lowers the ODE integration start frequency to match the *highest-m*
    mode in the request (see ``start_integration_early``), so a single-mode
    call integrates from a later starting point than a batched
    ``model.modes``-wide call would. That changes the merger-aligned
    absolute phase by a large amount (many extra low-frequency GW cycles),
    and is exactly the discrepancy tracked as ``[[batched-multimode-eob]]``.

    The model's ``time_shifts_predictor``/``mode_phases_predictor`` were
    *trained* on the batched convention (``Model._train_reference_predictors``
    uses ``Model._multimode_mode_residuals``, which requests every mode at
    once). So comparing them against a single-mode-generated ground truth
    compares against the wrong absolute phase reference entirely --- this
    function instead reproduces the batched convention, one EOB call (for
    every mode in ``model.modes`` together) per parameter point, exactly
    like the training path.

    Returns
    -------
    true_amplitudes, true_phases : np.ndarray
        Shape ``(n_valid, n_amp_points)`` / ``(n_valid, n_phase_points)``,
        at ``mode``'s own downsampling indices.
    parameter_set : ParameterSet
        Filtered to the waveforms for which the EOB call succeeded.
    """
    mode_model = model.mode_models[mode]
    downsampling = mode_model.downsampling_indices
    frequencies_natural = mode_model.dataset.frequencies
    generator = mode_model.waveform_generator

    waveform_params_list = parameter_set.waveform_parameters(mode_model.dataset)

    valid_indices = []
    true_amp_list = []
    true_phase_list = []
    for j, wp in enumerate(waveform_params_list):
        try:
            batched = generator.all_modes_amplitude_phase(
                wp, model.modes, frequencies_natural
            )
            _, amp_full, phase_full = batched[mode]
        except Exception:  # pragma: no cover - EOB blowups
            continue
        if len(amp_full) != len(frequencies_natural) or not (
            np.all(np.isfinite(amp_full)) and np.all(np.isfinite(phase_full))
        ):
            continue
        valid_indices.append(j)
        true_amp_list.append(amp_full[downsampling.amplitude_indices])
        true_phase_list.append(phase_full[downsampling.phase_indices])

    if len(valid_indices) < min_valid:
        raise RuntimeError(
            f"The batched EOB sweep produced fewer than {min_valid} valid "
            "waveforms; try a larger n_waveforms."
        )

    if len(valid_indices) != len(waveform_params_list):
        parameter_set = parameter_set[valid_indices]

    return np.stack(true_amp_list), np.stack(true_phase_list), parameter_set


def mlgw_eob_residuals(model: Model, mode: Mode, n_waveforms: int):
    r"""Return the mlgw-vs-EOB amplitude and phase residuals for one mode.

    Both waveform sets are taken in the model's own downsampled
    amplitude/phase representation. The ground truth comes from
    :func:`batched_true_waveforms`, not a single-mode EOB call (see its
    docstring for why: the time-shift/mode-phases predictors are trained
    against the batched-EOB-call convention, and comparing against a
    single-mode-generated ground truth compares against the wrong absolute
    phase reference). The EOB phase is kept with its native value at
    :math:`f_0` (not re-zeroed), so the two phase residuals below are on
    the same footing as the residuals plot's two rows:

    * ``phase_residuals_regressed`` --- ``phi_mlgw`` with the *predicted*
      merger time shift and the *predicted* per-mode reference-phase
      constant (``a * M_lm + b``, from the shared ``ModePhasesNN``, for
      HOM models) both added, minus ``phi_EOB``. What is left is the
      frequency-dependent error the surrogate actually contributes once
      every one of its own predictors has run; if this is large, the
      time/phase predictors are not doing their job. For non-HOM models
      (no mode-phases predictor), an empirical median is subtracted
      instead, since there is no reference-phase prediction to add back.
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
    mode_model = model.mode_models[mode]
    validator = SharedTimeshiftValidateModel(mode_model, model.time_shifts_predictor)
    parameter_set = validator.param_set(n_waveforms, SEED)

    true_amplitudes, true_phases, parameter_set = batched_true_waveforms(
        model, mode, parameter_set
    )
    predicted_waveforms = validator.predicted_waveforms(parameter_set)

    downsampling = mode_model.downsampling_indices
    frequencies_hz = mode_model.dataset.frequencies_hz
    phase_freqs = frequencies_hz[downsampling.phase_indices]

    time_shifts = (
        validator.time_shifts_predictor()
        .predict(parameter_set.parameter_array)
        .reshape(-1, 1)
    )

    amplitude_residuals = 2 * (
        predicted_waveforms.amplitudes - true_amplitudes
    ) / (
        np.abs(predicted_waveforms.amplitudes)
        + np.abs(true_amplitudes)
    )

    phase_residuals = predicted_waveforms.phases - true_phases

    # `predicted_waveforms` comes from `predict_waveforms_bulk`, which
    # recomposes the NN's residual plus the PN phase directly and so
    # never adds back the per-mode reference-phase constant that
    # `ModeModel.predict_amplitude_phase` restores via
    # `_predicted_mode_phase0` (`a * M_lm + b`, predicted by the shared
    # `ModePhasesNN` --- see `[[hom-per-mode-phase-anchor]]`), nor the
    # time-shift term. Rather than reconstruct those by hand (error-prone:
    # the phase0 constant is per-mode-scaled and easy to get the sign or
    # magnitude of wrong), call the model's own
    # `predict_amplitude_phase` --- the exact production code path ---
    # at the training total mass, which applies both correctly.
    if mode_model.mode_phases_predictor is not None:
        waveform_params_list = parameter_set.waveform_parameters(mode_model.dataset)
        predicted_full_phases = np.empty_like(phase_residuals)
        for j, wp in enumerate(waveform_params_list):
            extrinsic_params = ParametersWithExtrinsic(
                mass_ratio=wp.mass_ratio,
                lambda_1=wp.lambda_1,
                lambda_2=wp.lambda_2,
                chi_1=wp.chi_1,
                chi_2=wp.chi_2,
                distance_mpc=1.0,
                inclination=0.0,
                total_mass=mode_model.dataset.total_mass,
            )
            _, predicted_full_phases[j] = mode_model.predict_amplitude_phase(
                phase_freqs, extrinsic_params
            )
        phase_residuals_regressed = predicted_full_phases - true_phases
    else:
        # Non-HOM models re-zero the phase at f0 (no phase0 predictor to
        # restore), so only the time shift needs to be added back by hand;
        # an empirical median stands in for the (here, genuinely absent)
        # reference-phase constant.
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
    2. ``phi_mlgw - phi_EOB`` with the surrogate's predicted time shift
       *and* predicted per-mode reference phase applied, nothing removed
       afterwards --- the phase error left once the predictors have run.
       A failed reference-phase prediction shows up as a near-constant
       nonzero offset across the band;
    3. ``phi_mlgw - phi_EOB`` with its best-fit linear-in-frequency term
       removed by least squares --- genuine phase-shape error that no
       time-and-phase alignment can absorb.

    Row 3 is row 2 with an optimal linear alignment subtracted, so it
    lower-bounds row 2; they agree closely when the predictors are good.
    """
    fig, axes = plt.subplots(
        3, len(MODES), figsize=(6 * len(MODES), 10), squeeze=False
    )

    cmap = matplotlib.colormaps["viridis"]
    q_min, q_max = model.dataset.parameter_ranges.q_range

    residuals_by_mode = {}

    for i, mode in enumerate(MODES):
        amp_f, phi_f, amp_res, phi_reg, phi_det, param_set = mlgw_eob_residuals(
            model, mode, N_RESIDUAL_WAVEFORMS
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
        r"$\phi_{\rm mlgw} - \phi_{\rm EOB}$,"
        "\npredicted $\\Delta t$ + phase applied [rad]"
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


def predicted_shift_mismatches(
    model: Model,
    mode: Mode,
    validator: ValidateModel,
    n_waveforms: int,
    parameter_set=None,
):
    r"""Single-mode mismatch with the surrogate's predicted alignment applied.

    Unlike :meth:`ValidateModel.validation_mismatches`, **nothing is
    optimised**: the predicted waveform is built by the full production
    path (:meth:`ModeModel.predict_amplitude_phase`, which applies both
    the predicted per-mode reference phase and the predicted time shift),
    and the overlap uses the *real* Wiener product --- the reference phase
    is not marginalised either. This is the mismatch counterpart of the
    residuals plot's middle row: it says how good the model is once its
    predictors have run, and a large value means a predictor has failed
    (typically the reference-phase constant, which then shows up as a
    near-constant offset in that row).

    The ground truth is the *batched* multi-mode EOB waveform
    (:func:`batched_true_waveforms`), i.e. the same convention the
    reference-phase / time-shift predictors were trained on and the same
    one the residuals plot uses. ``ValidateModel.true_waveforms`` instead
    generates one mode at a time; its absolute phase differs by many
    low-frequency cycles (``[[batched-multimode-eob]]``), and with nothing
    optimised or marginalised here that offset alone gives mismatch ~ 1.

    ``parameter_set``, if given, is used instead of drawing
    ``n_waveforms`` fresh ones (``n_waveforms`` is then ignored) --- used
    by :func:`mismatch_vs_power_by_mode` to pair this up with a specific
    already-drawn sample.
    """
    if parameter_set is None:
        parameter_set = validator.param_set(n_waveforms, SEED)

    true_amplitudes, true_phases, parameter_set = batched_true_waveforms(
        model, mode, parameter_set, min_valid=1
    )
    true_waveforms = FDWaveforms(amplitudes=true_amplitudes, phases=true_phases)

    predicted_waveforms = validator.predicted_waveforms(parameter_set)

    # `predicted_waveforms` (from `predict_waveforms_bulk`) recomposes the NN
    # residual plus the PN phase directly: its phase carries neither the
    # per-mode reference-phase constant nor the predicted time shift.
    # Overwrite just the phase with the full production path
    # `ModeModel.predict_amplitude_phase`, which applies both --- exactly as
    # row 2 of the residuals plot does --- so the two views compare the same
    # waveform. (The bulk-path amplitude is already correct.)
    mode_model = validator.model
    phase_freqs = mode_model.dataset.frequencies_hz[
        mode_model.downsampling_indices.phase_indices
    ]
    for j, wp in enumerate(parameter_set.waveform_parameters(mode_model.dataset)):
        extrinsic_params = ParametersWithExtrinsic(
            mass_ratio=wp.mass_ratio,
            lambda_1=wp.lambda_1,
            lambda_2=wp.lambda_2,
            chi_1=wp.chi_1,
            chi_2=wp.chi_2,
            distance_mpc=1.0,
            inclination=0.0,
            total_mass=mode_model.dataset.total_mass,
        )
        _, predicted_waveforms.phases[j] = mode_model.predict_amplitude_phase(
            phase_freqs, extrinsic_params
        )

    true_cartesian, predicted_cartesian = validator.waveforms(
        true_waveforms, predicted_waveforms
    )
    weight = np.gradient(validator.frequencies) / validator.psd_values

    def inner(a, b):
        return np.sum(np.conj(a) * b * weight, axis=-1).real

    # Real Wiener product: the reference phase is *not* marginalised. This is
    # the mismatch counterpart of the residuals plot's middle row, so the two
    # compare the same predicted waveform (predicted time shift + predicted
    # per-mode reference phase applied, nothing optimised afterwards).
    overlap = inner(true_cartesian, predicted_cartesian) / np.sqrt(
        inner(true_cartesian, true_cartesian)
        * inner(predicted_cartesian, predicted_cartesian)
    )
    return 1.0 - overlap


def optimised_mismatches(validator: ValidateModel, parameter_set):
    r"""Single-mode mismatch for a given parameter set, residual time+phase
    marginalised --- i.e. the same computation as
    ``validator.validation_mismatches(..., include_time_shifts=True)``, but
    for an already-drawn ``parameter_set`` rather than one it draws itself.
    Used by :func:`mismatch_vs_power_by_mode` to pair this up with a
    specific sample.
    """
    true_waveforms, parameter_set = validator.true_waveforms(parameter_set)
    predicted_waveforms = validator.predicted_waveforms(parameter_set)
    validator._apply_predicted_time_shifts(predicted_waveforms, parameter_set)
    return validator.mismatch_array(true_waveforms, predicted_waveforms), parameter_set


def per_mode_mismatches(model: Model) -> dict:
    """Single-mode mismatch distributions for every mode, two configurations.

    Returns ``{mode: (optimised, regressed)}`` where ``optimised`` has a
    residual time shift and phase optimised (the canonical per-mode
    mismatch) and ``regressed`` instead applies the surrogate's predicted
    time shift and reference phase, with nothing optimised --- the same two routes as
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
            predicted_shift_mismatches(model, mode, validator, N_MISMATCH_WAVEFORMS)
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
        Full-waveform mismatches, time- and reference-phase-optimised.
    mismatches_no_opt : np.ndarray
        Full-waveform mismatches with **nothing optimised**: the real
        Wiener product between the summed surrogate waveform
        (:meth:`Model.predict_modes_dict`, its predicted per-mode
        :math:`\Delta t` and reference phases already applied) and the
        summed batched-EOB truth (:meth:`Model.get_teob_modes_dict`). The
        multi-mode counterpart of the per-mode ``predicted-shift``
        mismatch --- how good the surrogate waveform is straight out of
        ``Model.predict``, with no alignment tuning.
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

    def real_wiener_mismatch(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
        """``1 - Re(a|b) / sqrt((a|a)(b|b))`` --- nothing optimised."""
        fm = frequencies[mask]
        psd_m = validator.psd_values[mask]

        def ip(x, y):
            return np.trapezoid(np.conj(x[mask]) * y[mask] / psd_m, x=fm)

        denom = np.sqrt(ip(a, a).real * ip(b, b).real)
        return 1.0 if denom <= 0 else 1.0 - ip(a, b).real / denom

    mismatches = []
    mismatches_no_opt = []
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
        mismatches_no_opt.append(
            real_wiener_mismatch(sum(true.values()), sum(predicted.values()), support)
        )

        total_power = inner_product(sum(true.values()), support)
        for mode in MODES:
            key = (mode.l, mode.m)
            if key in true and total_power > 0:
                power_fractions[mode].append(
                    inner_product(true[key], support) / total_power
                )

    mismatches = np.array(mismatches)
    mismatches_no_opt = np.array(mismatches_no_opt)
    power_fractions = {
        mode: np.array(values) for mode, values in power_fractions.items()
    }
    print(
        f"  full waveform (optimised):     median {np.median(mismatches):.3e}, "
        f"worst {np.max(mismatches):.3e}"
    )
    print(
        f"  full waveform (not optimised): median {np.median(mismatches_no_opt):.3e}, "
        f"worst {np.max(mismatches_no_opt):.3e}"
    )
    if len(mismatches) <= 25:
        print("  per-waveform  [optimised | not optimised]:")
        for a, b in zip(mismatches, mismatches_no_opt):
            print(f"    {a:.3e} | {b:.3e}")
    return mismatches, mismatches_no_opt, power_fractions


def mismatch_vs_power_by_mode(model: Model, n_waveforms: int = N_FULL_WAVEFORM_MISMATCHES) -> dict:
    r"""Per-mode, per-sample mismatch paired with that mode's power share.

    The power fraction needs the full multi-mode Cartesian reconstruction
    (:meth:`Model.predict_modes_dict` and :meth:`Model.get_teob_modes_dict`),
    since it is a ratio of one mode's power to the summed waveform's. But
    those two methods disagree on the *absolute* merger-time convention
    (:meth:`Model.get_teob_modes_dict` keeps TEOBResumS's native phase,
    while the surrogate's reconstruction is built up in a different,
    pre-aligned frame) by an amount that swamps any single-mode mismatch
    unless it is optimised away --- which is exactly what
    :func:`full_waveform_mismatches` already does for the full waveform via
    ``max_delta_t=0.07``. A per-mode mismatch computed directly from these
    two dicts would therefore mostly measure that irrelevant offset, not
    the surrogate's quality.

    So instead, for each sampled waveform, the per-mode mismatches
    (``optimised`` and ``regressed``) are computed exactly as
    :func:`per_mode_mismatches` computes them --- via each mode's
    :class:`SharedTimeshiftValidateModel`, ``optimised`` using the same
    routine as ``validation_mismatches(..., include_time_shifts=True)``
    and ``regressed`` calling :func:`predicted_shift_mismatches` --- just
    for one already-drawn parameter set at a time, so each mismatch can be
    paired with that same sample's power fraction.

    ``power_fraction`` is :math:`(h_{\ell m}|h_{\ell m}) / (h|h)`, taking
    the *maximum* of the fraction computed from the EOB waveform and from
    the mlgw-predicted one (either can be the more informative one, e.g.
    if the surrogate over- or under-predicts a mode's amplitude relative
    to truth).

    Returns
    -------
    dict[Mode, dict[str, np.ndarray]]
        ``{mode: {"power_fraction": ..., "optimised": ..., "regressed": ...}}``.
    """
    validators = {
        mode: SharedTimeshiftValidateModel(
            model.mode_models[mode], model.time_shifts_predictor
        )
        for mode in MODES
    }
    power_validator = ValidateModel(model.mode_models[Mode(2, 2)])
    frequencies = power_validator.frequencies

    parameter_generator = model.dataset.make_parameter_generator(SEED)

    def power(a: np.ndarray, mask: np.ndarray) -> float:
        """PSD-weighted power of a complex waveform over the support."""
        weight = np.gradient(frequencies[mask]) / power_validator.psd_values[mask]
        return float(np.abs(np.sum(np.conj(a[mask]) * a[mask] * weight)))

    results: dict = {
        mode: {"power_fraction": [], "optimised": [], "regressed": []}
        for mode in MODES
    }

    for _ in range(n_waveforms):
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

        support = np.ones(len(frequencies), dtype=bool)
        for mode_array in true.values():
            support &= np.abs(mode_array) > 0
        if support.sum() < 2:
            continue

        total_power_true = power(sum(true.values()), support)
        total_power_pred = power(sum(predicted.values()), support)
        if total_power_true <= 0 or total_power_pred <= 0:
            continue

        parameter_set = ParameterSet.from_list_of_waveform_parameters([intrinsic])

        for mode in MODES:
            key = (mode.l, mode.m)
            if key not in true or key not in predicted:
                continue

            power_true = power(true[key], support) / total_power_true
            power_pred = power(predicted[key], support) / total_power_pred
            power_fraction = max(power_true, power_pred)

            validator = validators[mode]
            optimised_array, valid_param_set = optimised_mismatches(
                validator, parameter_set
            )
            if valid_param_set.parameter_array.shape[0] == 0:
                continue
            try:
                regressed_array = predicted_shift_mismatches(
                    model, mode, validator, 1, parameter_set=valid_param_set
                )
            except RuntimeError:
                continue  # batched EOB call failed for this single sample

            results[mode]["power_fraction"].append(power_fraction)
            results[mode]["optimised"].append(optimised_array[0])
            results[mode]["regressed"].append(regressed_array[0])

    return {
        mode: {key: np.array(values) for key, values in per_mode.items()}
        for mode, per_mode in results.items()
    }


def plot_mismatch_vs_power(data: dict) -> None:
    r"""Scatter mismatch against per-mode power fraction, colored by mode.

    ``data`` is the output of :func:`mismatch_vs_power_by_mode`. Optimised
    points (residual time+phase optimised) are drawn with transparency;
    regressed points (predicted time shift + reference phase, nothing
    optimised) are drawn
    in full color, so the two clouds for a given mode are visually
    distinguishable while sharing that mode's color.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for mode, color in zip(
        MODES, plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ):
        per_mode = data[mode]
        if not len(per_mode["power_fraction"]):
            continue
        label = rf"$(\ell, m) = ({mode.l}, {mode.m})$"
        ax.scatter(
            per_mode["power_fraction"],
            per_mode["optimised"],
            color=color,
            alpha=0.25,
            s=14,
            marker="o",
            linewidths=0,
        )
        ax.scatter(
            per_mode["power_fraction"],
            per_mode["regressed"],
            color=color,
            alpha=1.0,
            s=14,
            marker="o",
            linewidths=0,
            label=label,
        )

    ax.set_xscale("logit")
    ax.set_yscale("log")
    ax.set_xlabel("Power fraction, $\\max$(EOB, mlgw)")
    ax.set_ylabel("Mismatch")
    ax.set_title(
        "faint: residual time+phase optimised    "
        "full color: predicted \u0394t + reference phase, nothing optimised (regressed)",
        fontsize="small",
    )
    ax.grid(True)
    ax.legend()
    fig.suptitle("Per-mode mismatch vs. power fraction")
    fig.tight_layout()

    outfile = f"{OUTPUT_PREFIX}_mismatch_vs_power.png"
    fig.savefig(outfile, dpi=150)
    print(f"Saved plot to {outfile}")


def predictor_validation_data(
    model: Model,
    n_waveforms: int = N_PREDICTOR_VALIDATION_WAVEFORMS,
    seed: int = PREDICTOR_VALIDATION_SEED,
    reference_grid_points: int = REFERENCE_GRID_POINTS,
    reference_fmax_hz: float = REFERENCE_FMAX_HZ,
) -> dict:
    r"""Actual vs. predicted time shift and per-mode reference phase.

    Mirrors :meth:`Model._train_reference_predictors` exactly (same coarse
    geometric grid, same one-EOB-call-per-point sweep via
    :meth:`Model._multimode_mode_residuals`), but on a *fresh* parameter
    draw (``seed``, distinct from both :data:`SEED` and whatever seed
    ``Model.generate`` used), so this checks generalisation rather than
    refitting a training residual.

    The two "actual" targets are computed exactly as
    ``_train_reference_predictors`` computes them:

    * the shared merger time shift --- the least-squares low-frequency
      slope of the (2,2) mode's *raw* phase residual (EOB minus PN),
      via :meth:`Residuals.phase_timeshifts`;
    * each mode's reference phase :math:`\phi_{\ell m}(f_0)` --- simply
      that mode's raw phase residual at the first grid point.

    Returns
    -------
    dict
        ``{"time_shift": {"actual": ..., "predicted": ...},
        "phase0": {mode: {"actual": ..., "predicted": ...}, ...}}``.
        The ``"phase0"`` entry is empty if the model has no
        ``mode_phases_predictor`` (i.e. is not a HOM model).
    """
    reference_mode = Mode(2, 2)
    dataset = model.mode_models[reference_mode].dataset

    f0_natural = float(dataset.frequencies[0])
    grid_hz = np.geomspace(
        dataset.natural_units_to_hz(f0_natural),
        min(reference_fmax_hz, dataset.effective_srate_hz / 2),
        reference_grid_points,
    )
    f_ref_natural = dataset.hz_to_natural_units(grid_hz)
    f_ref_natural[0] = f0_natural
    grid_hz = dataset.natural_units_to_hz(f_ref_natural)

    parameter_generator = dataset.make_parameter_generator(seed=seed)
    params_list = [next(parameter_generator) for _ in range(n_waveforms)]

    parameter_array, _, phase_residuals = model._multimode_mode_residuals(
        params_list, f_ref_natural, progress_desc="Predictor validation sweep"
    )
    if len(parameter_array) < 2:
        raise RuntimeError(
            "The predictor-validation sweep produced fewer than 2 valid "
            "waveforms; try a larger n_waveforms."
        )

    reference_phase_residuals = phase_residuals[reference_mode]
    actual_time_shifts = Residuals(
        np.zeros_like(reference_phase_residuals), reference_phase_residuals
    ).phase_timeshifts(frequencies=grid_hz)
    predicted_time_shifts = np.asarray(
        model.time_shifts_predictor.predict(parameter_array)
    ).reshape(-1)

    result = {
        "time_shift": {
            "actual": actual_time_shifts,
            "predicted": predicted_time_shifts,
        },
        "phase0": {},
    }

    if model.mode_phases_predictor is not None:
        predicted_phase0_array = model.mode_phases_predictor.predict(parameter_array)
        for mode in model.modes:
            mode_phases_index = model.mode_models[mode].mode_phases_index
            if mode_phases_index is None:
                continue
            result["phase0"][mode] = {
                "actual": phase_residuals[mode][:, 0],
                "predicted": predicted_phase0_array[:, mode_phases_index],
            }

    return result


def plot_predictor_validation(data: dict) -> None:
    r"""Histogram actual vs. predicted time shift and per-mode phase0.

    ``data`` is the output of :func:`predictor_validation_data`. One
    panel for the shared time-shift predictor, plus one per mode with a
    reference-phase prediction, each overlaying the "actual" (fresh EOB
    residual) and "predicted" (model) distributions.
    """
    modes_with_phase0 = list(data["phase0"].keys())
    n_panels = 1 + len(modes_with_phase0)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), squeeze=False)
    axes = axes[0]

    def plot_panel(ax, actual: np.ndarray, predicted: np.ndarray, title: str, xlabel: str) -> None:
        ax.hist(predicted-actual, bins=40, alpha=0.55, density=True, label="actual (fresh EOB)")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(True)
        ax.legend(fontsize="small")

    plot_panel(
        axes[0],
        data["time_shift"]["actual"]*1e6,
        data["time_shift"]["predicted"]*1e6,
        "Shared time shift",
        r"$\Delta t$ [us]",
    )

    for i, mode in enumerate(modes_with_phase0):
        plot_panel(
            axes[i + 1],
            data["phase0"][mode]["actual"],
            data["phase0"][mode]["predicted"],
            rf"$(\ell, m) = ({mode.l}, {mode.m})$ reference phase",
            r"$\phi_{\ell m}(f_0)$ [rad]",
        )

    fig.suptitle(
        "Time-shift / mode-phase0 predictors: actual (fresh EOB) vs. predicted"
    )
    fig.tight_layout()

    outfile = f"{OUTPUT_PREFIX}_predictor_validation.png"
    fig.savefig(outfile, dpi=150)
    print(f"Saved plot to {outfile}")


def report_weighted_mismatches(
    mismatches_by_mode: dict, power_fractions: dict, full_mismatches: np.ndarray
) -> None:
    """Print per-mode mismatches next to each mode's share of the power.

    ``optimised`` is the canonical per-mode mismatch (residual time and
    phase optimised); ``pred-shift`` applies the surrogate's predicted
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
    full_mismatches_no_opt: np.ndarray,
    power_fractions: Optional[dict] = None,
) -> None:
    r"""Plot the per-mode and full-waveform mismatch distributions.

    Two stacked panels sharing the mismatch axis:

    * **optimised** --- per-mode mismatch with a residual time shift and
      reference phase marginalised, plus the (time-and-phase-optimised)
      full-waveform mismatch;
    * **not optimised** --- per-mode mismatch with only the surrogate's
      predicted alignment applied (real Wiener product), plus the
      matching non-optimised full-waveform mismatch.

    Each distribution is a KDE in :math:`\log_{10}` mismatch. Each mode's
    legend entry carries its share of the PSD-weighted power.
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    all_values = [arr for pair in mismatches_by_mode.values() for arr in pair]
    all_values += [full_mismatches, full_mismatches_no_opt]
    finite = np.concatenate([v[v > 0] for v in all_values if len(v)])
    log_grid = np.linspace(np.log10(finite.min()), np.log10(finite.max()), 400)
    grid = 10**log_grid

    # Per-panel peak of the *per-mode* KDEs only, so a tall full-waveform
    # spike (e.g. the not-optimised one piling up near 1) does not squash
    # the single-mode curves off the bottom of the axis.
    per_mode_peak = [0.0, 0.0]

    def plot_kde(i: int, values: np.ndarray, track: bool = False, **kwargs) -> None:
        positive = values[values > 0]
        if len(positive) < 2:
            return
        density = gaussian_kde(np.log10(positive))(log_grid)
        axes[i].plot(grid, density, **kwargs)
        if track:
            per_mode_peak[i] = max(per_mode_peak[i], float(density.max()))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for (mode, (optimised, regressed)), color in zip(mismatches_by_mode.items(), colors):
        label = rf"$(\ell, m) = ({mode.l}, {mode.m})$"
        if power_fractions and len(power_fractions.get(mode, [])):
            label += f"  [{np.median(power_fractions[mode]):.3%} of power]"
        plot_kde(0, optimised, track=True, linewidth=2.0, color=color, label=label)
        plot_kde(1, regressed, track=True, linewidth=2.0, color=color, label=label)

    plot_kde(0, full_mismatches, linewidth=2.4, linestyle="--",
             color="black", label="full waveform")
    plot_kde(1, full_mismatches_no_opt, linewidth=2.4, linestyle="--",
             color="black", label="full waveform")

    axes[0].set_title("residual time + reference phase optimised", fontsize="small")
    axes[1].set_title(
        "surrogate's predicted \u0394t + reference phase applied, nothing optimised",
        fontsize="small",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Mismatch")
    for i, ax in enumerate(axes):
        ax.set_ylabel(r"Density [per $\log_{10}$ mismatch]")
        ax.grid(True)
        ax.legend(fontsize="small")
        if per_mode_peak[i] > 0:
            ax.set_ylim(0, 1.15 * per_mode_peak[i])
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

    print("Validating time-shift and mode-phase0 predictors against fresh data...")
    plot_predictor_validation(predictor_validation_data(model))

    print("Computing per-mode mismatches...")
    mismatches_by_mode = per_mode_mismatches(model)

    print("Computing full-waveform mismatches and per-mode power shares...")
    full_mismatches, full_mismatches_no_opt, power_fractions = full_waveform_mismatches(
        model
    )

    report_weighted_mismatches(mismatches_by_mode, power_fractions, full_mismatches)
    plot_mismatches(
        mismatches_by_mode, full_mismatches, full_mismatches_no_opt, power_fractions
    )

    print("Computing per-mode mismatch vs. power fraction...")
    mismatch_vs_power = mismatch_vs_power_by_mode(model)
    plot_mismatch_vs_power(mismatch_vs_power)
