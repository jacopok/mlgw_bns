r"""Validate a trained :class:`Model` (as produced by
``make_default_dataset.py``), both mode-by-mode and for the full
multi-mode waveform reconstruction.

Five things are produced:

1. **mlgw-EOB residuals**, per mode: the fractional amplitude error
   ``2 (A_mlgw - A_EOB) / (|A_mlgw| + |A_EOB|)`` (bounded through the
   odd-m amplitude nodes), and two views of the phase error --- one with
   both phases referenced to their own merger (nothing removed
   afterwards), one with a single time shift and coalescence phase per
   waveform removed from all its modes at once. If the first phase row is
   much larger than the second, the surrogate's merger time or coalescence
   phase is off.
2. **Per-mode mismatches**, via :class:`ValidateModel`, in the same two
   configurations: residual time and phase optimised (``optimised``),
   and both referenced to their own merger with nothing optimised
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
4. **Precessing mismatches** of :class:`PrecessingModel` against
   TEOBResumS' own precessing :math:`h_+` and :math:`h_\times`.
5. **Prediction wall-clock times**, aligned and precessing, numpy and JAX.

The per-mode and full-waveform comparisons reference both the surrogate
and the EOB ground truth to the merger (the tangent to the (2,2) phase at
the top of the trained band), as :meth:`Model.predict_modes_dict` and
:meth:`Model.get_teob_modes_dict` do.

Run with: python visualization/validate_model.py
"""

import logging
import os
import pickle
import shutil
import sys

if not sys.stderr.isatty():
    # Logged to a file: a progress line a minute, not a carriage return a step.
    os.environ.setdefault("TQDM_MININTERVAL", "60")
from time import perf_counter
from typing import Optional

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm

from mlgw_bns.data_management import FDWaveforms
from mlgw_bns.dataset_generation import ParameterSet
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.mode_model import ParametersWithExtrinsic
from mlgw_bns.model_validation import ValidateModel
from mlgw_bns.model import Model, _build_mode_coeffs
from mlgw_bns.precessing_model import PrecessingModel, PrecessingParametersWithExtrinsic
from mlgw_bns.taylorf2 import SUN_MASS_SECONDS

logging.basicConfig(level=logging.WARNING)

#: LaTeX rendering (real ``text.usetex``, not just mathtext) for every
#: plot in this script, set globally so individual plotting functions
#: don't each have to opt in.
plt.rcParams.update(
    {
        # where there is LaTeX (not on every cluster node)
        "text.usetex": shutil.which("latex") is not None,
        "font.family": "serif",
    }
)


def _sci_latex(x: float, sig: int = 1) -> str:
    r"""Format ``x`` as LaTeX scientific notation, e.g. ``$8.9 \times 10^{-4}$``.

    Used for axis/tick labels where Python's ``e`` notation (``8.9e-04``)
    would look out of place next to LaTeX-rendered text.
    """
    if x <= 0:
        return "$0$"
    exponent = int(np.floor(np.log10(x)))
    mantissa = x / 10**exponent
    if round(mantissa, sig) >= 10.0:
        mantissa /= 10.0
        exponent += 1
    return rf"${mantissa:.{sig}f} \times 10^{{{exponent}}}$"

#: Default model to validate, relative to this directory. Override with
#: ``--model``; the figures are then named after it, so that several
#: models can be validated side by side without overwriting each other.
MODEL_FILENAME = "../default_hom"
OUTPUT_PREFIX = "validation"

MODES = [Mode(2, 2), Mode(2, 1), Mode(3, 1), Mode(3, 2), Mode(3, 3), Mode(4, 3), Mode(4, 4)]

N_RESIDUAL_WAVEFORMS = 40
N_MISMATCH_WAVEFORMS = 100
N_FULL_WAVEFORM_MISMATCHES = 100

#: Sample sizes for :func:`mode_subset_data`: ``N_MODE_SUBSET_WAVEFORMS``
#: intrinsic (mass/spin) draws, each re-evaluated at
#: ``N_MODE_SUBSET_ORIENTATIONS`` random inclination/coalescence-phase
#: draws --- the mode content of a waveform is fixed per intrinsic draw
#: (computed once), so this is the expensive axis: every
#: (waveform, orientation, mode subset) triple runs its own
#: time-and-phase-optimised ``full_waveform_mismatch``. Lower these for a
#: quick check.
N_MODE_SUBSET_WAVEFORMS = 20
N_MODE_SUBSET_ORIENTATIONS = 20
#: Distinct from SEED (used for the intrinsic draws below) so the
#: inclination/phase sampling doesn't reuse the same stream.
MODE_SUBSET_ORIENTATION_SEED = 20260917

SEED = 17

# Extrinsic parameters used for the full-waveform reconstruction; the
# intrinsic ones are drawn from the training distribution.
DISTANCE_MPC = 100.0
INCLINATION = 1.0
TOTAL_MASS = 2.8

#: Low-frequency floor for the full-waveform validation, in Hz. Below the
#: trained band's own edge (``dataset.effective_initial_frequency_hz``,
#: ~3.57 Hz for ``default_hom`` at ``TOTAL_MASS``), ``Model.predict`` /
#: ``predict_modes_dict`` splice in each mode's post-Newtonian low-frequency
#: extension (``ModeModel.predict_amplitude_phase``'s
#: ``extend_with_post_newtonian``) rather than raising. ``ValidateModel``'s
#: own machinery does not support this (it resamples the trained nodes and
#: would extrapolate), so :func:`full_waveform_mismatches` and
#: :func:`mismatch_vs_power_by_mode` build their own extended
#: frequency/PSD grid via :func:`extended_frequency_grid` instead of using
#: ``validator.frequencies`` directly, to exercise that code path with
#: every sampled waveform. 2 Hz is comfortably below where any mode
#: actually has EOB support (support masking still applies), so it
#: validates the extension itself rather than moving the mismatch numbers.
LOW_FREQUENCY_HZ = 2.0

#: Grid sizes for the wall-clock timing benchmark (see
#: :func:`timing_benchmark`); smaller than ``benchmark_evaluation_time.py``'s
#: own defaults since this runs as one step of a broader validation pass,
#: not a dedicated timing sweep.
TIMING_N_POINTS = (128, 256, 512, 1024, 2048, 4096)
TIMING_SEEDS = 5
TIMING_EPOCHS = 5
#: Lowered from 1024 -- see benchmark_evaluation_time.py's --jax-batch help:
#: its per-call cost scales ~linearly with this, and 128 already averages
#: the per-waveform time far better than repeating the call would.
TIMING_JAX_BATCH = 64

#: Precessing binaries compared against TEOBResumS' own precessing
#: polarizations (see :func:`precessing_mismatches`): the aligned draw of
#: ``benchmark_evaluation_time.random_parameters`` plus in-plane spins up to
#: this magnitude, given at this frequency (0.95 times TEOBResumS'
#: ``initial_frequency``, where it imposes them), and compared above
#: ``PRECESSING_MIN_FREQUENCY_HZ``, where every inertial multipole up to
#: m = 4 is present in TEOBResumS' output (above m / 2 times its start).
N_PRECESSING_WAVEFORMS = 100
PRECESSING_MAX_IN_PLANE_SPIN = 0.5
PRECESSING_REFERENCE_FREQUENCY_HZ = 9.5
PRECESSING_MIN_FREQUENCY_HZ = 21.0
#: TEOBResumS' frequency-domain twist leaves spurious ~2 pi steps in its
#: alpha (``unwrap_HM`` in ``prolong_euler_angles_FD``, see
#: ``visualization/teob_alpha_unwrap_bug.md``), and its cubic spline of alpha
#: rings across each: the bins where any multipole reads alpha within this
#: many spline intervals of a step are masked out of the comparison.
ALPHA_STEP_MARGIN = 6


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

    The phases are referenced to the merger with the (2,2) of the same
    EOB call, as the surrogate's are (see
    :meth:`ModeModel.merger_reference`).

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

    reference_model = model.mode_models[Mode(2, 2)]
    reference_knots = reference_model.downsampling_indices.phase_indices
    knots_hz = mode_model.dataset.frequencies_hz[downsampling.phase_indices]

    valid_indices = []
    true_amp_list = []
    true_phase_list = []
    for j, wp in enumerate(waveform_params_list):
        try:
            batched = generator.all_modes_amplitude_phase(
                wp, model.modes, frequencies_natural
            )
            _, amp_full, phase_full = batched[mode]
            _, _, phase_22 = batched[Mode(2, 2)]
        except Exception:  # pragma: no cover - EOB blowups
            continue
        if len(amp_full) != len(frequencies_natural) or not (
            np.all(np.isfinite(amp_full)) and np.all(np.isfinite(phase_full))
        ):
            continue
        slope, intercept = reference_model.merger_reference(
            wp, phase_nodes=phase_22[reference_knots]
        )
        valid_indices.append(j)
        true_amp_list.append(amp_full[downsampling.amplitude_indices])
        true_phase_list.append(
            phase_full[downsampling.phase_indices]
            - slope * knots_hz
            - mode.m / 2 * intercept
        )

    if len(valid_indices) < min_valid:
        raise RuntimeError(
            f"The batched EOB sweep produced fewer than {min_valid} valid "
            "waveforms; try a larger n_waveforms."
        )

    if len(valid_indices) != len(waveform_params_list):
        parameter_set = parameter_set[valid_indices]

    return np.stack(true_amp_list), np.stack(true_phase_list), parameter_set


def production_phases(model: Model, mode: Mode, parameter_set, frequencies_hz):
    """The surrogate's phases of ``mode`` at the reference total mass, as
    :meth:`Model.predict_amplitude_phase_mode` returns them."""
    mode_model = model.mode_models[mode]
    phases = []
    for wp in parameter_set.waveform_parameters(mode_model.dataset):
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
        phases.append(
            model.predict_amplitude_phase_mode(mode, frequencies_hz, extrinsic_params)[1]
        )
    return np.stack(phases)


def mlgw_eob_residuals(model: Model, mode: Mode, n_waveforms: int):
    r"""Return the mlgw-vs-EOB amplitude and phase residuals for one mode.

    Both waveform sets are taken in the model's own downsampled
    amplitude/phase representation; the ground truth comes from
    :func:`batched_true_waveforms`, referenced to the merger. The phase
    residual is the surrogate's phase, from
    :meth:`Model.predict_amplitude_phase_mode` (the production path,
    referenced to the merger with the surrogate's own (2,2)), minus
    ``phi_EOB``: nothing is fitted, it is the error in the phase the
    surrogate returns.

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
    phase_residuals : np.ndarray
        Shape ``(n_waveforms, n_phase_points)``, see above.
    true_amplitudes : np.ndarray
        ``A_EOB``, shape ``(n_waveforms, n_amp_points)``.
    parameter_set : ParameterSet
        The (filtered) parameters these residuals correspond to.
    """
    mode_model = model.mode_models[mode]
    validator = ValidateModel(mode_model)
    parameter_set = validator.param_set(n_waveforms, SEED)

    true_amplitudes, true_phases, parameter_set = batched_true_waveforms(
        model, mode, parameter_set
    )
    predicted_waveforms = validator.predicted_waveforms(parameter_set)

    downsampling = mode_model.downsampling_indices
    frequencies_hz = mode_model.dataset.frequencies_hz
    phase_freqs = frequencies_hz[downsampling.phase_indices]

    amplitude_residuals = 2 * (
        predicted_waveforms.amplitudes - true_amplitudes
    ) / (
        np.abs(predicted_waveforms.amplitudes)
        + np.abs(true_amplitudes)
    )

    phase_residuals = (
        production_phases(model, mode, parameter_set, phase_freqs) - true_phases
    )

    return (
        frequencies_hz[downsampling.amplitude_indices],
        phase_freqs,
        amplitude_residuals,
        phase_residuals,
        true_amplitudes,
        parameter_set,
    )


def global_alignment(residuals: dict, psd) -> dict:
    r"""Remove one time shift and one coalescence phase from every mode of
    each waveform.

    A time shift :math:`\Delta t` moves the phase of every mode by
    :math:`2 \pi f \Delta t`, a rotation of the orbital phase by
    :math:`\phi_c` that of the :math:`(\ell, m)` mode by :math:`m \phi_c`;
    these are the only phase errors the full waveform can absorb. For each
    waveform, they are fitted by least squares to the phase residuals of
    all modes at once: uniformly over each mode's phase nodes, each mode
    weighted by its PSD-weighted power :math:`\int A_{\rm EOB}^2 \, df /
    S_n`, so the fit is effectively the (2,2)'s. Weighting each node as it
    enters the mismatch instead (by :math:`A^2 \Delta f / S_n`) pins the
    fit to the lowest tens of hertz (to ~1 mrad there) and leaves a few
    tenths of a radian more at a kilohertz, where the mismatch, but not
    this plot, is nearly blind.

    Parameters
    ----------
    residuals : dict
        ``{mode: (amp_f, phi_f, amp_res, phi_res, true_amplitudes,
        param_set)}``, as :func:`mlgw_eob_residuals` returns, already
        restricted to the waveforms common to every mode, in the same order.
    psd : callable
        The PSD as a function of frequency in Hz.

    Returns
    -------
    dict
        ``{mode: phi_res - 2 pi f dt - m phi_c}`` and the fitted ``"dt"``,
        ``"phi_c"`` arrays, one entry per waveform.
    """
    n_waveforms = len(next(iter(residuals.values()))[3])
    rows = []  # per mode: design matrix, weights
    for mode, (amp_f, phi_f, _, phi_res, true_amp, _) in residuals.items():
        design = np.stack([2 * np.pi * phi_f, np.full_like(phi_f, mode.m)], axis=1)
        log_amp = np.log(np.maximum(np.abs(true_amp), 1e-300))
        amp_at_phi = np.exp(
            np.stack([np.interp(np.log(phi_f), np.log(amp_f), row) for row in log_amp])
        )
        power = np.sum(amp_at_phi**2 * np.gradient(phi_f) / psd(phi_f), axis=1)
        weights = np.repeat((power / len(phi_f))[:, None], len(phi_f), axis=1)
        rows.append((mode, design, weights, phi_res))

    dt = np.empty(n_waveforms)
    phi_c = np.empty(n_waveforms)
    for j in range(n_waveforms):
        normal = sum(design.T @ (w[j, :, None] * design) for _, design, w, _ in rows)
        rhs = sum(design.T @ (w[j] * res[j]) for _, design, w, res in rows)
        dt[j], phi_c[j] = np.linalg.solve(normal, rhs)

    aligned = {
        mode: res - dt[:, None] * design[None, :, 0] - mode.m * phi_c[:, None]
        for mode, design, _, res in rows
    }
    aligned["dt"] = dt
    aligned["phi_c"] = phi_c
    return aligned


def plot_residuals(model: Model) -> dict:
    """Plot the per-mode mlgw-EOB residuals; return them keyed by mode.

    Three rows:

    1. ``2 (A_mlgw - A_EOB) / (|A_mlgw| + |A_EOB|)`` --- the fractional
       amplitude error, bounded to ``[-2, 2]`` through the odd-m modes'
       amplitude nodes;
    2. ``phi_mlgw - phi_EOB``, both referenced to their own merger,
       nothing removed afterwards --- the error in the phase the surrogate
       returns;
    3. the same with one time shift and one coalescence phase per waveform
       removed from all its modes at once (:func:`global_alignment`) ---
       the phase error the full waveform cannot absorb.

    The waveforms are those for which every mode's EOB call succeeded.
    """
    fig, axes = plt.subplots(
        3, len(MODES), figsize=(6 * len(MODES), 10), squeeze=False
    )

    cmap = matplotlib.colormaps["viridis"]
    q_min, q_max = model.dataset.parameter_ranges.q_range

    residuals = {
        mode: mlgw_eob_residuals(model, mode, N_RESIDUAL_WAVEFORMS) for mode in MODES
    }
    # keep the waveforms common to every mode, in one order
    common = set.intersection(*(
        {tuple(row) for row in values[5].parameter_array} for values in residuals.values()
    ))
    for mode, values in residuals.items():
        keep = np.array([tuple(row) in common for row in values[5].parameter_array])
        order = np.lexsort(values[5].parameter_array[keep].T)
        residuals[mode] = tuple(
            v if i < 2 else (
                v[keep][order] if i < 5 else values[5].parameter_array[keep][order]
            )
            for i, v in enumerate(values)
        )
    parameter_array = residuals[MODES[0]][5]

    psd = ValidateModel(model.mode_models[Mode(2, 2)]).psd_at_frequencies
    aligned = global_alignment(residuals, psd)
    print(
        f"  global alignment over {len(parameter_array)} waveforms: "
        f"|dt| median {np.median(np.abs(aligned['dt'])) * 1e6:.1f} us, "
        f"|phi_c| median {np.median(np.abs(aligned['phi_c'])):.3f} rad"
    )

    for i, mode in enumerate(MODES):
        amp_f, phi_f, amp_res, phi_res, _, _ = residuals[mode]
        for j in range(len(amp_res)):
            q = parameter_array[j, 0]
            color = cmap((q - q_min) / (q_max - q_min))
            axes[0, i].plot(amp_f, amp_res[j], color=color, alpha=0.6, linewidth=0.8)
            axes[1, i].plot(phi_f, phi_res[j], color=color, alpha=0.6, linewidth=0.8)
            axes[2, i].plot(phi_f, aligned[mode][j], color=color, alpha=0.6, linewidth=0.8)

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
        r"$\phi_{\rm mlgw} - \phi_{\rm EOB}$, one $\Delta t$ and"
        "\n" r"$\phi_c$ per waveform removed from all modes [rad]"
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
        f"mlgw-EOB reconstruction residuals, {len(parameter_array)} waveforms "
        "from the training distribution"
    )

    outfile = f"{OUTPUT_PREFIX}_residuals.png"
    fig.savefig(outfile, dpi=150)
    print(f"Saved plot to {outfile}")

    return residuals


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
    path (:meth:`Model.predict_amplitude_phase_mode`, referenced to the
    merger with the surrogate's own (2,2)), and the overlap uses the
    *real* Wiener product --- the phase is not marginalised either. This
    is the mismatch counterpart of the residuals plot's middle row: it says
    how good the model is including its merger time and coalescence phase.

    The ground truth is the *batched* multi-mode EOB waveform
    (:func:`batched_true_waveforms`), referenced to the merger with its own
    (2,2), as the residuals plot uses.

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

    mode_model = validator.model
    phase_freqs = mode_model.dataset.frequencies_hz[
        mode_model.downsampling_indices.phase_indices
    ]
    predicted_waveforms.phases = production_phases(
        model, mode, parameter_set, phase_freqs
    )

    true_cartesian, predicted_cartesian = validator.waveforms(
        true_waveforms, predicted_waveforms
    )
    weight = np.gradient(validator.frequencies) / validator.psd_values

    def inner(a, b):
        return np.sum(np.conj(a) * b * weight, axis=-1).real

    # Real Wiener product: the reference phase is *not* marginalised. This is
    # the mismatch counterpart of the residuals plot's middle row, so the two
    # compare the same predicted waveform (nothing optimised).
    overlap = inner(true_cartesian, predicted_cartesian) / np.sqrt(
        inner(true_cartesian, true_cartesian)
        * inner(predicted_cartesian, predicted_cartesian)
    )
    return 1.0 - overlap


def optimised_mismatches(validator: ValidateModel, parameter_set):
    r"""Single-mode mismatch for a given parameter set, residual time+phase
    marginalised --- i.e. the same computation as
    ``validator.validation_mismatches(...)``, but
    for an already-drawn ``parameter_set`` rather than one it draws itself.
    Used by :func:`mismatch_vs_power_by_mode` to pair this up with a
    specific sample.
    """
    true_waveforms, parameter_set = validator.true_waveforms(parameter_set)
    true_waveforms = validator.merger_referenced(true_waveforms)
    predicted_waveforms = validator.merger_referenced(
        validator.predicted_waveforms(parameter_set)
    )
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
        validator = ValidateModel(model.mode_models[mode])
        optimised = np.array(
            validator.validation_mismatches(
                N_MISMATCH_WAVEFORMS, seed=SEED
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


def extended_frequency_grid(
    validator: ValidateModel, low_freq_hz: float = LOW_FREQUENCY_HZ
):
    r"""``validator``'s own PSD grid, extended down to ``low_freq_hz``.

    :class:`ValidateModel` deliberately keeps :attr:`ValidateModel.frequencies`
    at or above the trained band's own edge (see its docstring): its
    ``true_waveforms``/``predicted_waveforms`` resample the trained nodes,
    and going lower would extrapolate them. But ``Model.predict`` /
    ``predict_modes_dict`` --- the production entry points this script is
    validating --- handle frequencies below that edge themselves, via each
    mode's post-Newtonian extension (see :data:`LOW_FREQUENCY_HZ`). This
    builds a wider ``(frequencies, psd_values)`` pair, sourced from
    ``validator``'s own (otherwise-unmasked) ``psd_data``, for callers that
    want to exercise that code path instead.

    Returns
    -------
    frequencies, psd_values : np.ndarray
        Both restricted to ``[low_freq_hz, validator.frequencies[-1]]``.
    """
    all_frequencies = validator.psd_data[:, 0]
    mask = (all_frequencies >= low_freq_hz) & (
        all_frequencies <= validator.frequencies[-1]
    )
    return all_frequencies[mask], validator.psd_data[:, 1][mask]


def full_waveform_mismatches(model: Model) -> tuple:
    r"""Compute the multi-mode full-waveform mismatch distribution.

    Compares :meth:`Model.predict_modes_dict` against the EOB ground
    truth from :meth:`Model.get_teob_modes_dict`, restricted to the
    band where the EOB waveform is actually defined (it is zero-padded
    below its starting frequency). The frequency grid itself
    (:func:`extended_frequency_grid`) reaches down to
    :data:`LOW_FREQUENCY_HZ`, well below where any mode has EOB support,
    so that every ``predict_modes_dict`` call here exercises each mode's
    post-Newtonian low-frequency extension; the *mismatch numbers* are
    unaffected (the EOB-support masking below still excludes that region),
    but a broken splice there --- e.g. a NaN or a discontinuity large
    enough to leak into the supported band --- would otherwise pass
    unnoticed, since :class:`ValidateModel`'s own machinery never
    evaluates that code path at all. A non-finite value anywhere in the
    predicted waveform raises immediately (see the assertion below).

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
    frequencies, psd_values = extended_frequency_grid(validator)

    parameter_generator = model.dataset.make_parameter_generator(SEED)

    def inner_product(a: np.ndarray, mask: np.ndarray) -> float:
        """PSD-weighted power of a complex waveform over the support."""
        return float(
            np.abs(
                np.trapezoid(
                    np.conj(a[mask]) * a[mask] / psd_values[mask],
                    x=frequencies[mask],
                )
            )
        )

    def real_wiener_mismatch(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
        """``1 - Re(a|b) / sqrt((a|a)(b|b))`` --- nothing optimised."""
        fm = frequencies[mask]
        psd_m = psd_values[mask]

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

        for key, mode_array in predicted.items():
            if not np.all(np.isfinite(mode_array)):
                raise RuntimeError(
                    f"Non-finite values in the {key} mode's prediction down "
                    f"to {LOW_FREQUENCY_HZ} Hz (q={intrinsic.mass_ratio:.3f}) "
                    "-- the post-Newtonian low-frequency extension is broken."
                )

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
    :class:`ValidateModel`, ``optimised`` using the same
    routine as ``validation_mismatches(...)``
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
        mode: ValidateModel(model.mode_models[mode])
        for mode in MODES
    }
    power_validator = ValidateModel(model.mode_models[Mode(2, 2)])
    # Down to LOW_FREQUENCY_HZ, like full_waveform_mismatches: exercises
    # the post-Newtonian low-frequency extension in predict_modes_dict, and
    # lets a mode's power fraction reflect its own EOB starting threshold
    # if that happens to fall in [LOW_FREQUENCY_HZ, effective_initial_frequency_hz)
    # (e.g. (2,1)'s, at m/2 * f0).
    frequencies, psd_values = extended_frequency_grid(power_validator)

    parameter_generator = model.dataset.make_parameter_generator(SEED)

    def power(a: np.ndarray, mask: np.ndarray) -> float:
        """PSD-weighted power of a complex waveform over the support."""
        weight = np.gradient(frequencies[mask]) / psd_values[mask]
        return float(np.abs(np.sum(np.conj(a[mask]) * a[mask] * weight)))

    results: dict = {
        mode: {"power_fraction": [], "optimised": [], "regressed": []}
        for mode in MODES
    }

    for _ in tqdm(range(n_waveforms), unit="waveform"):
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
    regressed points (merger-referenced, nothing optimised) are drawn
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
            s=14,
            alpha=1.0,
            marker="o",
            linewidths=0,
            label=label,
        )
        ax.scatter(
            per_mode["power_fraction"],
            per_mode["regressed"],
            color=color,
            s=14,
            marker="o",
            alpha=0.25,
            linewidths=0,
        )

    ax.set_xscale("logit")
    ax.set_yscale("log")
    ax.set_xlabel("Power fraction, $\\max$(EOB, mlgw)")
    ax.set_ylabel("Mismatch")
    ax.set_title(
        "full: residual time+phase optimised    "
        r"faint: predicted $\Delta t$ + reference phase, nothing optimised (regressed)",
        fontsize="small",
    )
    ax.grid(True)
    ax.legend()
    fig.suptitle("Per-mode mismatch vs. power fraction")
    fig.tight_layout()
    ax.set_xlim(ax.get_xlim()[0], 1-1e-5)

    outfile = f"{OUTPUT_PREFIX}_mismatch_vs_power.png"
    fig.savefig(outfile, dpi=150)
    print(f"Saved plot to {outfile}")


def _mode_bases(model: Model, params: ParametersWithExtrinsic, frequencies: np.ndarray):
    r"""Per-mode EOB/surrogate amplitude and phase, before :math:`Y_{\ell m}`.

    Both the reference-phase rotation (:math:`m \varphi_c`) and the
    inclination-dependent :math:`{}_{-2}Y_{\ell m}(\iota,\varphi)` weight
    are cheap to apply after the fact (see :func:`_mode_cartesian`), so
    this factors out the two expensive parts --- the EOB call and the
    surrogate NN inference --- computed once per intrinsic draw with
    ``params.coalescence_phase`` forced to zero. :func:`mode_subset_data`
    then reuses these bases across every sampled inclination/phase pair,
    rather than re-running the EOB generator or the NN once per
    orientation.

    Mirrors :meth:`Model.get_teob_modes_dict` (for the truth, via
    :meth:`Model.teob_modes_amp_phase`) and :meth:`Model.predict_modes_dict`
    (for the surrogate); both apply ``params.coalescence_phase``, hence
    forcing it to zero here.

    Returns
    -------
    amp_true, phase_true, amp_pred, phase_pred : dict[Mode, np.ndarray]
        One array per mode in ``model.modes``.
    support : np.ndarray
        Boolean mask, ``True`` where every mode has nonzero EOB truth
        amplitude (i.e. where the EOB waveform is not zero-padded).
    """
    assert params.coalescence_phase == 0.0

    truth = model.teob_modes_amp_phase(frequencies, params)
    amp_true = {mode: truth[mode][0] for mode in model.modes}
    phase_true = {mode: truth[mode][1] for mode in model.modes}

    merger_reference = model.merger_reference(params)
    amp_pred = {}
    phase_pred = {}
    for mode in model.modes:
        amp_pred[mode], phase_pred[mode] = model.mode_models[
            mode
        ].predict_amplitude_phase(frequencies, params, merger_reference=merger_reference)

    support = np.ones(len(frequencies), dtype=bool)
    for mode in model.modes:
        support &= np.abs(amp_true[mode]) > 0

    return amp_true, phase_true, amp_pred, phase_pred, support


def _mode_cartesian(amp: np.ndarray, phase: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    r"""One mode's :math:`h_+ + i h_\times` contribution, given its coefficients.

    ``coeffs`` (shape ``(8,)``) are the per-mode spherical-harmonic
    coefficients built by :func:`~mlgw_bns.model._build_mode_coeffs`,
    already encoding the requested inclination; this applies the same
    combination as :meth:`Model.predict_modes_dict` /
    :meth:`Model.get_teob_modes_dict`. The overall :math:`1/(2\eta)`
    normalisation both of those apply is omitted --- it is a real,
    positive, waveform-independent factor and so cancels out of every
    mismatch computed from the result.
    """
    cosphi = np.cos(phase)
    sinphi = np.sin(phase)
    h_plus = amp * (cosphi * coeffs[0] + sinphi * coeffs[1]) + 1j * amp * (
        cosphi * coeffs[2] + sinphi * coeffs[3]
    )
    h_cross = amp * (cosphi * coeffs[4] + sinphi * coeffs[5]) + 1j * amp * (
        cosphi * coeffs[6] + sinphi * coeffs[7]
    )
    return h_plus + 1j * h_cross


def _mode_subset_bases_cache_path(n_waveforms: int) -> str:
    """Where :func:`mode_subset_data` caches its expensive per-waveform bases.

    Keyed by ``OUTPUT_PREFIX`` (so different models don't collide) and by
    the number of waveforms it holds.
    """
    return f"{OUTPUT_PREFIX}_mode_subset_bases_n{n_waveforms}.pkl"


def _load_mode_subset_bases_cache() -> Optional[dict]:
    """Load the largest on-disk bases cache for the current ``OUTPUT_PREFIX``.

    Returned regardless of how it compares to the caller's requested
    ``n_waveforms``: :func:`mode_subset_data` truncates it if it is
    larger, or tops it up (continuing the same parameter-generator
    stream) if it is smaller.
    """
    candidates = []
    prefix = f"{OUTPUT_PREFIX}_mode_subset_bases_n"
    directory = os.path.dirname(prefix) or "."
    base = os.path.basename(prefix)
    for name in os.listdir(directory):
        if name.startswith(base) and name.endswith(".pkl"):
            try:
                count = int(name[len(base) : -len(".pkl")])
            except ValueError:
                continue
            candidates.append((count, os.path.join(directory, name)))
    if not candidates:
        return None
    _, path = max(candidates)
    with open(path, "rb") as f:
        cached = pickle.load(f)
    print(f"  Loaded {len(cached['per_waveform'])} cached waveform bases from {path}")
    return cached


def mode_subset_data(
    model: Model,
    n_waveforms: int = N_MODE_SUBSET_WAVEFORMS,
    n_orientations: int = N_MODE_SUBSET_ORIENTATIONS,
    use_cache: bool = True,
) -> dict:
    r"""Full-waveform mismatch vs. how many modes the surrogate reconstructs.

    Compares the *full* (all-modes) EOB truth against the surrogate
    restricted to a cumulative subset of modes --- (2,2) only; (2,2) plus
    the next most powerful mode; and so forth --- to see how many modes
    are actually needed for an accurate reconstruction. The cumulative
    order is determined empirically, ranking :attr:`Model.modes` by their
    median PSD-weighted EOB power (the same quantity reported by
    :func:`full_waveform_mismatches`'s ``power_fractions``, but computed
    directly here since the per-mode EOB amplitudes are already at hand).

    Per :func:`_mode_bases`, the expensive per-mode EOB/surrogate
    evaluation happens once per intrinsic (mass/spin) draw; inclination
    and coalescence phase are then resampled ``n_orientations`` times
    per draw as a cheap postprocessing step (an analytic
    :math:`{}_{-2}Y_{\ell m}(\iota,\varphi)` reweighting plus an
    :math:`m \varphi_c` phase rotation), and only *that* recombination
    feeds into the (expensive) time-and-phase-optimised
    :meth:`~mlgw_bns.model_validation.ValidateModel.full_waveform_mismatch`
    call, once per (waveform, orientation, mode subset) triple.

    Inclination is sampled isotropically (:math:`\cos\iota \sim
    \mathrm{Uniform}(-1, 1)`) and the coalescence phase uniformly on
    :math:`[0, 2\pi)`, matching the usual priors for these angles.

    The per-waveform bases (the EOB call plus the surrogate NN
    inference --- the only part of this function that isn't cheap
    postprocessing) are cached to disk, keyed by ``OUTPUT_PREFIX`` and
    ``n_waveforms`` (:func:`_mode_subset_bases_cache_path`); a rerun at
    the same or a smaller ``n_waveforms`` reuses that cache instead of
    recomputing it, and a larger one tops it up (advancing the same
    parameter generator from where the cached run left off, so the
    combined sequence matches what a single from-scratch run at the
    larger count would have drawn). Pass ``use_cache=False`` to force a
    recompute. Progress is logged via ``tqdm`` for both the basis
    computation and the (much slower) orientation/subset sweep.

    Returns
    -------
    dict
        ``{"subset_labels": [...], "cumulative_power": [...],
        "mode_power_fraction": {mode: ...},
        "results": {label: {"mismatch": ..., "delta_t": ...,
        "delta_phi": ...}}}``, arrays one entry per
        (waveform, orientation) pair that succeeded.
    """
    reference_model = model.mode_models[Mode(2, 2)]
    validator = ValidateModel(reference_model)
    frequencies, psd_values = extended_frequency_grid(validator)

    parameter_generator = model.dataset.make_parameter_generator(SEED)

    cached = _load_mode_subset_bases_cache() if use_cache else None
    if cached is not None:
        per_waveform = list(cached["per_waveform"])
        mode_power = {mode: list(values) for mode, values in cached["mode_power"].items()}
        n_draws_consumed = cached["n_draws_consumed"]
        if len(per_waveform) > n_waveforms:
            # More than requested: truncate. `n_draws_consumed` then no
            # longer matches this prefix, but that's harmless --- with
            # `len(per_waveform) == n_waveforms` already, the top-up
            # below never runs, so it's never read.
            per_waveform = per_waveform[:n_waveforms]
            mode_power = {mode: values[:n_waveforms] for mode, values in mode_power.items()}
    else:
        per_waveform = []
        mode_power = {mode: [] for mode in model.modes}
        n_draws_consumed = 0

    if len(per_waveform) < n_waveforms:
        print(
            f"  Computing per-mode EOB/surrogate bases for "
            f"{n_waveforms - len(per_waveform)} more waveforms "
            f"({len(per_waveform)} cached)..."
        )
        for _ in range(n_draws_consumed):
            next(parameter_generator)
        with tqdm(total=n_waveforms, initial=len(per_waveform), unit="waveform") as pbar:
            while len(per_waveform) < n_waveforms:
                intrinsic = next(parameter_generator)
                n_draws_consumed += 1
                params0 = ParametersWithExtrinsic(
                    mass_ratio=intrinsic.mass_ratio,
                    lambda_1=intrinsic.lambda_1,
                    lambda_2=intrinsic.lambda_2,
                    chi_1=intrinsic.chi_1,
                    chi_2=intrinsic.chi_2,
                    distance_mpc=DISTANCE_MPC,
                    inclination=0.0,
                    coalescence_phase=0.0,
                    total_mass=TOTAL_MASS,
                )
                try:
                    amp_true, phase_true, amp_pred, phase_pred, support = _mode_bases(
                        model, params0, frequencies
                    )
                except Exception:  # pragma: no cover - EOB blowups
                    continue

                if support.sum() < 2 or not all(
                    np.all(np.isfinite(amp_pred[m])) and np.all(np.isfinite(phase_pred[m]))
                    for m in model.modes
                ):
                    continue

                for mode in model.modes:
                    mode_power[mode].append(
                        float(
                            np.trapezoid(
                                amp_true[mode][support] ** 2 / psd_values[support],
                                x=frequencies[support],
                            )
                        )
                    )
                per_waveform.append((amp_true, phase_true, amp_pred, phase_pred, support))
                pbar.update(1)

        if use_cache:
            cache_path = _mode_subset_bases_cache_path(len(per_waveform))
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "per_waveform": per_waveform,
                        "mode_power": mode_power,
                        "n_draws_consumed": n_draws_consumed,
                    },
                    f,
                )
            print(f"  Cached {len(per_waveform)} waveform bases to {cache_path}")

    ordered_modes = sorted(model.modes, key=lambda m: -np.median(mode_power[m]))
    total_power = sum(np.median(mode_power[m]) for m in model.modes)
    mode_power_fraction = {
        mode: np.median(mode_power[mode]) / total_power for mode in model.modes
    }

    subsets = [ordered_modes[: k + 1] for k in range(len(ordered_modes))]
    subset_labels = ["+".join(f"({m.l},{m.m})" for m in subset) for subset in subsets]
    cumulative_power = [
        sum(mode_power_fraction[m] for m in subset) for subset in subsets
    ]

    results = {
        label: {"mismatch": [], "delta_t": [], "delta_phi": []}
        for label in subset_labels
    }

    orientation_rng = np.random.default_rng(MODE_SUBSET_ORIENTATION_SEED)

    print(
        f"  Sweeping {n_orientations} inclination/phase draws per waveform, "
        f"across {len(subsets)} mode subsets..."
    )
    sweep_progress = tqdm(
        total=len(per_waveform) * n_orientations, unit="orientation"
    )
    for amp_true, phase_true, amp_pred, phase_pred, support in per_waveform:
        support_freqs = frequencies[support]
        for _ in range(n_orientations):
            iota = np.arccos(orientation_rng.uniform(-1.0, 1.0))
            phi_c = orientation_rng.uniform(0.0, 2 * np.pi)

            Ylm_real, Ylm_imag, Ylm_real_mneg, Ylm_imag_mneg = model._compute_Ylm_modes(
                modes=model.modes, phi=0.0, iota=iota
            )
            coeffs = _build_mode_coeffs(
                model.modes,
                list(range(len(model.modes))),
                Ylm_real,
                Ylm_imag,
                Ylm_real_mneg,
                Ylm_imag_mneg,
            )

            true_modes = {}
            pred_modes = {}
            for idx, mode in enumerate(model.modes):
                c = coeffs[idx]
                key = (mode.l, mode.m)
                true_modes[key] = _mode_cartesian(
                    amp_true[mode][support],
                    phase_true[mode][support] + mode.m * phi_c,
                    c,
                )
                pred_modes[key] = _mode_cartesian(
                    amp_pred[mode][support],
                    phase_pred[mode][support] + mode.m * phi_c,
                    c,
                )

            for subset, label in zip(subsets, subset_labels):
                pred_subset = {(m.l, m.m): pred_modes[(m.l, m.m)] for m in subset}
                mismatch, delta_t, delta_phi = validator.full_waveform_mismatch(
                    modes_1=true_modes,
                    modes_2=pred_subset,
                    frequencies=support_freqs,
                    return_shifts=True,
                )
                # `full_waveform_mismatch`'s phi_grid spans [-2pi, 2pi] ---
                # twice the true period, since every mode's e^{i m phi_c}
                # is 2pi-periodic for integer m --- so phi=0, -2pi and
                # +2pi are all physically identical alignments but land
                # at wildly different raw numbers. Left unwrapped, that
                # aliasing alone makes this column's boxplot look like a
                # uniform, badly-constrained phase when the true residual
                # offset is small and tightly clustered near zero. Wrap to
                # the canonical (-pi, pi] before storing.
                delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
                results[label]["mismatch"].append(mismatch)
                results[label]["delta_t"].append(delta_t)
                results[label]["delta_phi"].append(delta_phi)
            sweep_progress.update(1)
    sweep_progress.close()

    return {
        "subset_labels": subset_labels,
        "cumulative_power": cumulative_power,
        "mode_power_fraction": mode_power_fraction,
        "n_waveforms": len(per_waveform),
        "n_orientations": n_orientations,
        "results": {
            label: {key: np.array(values) for key, values in per_label.items()}
            for label, per_label in results.items()
        },
    }


def plot_mode_subset_boxplots(data: dict) -> None:
    r"""Boxplot the mismatch and required alignment vs. mode subset.

    ``data`` is the output of :func:`mode_subset_data`. Three rows
    (height ratios 2:1:1), one boxplot column per cumulative mode
    subset:

    1. full-waveform mismatch (time and reference phase optimised),
       log-scaled;
    2. the time shift :math:`\Delta t` that optimisation needed to
       align the partial surrogate reconstruction to the full EOB
       truth;
    3. the same for the reference-phase shift :math:`\Delta\varphi`.

    Boxes span the 25-75 interquartile range; whiskers extend to the
    5th/95th percentiles (``whis=[5, 95]``), with outliers beyond that
    not drawn separately (``showfliers=False``) since with this many
    samples they would just clutter the plot.
    """
    subset_labels = data["subset_labels"]
    results = data["results"]
    cumulative_power = data["cumulative_power"]

    n_cols = len(subset_labels)
    positions = np.arange(1, n_cols + 1)

    fig, axes = plt.subplots(
        3, 1, figsize=(1.7 * n_cols + 2, 9.5),
        gridspec_kw={"height_ratios": [2, 1, 1]},
        sharex=True,
    )

    mismatches = [results[label]["mismatch"] for label in subset_labels]
    # Rows 2/3 report the *magnitude* of the required alignment shift on
    # a log axis --- what matters here is how large a correction is
    # needed, not its sign, and the two point in genuinely different
    # directions across waveforms (unlike the mismatch, which is
    # positive by construction).
    delta_t_us = [np.abs(results[label]["delta_t"]) * 1e6 for label in subset_labels]
    delta_phi = [np.abs(results[label]["delta_phi"]) for label in subset_labels]

    box_kwargs = dict(whis=[5, 95], showfliers=False, widths=0.6)

    axes[0].boxplot(mismatches, positions=positions, **box_kwargs)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Full-waveform mismatch\n(time + phase optimised)")

    axes[1].boxplot(delta_t_us, positions=positions, **box_kwargs)
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"$|\Delta t|$ [$\mu$s]")

    axes[2].boxplot(delta_phi, positions=positions, **box_kwargs)
    axes[2].set_yscale("log")
    axes[2].set_ylabel(r"$|\Delta \varphi|$ [rad]")

    for ax in axes:
        ax.grid(True, which="both", axis="y", lw=0.3)

    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(
        [
            f"{label}\n[{_sci_latex(max(1.0 - power, 0.0))} power missing]"
            for label, power in zip(subset_labels, cumulative_power)
        ],
        rotation=30, ha="right", fontsize="small",
    )
    axes[2].set_xlabel("Cumulative mode subset (surrogate), ordered by EOB power")

    fig.suptitle(
        "Full-waveform mismatch vs. number of modes reconstructed "
        f"({data['n_waveforms']} waveforms, {data['n_orientations']} "
        "inclination/phase draws each)"
    )
    fig.tight_layout()

    outfile = f"{OUTPUT_PREFIX}_mode_subset.png"
    fig.savefig(outfile, dpi=150)
    print(f"Saved plot to {outfile}")


def report_weighted_mismatches(
    mismatches_by_mode: dict, power_fractions: dict, full_mismatches: np.ndarray
) -> None:
    """Print per-mode mismatches next to each mode's share of the power.

    ``optimised`` is the canonical per-mode mismatch (residual time and
    phase optimised); ``pred-shift`` compares the merger-referenced
    phases instead, with nothing optimised. The ``product`` column is ``optimised`` times the
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
            # `\%` --- a literal `%` would otherwise be read as a LaTeX
            # comment character now that `text.usetex` is on globally.
            label += rf"  [{np.median(power_fractions[mode]) * 100:.3f}\% of power]"
        plot_kde(0, optimised, track=True, linewidth=2.0, color=color, label=label)
        plot_kde(1, regressed, track=True, linewidth=2.0, color=color, label=label)

    plot_kde(0, full_mismatches, linewidth=2.4, linestyle="--",
             color="black", label="full waveform")
    plot_kde(1, full_mismatches_no_opt, linewidth=2.4, linestyle="--",
             color="black", label="full waveform")

    axes[0].set_title("residual time + reference phase optimised", fontsize="small")
    axes[1].set_title(
        r"surrogate's predicted $\Delta t$ + reference phase applied, nothing optimised",
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


def random_precessing_parameters(
    model: Model, seed: int
) -> PrecessingParametersWithExtrinsic:
    r"""A precessing binary: :func:`random_parameters`' intrinsic point
    (its aligned spins become the spins' :math:`z` components), an in-plane
    spin of magnitude uniform in ``[0, PRECESSING_MAX_IN_PLANE_SPIN]`` and
    random direction for each body, an isotropic line of sight, and the
    spins given at ``PRECESSING_REFERENCE_FREQUENCY_HZ``."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from benchmark_evaluation_time import random_parameters

    aligned = random_parameters(model, seed)
    rng = np.random.default_rng([seed, 1])

    def spin(chi_z: float) -> tuple:
        magnitude = rng.uniform(0.0, PRECESSING_MAX_IN_PLANE_SPIN)
        angle = rng.uniform(0.0, 2 * np.pi)
        return (magnitude * np.cos(angle), magnitude * np.sin(angle), chi_z)

    return PrecessingParametersWithExtrinsic(
        mass_ratio=aligned.mass_ratio,
        lambda_1=aligned.lambda_1,
        lambda_2=aligned.lambda_2,
        chi_1=spin(aligned.chi_1),
        chi_2=spin(aligned.chi_2),
        distance_mpc=DISTANCE_MPC,
        inclination=float(np.arccos(rng.uniform(-1.0, 1.0))),
        azimuth=rng.uniform(0.0, 2 * np.pi),
        total_mass=TOTAL_MASS,
        reference_frequency_hz=PRECESSING_REFERENCE_FREQUENCY_HZ,
    )


def teob_precessing_polarizations(
    model: Model,
    precessing: PrecessingModel,
    params: PrecessingParametersWithExtrinsic,
    frequencies: np.ndarray,
) -> tuple:
    r"""TEOBResumS' own precessing :math:`\tilde{h}_+, \tilde{h}_\times` at
    ``frequencies``, in the mlgw_bns Fourier convention, and a mask of the
    bins glitched by its alpha steps (see ``ALPHA_STEP_MARGIN``), as in
    ``docs/examples/precessing_vs_teobresums.py``: the model's co-precessing
    multipoles twisted into every inertial one up to :math:`\ell = 4`, the
    spins imposed where TEOBResumS' frequency-domain path imposes them (0.95
    times its ``initial_frequency``). Sets ``params.reference_phase`` to the
    one that reproduces the run (:meth:`PrecessingModel.teob_reference_phase`).
    """
    import tempfile

    from EOBRun_module import EOBRunPy

    from mlgw_bns.higher_order_modes import mode_to_k

    initial_frequency = params.reference_frequency_hz / 0.95
    largest_l = max(mode.l for mode in model.modes)
    output_dir = tempfile.TemporaryDirectory()
    _, hp_re, hp_im, hc_re, hc_im, _, hlm, dynamics = EOBRunPy(dict(
        M=params.total_mass,
        q=params.mass_ratio,
        LambdaAl2=params.lambda_1,
        LambdaBl2=params.lambda_2,
        chi1x=params.chi_1[0], chi1y=params.chi_1[1], chi1z=params.chi_1[2],
        chi2x=params.chi_2[0], chi2y=params.chi_2[1], chi2z=params.chi_2[2],
        distance=params.distance_mpc,
        inclination=params.inclination,
        coalescence_angle=np.pi / 2 + params.azimuth,
        use_spins=2,
        domain=1,
        initial_frequency=initial_frequency,
        srate_interp=4096.0,
        # TEOBResumS interpolates onto the requested grid, which must start
        # at its initial frequency; ~0.3 s instead of minutes on a uniform one
        interp_freqs="yes",
        freqs=[initial_frequency, *frequencies],
        use_geometric_units="no",
        output_hpc="no",
        arg_out="yes",
        use_mode_lm=sorted(mode_to_k(mode) for mode in model.modes),
        use_mode_lm_inertial=list(range(mode_to_k(Mode(largest_l, largest_l)) + 1)),
        output_dynamics="yes",
        output_dir=output_dir.name,
    ))
    with output_dir:
        path = os.path.join(output_dir.name, "dynspin.txt")
        # not written without in-plane spins, when nothing is twisted
        dynspin = np.loadtxt(path) if os.path.exists(path) else np.zeros((0, 14))
    # alpha as twist_hlm_FD splines it, against M Omega
    momega, alpha = dynspin[:, 13], dynspin[:, 10]
    steps = np.flatnonzero(np.abs(np.diff(alpha)) > np.pi)
    glitched = np.zeros(len(frequencies), dtype=bool)
    # the co-precessing (l, m) multipole reads the angles at the orbital
    # frequency f / m, so at M Omega = 2 pi f M / m
    momega_of_f = 2 * np.pi * np.asarray(frequencies) * params.total_mass * SUN_MASS_SECONDS
    for step in steps:
        low = momega[max(step - ALPHA_STEP_MARGIN, 0)]
        high = momega[min(step + 1 + ALPHA_STEP_MARGIN, len(momega) - 1)]
        for m in {mode.m for mode in model.modes}:
            glitched |= (momega_of_f / m >= low) & (momega_of_f / m <= high)
    params.reference_phase = precessing.teob_reference_phase(
        params,
        start_momega=dynamics["MOmega"][0],
        start_orbital_phase=dynamics["phi"][0],
        start_phase_22=hlm["1"][1][0],  # (l, m) = (2, 2)
    )
    # mlgw_bns' Fourier transform is the complex conjugate of TEOBResumS'
    hp = np.conj(np.asarray(hp_re) + 1j * np.asarray(hp_im))[1:]
    hc = np.conj(np.asarray(hc_re) + 1j * np.asarray(hc_im))[1:]
    return hp, hc, glitched


def precessing_mismatches(
    model: Model, n_waveforms: int = N_PRECESSING_WAVEFORMS
) -> dict:
    r"""Mismatches of the precessing model's JAX predictor
    (:meth:`PrecessingModel.jax_predict`, the one fast enough to use)
    against TEOBResumS' precessing :math:`\tilde{h}_+` and
    :math:`\tilde{h}_\times`.

    Each polarization is compared on its own, on the validator's grid
    between ``PRECESSING_MIN_FREQUENCY_HZ`` (where TEOBResumS has every
    inertial multipole up to :math:`m = 4`) and the top of the trained band,
    weighted by the validator's PSD, in two configurations:

    * **optimised**: time shift and constant phase maximised over;
    * **not optimised**: the real Wiener product as the model returns the
      polarizations, with the orbital phase anchored to TEOBResumS' run
      (:meth:`PrecessingModel.teob_reference_phase`) and nothing else
      tuned. It is sensitive to the relative sign and phase of
      :math:`h_+` and :math:`h_\times`, which the optimised one is not, but
      also to the error of the regressed merger time.

    Both leave out the bins TEOBResumS' alpha steps glitch
    (``ALPHA_STEP_MARGIN``), a few percent of them; the optimised mismatch
    over every bin is kept too (``*_all_bins``).

    Returns
    -------
    dict
        ``{"plus": [...], "cross": [...], "plus_no_optimisation": [...],
        "cross_no_optimisation": [...], "plus_all_bins": [...],
        "cross_all_bins": [...], "glitched_fraction": [...],
        "beta_max": [...], "params": [...]}``, one entry per binary;
        ``beta_max`` is the largest opening angle of the precession cone
        over the band.
    """
    import jax

    from dataclasses import fields

    from mlgw_bns.batched_precession import TabulatedAngles, batch_arguments, precession_angles

    precessing = PrecessingModel(model)
    predict = jax.jit(precessing.jax_predict())
    angles = jax.jit(precession_angles(model))
    state_index = [field.name for field in fields(TabulatedAngles)].index("state")
    validator = ValidateModel(model.mode_models[Mode(2, 2)])
    grid = validator.frequencies
    grid = grid[(grid >= PRECESSING_MIN_FREQUENCY_HZ) & (grid < grid[-1])]
    weight = np.gradient(grid) / validator.psd_at_frequencies(grid)

    def unoptimised(truth: np.ndarray, ours: np.ndarray, keep: np.ndarray) -> float:
        def product(a, b):
            return np.sum((np.conj(a) * b * weight)[keep]).real

        return 1 - product(truth, ours) / np.sqrt(
            product(truth, truth) * product(ours, ours)
        )

    result: dict = {key: [] for key in (
        "plus", "cross", "plus_no_optimisation", "cross_no_optimisation",
        "plus_all_bins", "cross_all_bins", "glitched_fraction", "beta_max", "params",
    )}
    for seed in tqdm(range(n_waveforms), unit="waveform"):
        params = random_precessing_parameters(model, seed)
        hp_teob, hc_teob, glitched = teob_precessing_polarizations(
            model, precessing, params, grid
        )
        keep = ~glitched
        hp, hc = (np.asarray(h[0]) for h in predict(*batch_arguments([params], grid)))
        for name, truth, ours in (("plus", hp_teob, hp), ("cross", hc_teob, hc)):
            result[name].append(validator.mismatch(
                truth[keep], ours[keep], frequencies=grid[keep], max_delta_t=1e-3
            ))
            result[f"{name}_all_bins"].append(
                validator.mismatch(truth, ours, frequencies=grid, max_delta_t=1e-3)
            )
            result[f"{name}_no_optimisation"].append(unoptimised(truth, ours, keep))
        result["glitched_fraction"].append(float(glitched.mean()))
        # the direction of L at the integration's nodes
        l_hat = np.asarray(angles(
            batch_arguments([params], grid)[0], params.total_mass,
            params.reference_frequency_hz, float(grid[0]),
        )[state_index])[0, :, :3]
        result["beta_max"].append(
            float(np.max(np.arctan2(np.hypot(l_hat[:, 0], l_hat[:, 1]), l_hat[:, 2])))
        )
        result["params"].append(params)

    print(
        "  bins masked around TEOBResumS' alpha steps: median "
        f"{np.median(result['glitched_fraction']):.1%}, at most "
        f"{np.max(result['glitched_fraction']):.1%}"
    )
    for name in ("plus", "cross"):
        print(
            f"  h_{name}, all bins: optimised median "
            f"{np.median(result[name + '_all_bins']):.3e}, worst "
            f"{np.max(result[name + '_all_bins']):.3e}"
        )
        print(
            f"  h_{name}: optimised median {np.median(result[name]):.3e}, "
            f"worst {np.max(result[name]):.3e}; not optimised median "
            f"{np.median(result[name + '_no_optimisation']):.3e}, worst "
            f"{np.max(result[name + '_no_optimisation']):.3e}"
        )
    return result


def timing_benchmark(
    model: Model,
    n_points_list=TIMING_N_POINTS,
    n_seeds: int = TIMING_SEEDS,
    n_epochs: int = TIMING_EPOCHS,
    jax_batch: int = TIMING_JAX_BATCH,
) -> dict:
    r"""Median wall-clock time per waveform, numpy vs. JAX ``Model.predict``.

    Compares :meth:`Model.predict` against the JAX port
    (:func:`mlgw_bns.jax_predict.model_to_jax_waveform`): a single
    JIT-compiled call, and a ``jax.vmap``-batched call (``jax_batch``
    waveforms per call, time reported per-waveform), across grid sizes.
    JIT compilation is warmed up once per grid size before timing starts,
    so the reported times are steady-state. Uses
    ``benchmark_evaluation_time.random_parameters`` (added to ``sys.path``
    since it lives alongside this script) for the intrinsic/extrinsic
    draw, so the two scripts sample identically.

    Skipped --- with a warning, returning only the ``"numpy"`` entry ---
    if ``jax`` is not importable; it is an optional extra
    (``pyproject.toml``'s ``[jax]`` group), not a hard dependency.

    Returns
    -------
    dict
        ``{"n_points": [...], "numpy": [median ms, one per n_points],
        "jax": [...], "jax_batch": [...]}``, the last two omitted if JAX
        is unavailable.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from benchmark_evaluation_time import random_parameters

    frequencies_hz = model.dataset.frequencies_hz
    f_min = float(np.min(frequencies_hz))
    f_max = float(np.max(frequencies_hz)) - 1

    result: dict = {"n_points": list(n_points_list), "numpy": []}

    # One untimed call first: numba JIT-compiles on first use and the
    # model has its own lazy fixed-cost caches (mode-phase cache, sklearn
    # config, ...), both one-off costs that would otherwise contaminate
    # the smallest n_points bucket.
    model.predict(np.linspace(f_min, f_max, num=n_points_list[0]), random_parameters(model, 0))

    print("  numpy Model.predict:")
    for n_points in n_points_list:
        freqs = np.linspace(f_min, f_max, num=n_points)
        times_ms = []
        for _ in range(n_epochs):
            for seed in range(n_seeds):
                params = random_parameters(model, seed)
                start = perf_counter()
                model.predict(freqs, params)
                times_ms.append((perf_counter() - start) * 1e3)
        result["numpy"].append(float(np.median(times_ms)))
        print(f"    n_points={n_points:>6}  median {result['numpy'][-1]:.3f} ms")

    try:
        import jax
        import jax.numpy as jnp

        from mlgw_bns.jax_predict import model_to_jax_waveform
    except Exception as exc:  # pragma: no cover - environment dependent
        logging.warning("JAX not available (%s); skipping JAX timing", exc)
        return result

    def pack(params, freqs):
        return (
            jnp.asarray(
                [params.mass_ratio, params.lambda_1, params.lambda_2,
                 params.chi_1, params.chi_2]
            ),
            jnp.asarray(freqs),
            jnp.asarray(params.total_mass),
            jnp.asarray(params.distance_mpc),
            jnp.asarray(params.inclination),
            jnp.asarray(params.coalescence_phase),
        )

    predict_single = jax.jit(model_to_jax_waveform(model))
    predict_batch = jax.jit(
        jax.vmap(
            model_to_jax_waveform(model), in_axes=(0, None, None, None, None, None)
        )
    )
    rng = np.random.default_rng(0)

    result["jax"] = []
    result["jax_batch"] = []

    print("  JAX Model.predict (single call, JIT-compiled):")
    for n_points in n_points_list:
        freqs = np.linspace(f_min, f_max, num=n_points)

        args = pack(random_parameters(model, 0), freqs)
        jax.block_until_ready(predict_single(*args))  # compile, not timed

        times_ms = []
        for _ in range(n_epochs):
            for seed in range(n_seeds):
                args = pack(random_parameters(model, seed), freqs)
                start = perf_counter()
                jax.block_until_ready(predict_single(*args))
                times_ms.append((perf_counter() - start) * 1e3)
        result["jax"].append(float(np.median(times_ms)))
        print(f"    n_points={n_points:>6}  median {result['jax'][-1]:.3f} ms")

    print(f"  JAX Model.predict (jax.vmap batch of {jax_batch}, per-waveform):")
    for n_points in n_points_list:
        freqs = np.linspace(f_min, f_max, num=n_points)

        centre = random_parameters(model, 0)
        base = np.array(
            [centre.mass_ratio, centre.lambda_1, centre.lambda_2,
             centre.chi_1, centre.chi_2]
        )
        batch_params = base * (1.0 + 0.05 * rng.standard_normal((jax_batch, 5)))
        batch_args = (
            jnp.asarray(batch_params),
            jnp.asarray(freqs),
            jnp.asarray(centre.total_mass),
            jnp.asarray(centre.distance_mpc),
            jnp.asarray(centre.inclination),
            jnp.asarray(centre.coalescence_phase),
        )
        jax.block_until_ready(predict_batch(*batch_args))  # compile, not timed

        times_ms = []
        for _ in range(n_epochs):
            start = perf_counter()
            jax.block_until_ready(predict_batch(*batch_args))
            times_ms.append((perf_counter() - start) * 1e3 / jax_batch)
        result["jax_batch"].append(float(np.median(times_ms)))
        print(f"    n_points={n_points:>6}  median {result['jax_batch'][-1]:.4f} ms")

    return result


def precessing_timing_benchmark(
    model: Model,
    n_points_list=TIMING_N_POINTS,
    n_seeds: int = TIMING_SEEDS,
    n_epochs: int = TIMING_EPOCHS,
    jax_batch: int = TIMING_JAX_BATCH,
) -> dict:
    r"""Median wall-clock time per waveform of the precessing model, as
    :func:`timing_benchmark` does for the aligned one: numpy
    :meth:`PrecessingModel.predict_numpy` (which integrates the PN precession
    afresh at every call) and its JAX port
    (:meth:`PrecessingModel.jax_predict`), for one binary per call and for
    ``jax_batch`` of them, over the same band, on :func:`random_precessing_parameters`.

    Returns
    -------
    dict
        ``{"n_points": [...], "precessing_numpy": [...],
        "precessing_jax": [...], "precessing_jax_batch": [...]}``, the last
        two omitted if JAX is unavailable.
    """
    precessing = PrecessingModel(model)
    frequencies_hz = model.dataset.frequencies_hz
    f_min = float(np.min(frequencies_hz))
    f_max = float(np.max(frequencies_hz)) - 1

    result: dict = {"n_points": list(n_points_list), "precessing_numpy": []}

    def median_ms(call, n_waveforms: int = 1, epochs: int = n_epochs) -> float:
        times_ms = []
        for _ in range(epochs):
            for seed in range(n_seeds):
                start = perf_counter()
                call(seed)
                times_ms.append((perf_counter() - start) * 1e3 / n_waveforms)
        return float(np.median(times_ms))

    precessing.predict_numpy(
        np.linspace(f_min, f_max, num=n_points_list[0]),
        random_precessing_parameters(model, 0),
    )
    print("  numpy PrecessingModel.predict_numpy:")
    for n_points in n_points_list:
        freqs = np.linspace(f_min, f_max, num=n_points)
        params = [random_precessing_parameters(model, seed) for seed in range(n_seeds)]
        result["precessing_numpy"].append(
            # tens of seconds a call (scipy integrates the precession from
            # the bottom of the band every time): once per seed
            median_ms(lambda seed: precessing.predict_numpy(freqs, params[seed]), epochs=1)
        )
        print(f"    n_points={n_points:>6}  median {result['precessing_numpy'][-1]:.3f} ms")

    try:
        import jax

        from mlgw_bns.batched_precession import batch_arguments
    except Exception as exc:  # pragma: no cover - environment dependent
        logging.warning("JAX not available (%s); skipping JAX timing", exc)
        return result

    predict = jax.jit(precessing.jax_predict())
    for key, size in (("precessing_jax", 1), ("precessing_jax_batch", jax_batch)):
        result[key] = []
        print(
            "  JAX PrecessingModel.jax_predict "
            + ("(single binary):" if size == 1 else f"(batch of {size}, per-waveform):")
        )
        for n_points in n_points_list:
            freqs = np.linspace(f_min, f_max, num=n_points)
            args = [
                batch_arguments(
                    [random_precessing_parameters(model, size * seed + i) for i in range(size)],
                    freqs,
                )
                for seed in range(n_seeds)
            ]
            jax.block_until_ready(predict(*args[0]))  # compile, not timed
            result[key].append(
                median_ms(lambda seed: jax.block_until_ready(predict(*args[seed])), size)
            )
            print(f"    n_points={n_points:>6}  median {result[key][-1]:.4f} ms")

    return result


def plot_precessing_mismatches(data: dict) -> None:
    r"""KDEs of :func:`precessing_mismatches`, optimised and not, for
    :math:`h_+` and :math:`h_\times`, beside each binary's mismatch against
    its largest precession opening angle."""
    fig, (ax_kde, ax_beta) = plt.subplots(1, 2, figsize=(13, 4.8))

    values = np.concatenate([
        data[key] for key in data if key not in ("params", "beta_max", "glitched_fraction")
    ])
    grid = np.linspace(np.log10(values.min()) - 0.5, np.log10(values.max()) + 0.5, 400)
    for name, color in (("plus", "C0"), ("cross", "C1")):
        label = r"$h_+$" if name == "plus" else r"$h_\times$"
        for suffix, style, what in (
            ("", "-", "optimised"),
            ("_all_bins", ":", "optimised, alpha-step bins kept"),
            ("_no_optimisation", "--", "not optimised"),
        ):
            log_values = np.log10(data[name + suffix])
            ax_kde.plot(
                10**grid, gaussian_kde(log_values)(grid), color=color, linestyle=style,
                label=f"{label}, {what} (median {_sci_latex(np.median(data[name + suffix]))})",
            )
        ax_beta.scatter(data["beta_max"], data[name], s=12, color=color, label=label)
    ax_kde.set_xscale("log")
    ax_kde.set_xlabel("Mismatch")
    ax_kde.set_ylabel(r"Density [per $\log_{10}$ mismatch]")
    ax_kde.legend(fontsize=8)
    ax_beta.set_xlabel(r"largest opening angle $\beta$ over the band [rad]")
    ax_beta.set_ylabel("Mismatch (optimised)")
    ax_beta.set_yscale("log")
    ax_beta.legend()
    for ax in (ax_kde, ax_beta):
        ax.grid(True, which="both", lw=0.3)
    fig.suptitle(
        f"Precessing model vs. TEOBResumS' precessing $h_+$, $h_\\times$, "
        f"{len(data['plus'])} binaries, $f \\geq {PRECESSING_MIN_FREQUENCY_HZ:g}$ Hz"
    )
    fig.tight_layout()

    outfile = f"{OUTPUT_PREFIX}_precessing_mismatches.png"
    fig.savefig(outfile, dpi=150)
    print(f"Saved plot to {outfile}")


def plot_timing_benchmark(data: dict) -> None:
    r"""Log-log plot of :func:`timing_benchmark`'s per-waveform timings.

    ``data`` is that function's return value; the ``"jax"`` / ``"jax_batch"``
    curves are omitted if it did not have JAX available.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    n_points = data["n_points"]
    ax.loglog(n_points, data["numpy"], marker="o", label="numpy Model.predict")
    if "jax" in data:
        ax.loglog(n_points, data["jax"], marker="o", label="JAX (single call)")
    if "jax_batch" in data:
        ax.loglog(
            n_points, data["jax_batch"], marker="o",
            label=f"JAX (jax.vmap batch of {TIMING_JAX_BATCH})",
        )
    for key, color, label in (
        ("precessing_numpy", "C0", "precessing, numpy"),
        ("precessing_jax", "C1", "precessing, JAX (single binary)"),
        ("precessing_jax_batch", "C2", f"precessing, JAX (batch of {TIMING_JAX_BATCH})"),
    ):
        if key in data:
            ax.loglog(n_points, data[key], marker="s", linestyle="--", color=color, label=label)

    ax.set_xlabel("Number of evaluation points")
    ax.set_ylabel("Time per waveform [ms]")
    ax.grid(True, which="both", lw=0.3)
    ax.legend(loc="center right", fontsize=8)
    fig.suptitle("Prediction wall-clock time: aligned spins (solid) and precessing (dashed)")
    fig.tight_layout()

    outfile = f"{OUTPUT_PREFIX}_timing.png"
    fig.savefig(outfile, dpi=150)
    print(f"Saved plot to {outfile}")


def plot_evaluation_time_fit(
    model_name: str,
    n_points_list=TIMING_N_POINTS,
    n_seeds: int = TIMING_SEEDS,
    n_epochs: int = TIMING_EPOCHS,
) -> None:
    r"""Loglog wall-clock timing with ``c1 + c2 * N`` fit lines per approximant.

    Thin wrapper around ``benchmark_evaluation_time.py``'s own
    :func:`~benchmark_evaluation_time.create_and_run_tests` /
    :func:`~benchmark_evaluation_time.make_figure` (added to ``sys.path``
    since it lives alongside this script), comparing the full model, a
    reduced-mode model (:class:`~benchmark_evaluation_time.MlgwBnsReducedModes`,
    only ``benchmark_evaluation_time.REDUCED_MODES``), TEOBResumS-SPA and,
    when ``jax`` is importable, the JAX port (single call + a
    ``jax.vmap``-batched call, see ``TIMING_JAX_BATCH``) -- both of the
    latter also in a reduced-mode flavour.

    Skipped --- with a warning --- if ``model_name`` is not one of
    :data:`mlgw_bns.model.MODELS_AVAILABLE`, since the approximants there
    load a *shipped* pretrained model by name rather than reusing an
    already-loaded :class:`Model` instance.
    """
    from mlgw_bns.model import MODELS_AVAILABLE

    if model_name not in MODELS_AVAILABLE:
        logging.warning(
            "%r not in MODELS_AVAILABLE=%r; skipping the fit-line timing "
            "benchmark (it loads a shipped pretrained model by name)",
            model_name,
            MODELS_AVAILABLE,
        )
        return

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from benchmark_evaluation_time import (
        _HAVE_JAX,
        MlgwBns,
        MlgwBnsJax,
        MlgwBnsJaxBatch,
        MlgwBnsJaxBatchReducedModes,
        MlgwBnsJaxReducedModes,
        MlgwBnsReducedModes,
        TEOBResumSPA,
        create_and_run_tests,
        make_figure,
    )

    approximants = [
        MlgwBns(model_name),
        MlgwBnsReducedModes(model_name),
        TEOBResumSPA(model_name),
    ]
    if _HAVE_JAX:
        approximants.append(MlgwBnsJax(model_name))
        approximants.append(MlgwBnsJaxReducedModes(model_name))
        approximants.append(MlgwBnsJaxBatch(TIMING_JAX_BATCH, model_name))
        approximants.append(MlgwBnsJaxBatchReducedModes(TIMING_JAX_BATCH, model_name))
    else:
        logging.warning("JAX not importable -- skipping it in the fit-line timing benchmark")

    tests = create_and_run_tests(n_seeds, n_epochs, list(n_points_list), approximants)
    make_figure(tests, approximants, list(n_points_list))

    outfile = f"{OUTPUT_PREFIX}_timing_fit.png"
    plt.savefig(outfile, dpi=150)
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
        N_PRECESSING_WAVEFORMS = args.n_mismatches

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

    print(
        "Computing full-waveform mismatches and per-mode power shares "
        f"(frequency grid down to {LOW_FREQUENCY_HZ} Hz, exercising the "
        "post-Newtonian low-frequency extension)..."
    )
    full_mismatches, full_mismatches_no_opt, power_fractions = full_waveform_mismatches(
        model
    )

    report_weighted_mismatches(mismatches_by_mode, power_fractions, full_mismatches)
    np.savez(
        f"{OUTPUT_PREFIX}_mismatches.npz",
        full=np.asarray(full_mismatches),
        full_no_optimisation=np.asarray(full_mismatches_no_opt),
        **{
            f"{name} l{mode.l}_m{mode.m}": np.asarray(values)
            for mode, pair in mismatches_by_mode.items()
            for name, values in zip(("optimised", "regressed"), pair)
        },
    )
    plot_mismatches(
        mismatches_by_mode, full_mismatches, full_mismatches_no_opt, power_fractions
    )

    print("Computing per-mode mismatch vs. power fraction...")
    mismatch_vs_power = mismatch_vs_power_by_mode(model)
    plot_mismatch_vs_power(mismatch_vs_power)

    print("Computing full-waveform mismatch vs. mode subset...")
    plot_mode_subset_boxplots(mode_subset_data(model))

    print("Computing precessing mismatches against TEOBResumS' precessing h_+, h_x...")
    precessing_data = precessing_mismatches(model, N_PRECESSING_WAVEFORMS)
    np.savez(
        f"{OUTPUT_PREFIX}_precessing_mismatches.npz",
        **{key: np.asarray(value) for key, value in precessing_data.items() if key != "params"},
    )
    plot_precessing_mismatches(precessing_data)

    print("Benchmarking prediction wall-clock time (numpy vs. JAX)...")
    timing_data = timing_benchmark(model)
    timing_data.update(precessing_timing_benchmark(model))
    plot_timing_benchmark(timing_data)

    print("Benchmarking prediction wall-clock time with fit lines (full vs. reduced-mode vs. TEOB)...")
    plot_evaluation_time_fit(os.path.basename(args.model.rstrip("/")) or "default_hom")
