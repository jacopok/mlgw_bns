r"""Scatter the (2,1) mode's reconstruction accuracy against how much it
actually contributes to the observable strain.

A per-mode mismatch is a *relative* measure: it is invariant under
rescaling the mode, so it says nothing on its own about whether an error
matters. The (2,1) mode makes that concrete. Its amplitude carries a
mass-asymmetry factor which vanishes at equal mass, so as :math:`q \to 1`
the mode both becomes hardest to reconstruct --- its amplitude passes
through zero inside the band, a feature the surrogate represents poorly
--- and contributes least to the waveform. The two effects are strongly
anti-correlated, which is the point of this figure.

For each parameter tuple it plots

* :math:`x`: the fraction of the total PSD-weighted power carried by the
  (2,1) mode, :math:`(h_{21}|h_{21}) / (h|h)` with
  :math:`h = \sum_{\ell m} h_{\ell m}`,
* :math:`y`: the mismatch between the reconstructed and true (2,1) mode,
* colour: the mass ratio :math:`q`.

Diagonal guides mark constant :math:`\text{mismatch} \times \text{power}`,
a first-order estimate of a mode's contribution to the full-waveform
error: points on the same diagonal matter equally, however different
their mismatches look.

The model is trained on first use and cached under `MODEL_FILENAME`;
delete those files to retrain.

Run with: python visualization/plot_mode_accuracy_vs_power.py
"""

import logging
from pathlib import Path

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.mode_model import ParametersWithExtrinsic
from mlgw_bns.model_validation import ValidateModel
from mlgw_bns.model import Model

from validate_model import SharedTimeshiftValidateModel

logging.basicConfig(level=logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parent.parent

MODES = [Mode(2, 2), Mode(2, 1), Mode(3, 3), Mode(4, 4)]

#: The mode this figure is about; the others are here because the power
#: fraction needs the summed waveform in its denominator.
TARGET_MODE = Mode(2, 1)

MODEL_FILENAME = REPO_ROOT / "accuracy_vs_power_model"
INITIAL_FREQUENCY_HZ = 20.0

#: Training sizes: downsampling, PCA, network. Modest --- the figure is
#: about the *correlation* between accuracy and contribution, which is a
#: property of the waveforms, not about the absolute accuracy of a
#: production-sized model.
TRAINING_SIZES = (2 ** 6, 2 ** 8, 2 ** 9)

N_WAVEFORMS = 1000
SEED = 17

# Extrinsic parameters. The inclination matters: the (2,1) mode is
# weighted by ${}_{-2}Y_{21}$, which vanishes face-on and peaks near
# edge-on, so a face-on choice would suppress it for trivial reasons.
DISTANCE_MPC = 100.0
INCLINATION = 1.0
TOTAL_MASS = 2.8

OUTFILE = REPO_ROOT / "mode_accuracy_vs_power.png"

#: Collected arrays, so that re-plotting does not regenerate waveforms.
#: Delete to force recollection.
DATA_CACHE = REPO_ROOT / "mode_accuracy_vs_power.npz"


def load_or_train() -> Model:
    """Load the cached model, training and saving it if it is not there."""
    model = Model(
        modes=MODES,
        filename=str(MODEL_FILENAME),
        initial_frequency_hz=INITIAL_FREQUENCY_HZ,
    )

    if Path(f"{MODEL_FILENAME}_l2_m2_arrays.h5").exists():
        model.load()
        if model.nn_available:
            print(f"Loaded cached model from {MODEL_FILENAME}*")
            return model

    print(f"Training a model (sizes {TRAINING_SIZES})...")
    model.generate(*TRAINING_SIZES)
    model.set_hyper_and_train_nn()
    model.save(include_training_data=False)
    return model


def collect(model: Model):
    """Return per-waveform accuracy, power fraction, mass ratio, sign flag.

    The two axes are measured by different routes, deliberately.

    The *accuracy* is the per-mode mismatch from :class:`ValidateModel`,
    the same quantity ``validate_model.py`` reports: the mode is
    compared in its own amplitude/phase representation, with the learned
    time-shift correction applied and the phase anchored at the start of
    the band.

    The *power fraction* has to come from the summed waveform, so it uses
    :meth:`Model.predict_modes_dict` / ``get_teob_modes_dict``.

    Scoring *any* mode through the mode dictionaries with
    :meth:`ValidateModel.mismatch` would be wrong, for a reason which has
    nothing to do with which mode it is. The waveforms
    :meth:`Model.predict_modes_dict` returns sit about 15 ms away in
    time from the ones ``get_teob_modes_dict`` returns --- a common
    origin offset, the same for every mode to within a few percent --- and
    :meth:`ValidateModel.mismatch` searches time shifts only within
    ``max_delta_t = 0.007`` s by default, so it cannot find the
    alignment and reports 0.67 to 0.88 for all of them, the (2,2)
    included. :meth:`ValidateModel.full_waveform_mismatch` uses
    ``max_delta_t = 0.07`` s, absorbs the offset, and reports ~1e-5;
    that is why the summed waveform scores well.

    Note this is *not* a per-mode time-shift disagreement: each mode's own
    time-shift target agrees with the (2,2)'s to about 1% (2e-4 s), and
    the shared predictor reproduces the (2,1) target to 3e-4 s, forty
    times inside the default window.

    Both routes are driven from the same parameter array, so the two axes
    line up waveform by waveform.
    """
    target_key = (TARGET_MODE.l, TARGET_MODE.m)

    # --- accuracy, through the per-mode validator -------------------------
    mode_validator = SharedTimeshiftValidateModel(
        model.mode_models[TARGET_MODE], model.time_shifts_predictor
    )
    parameter_set = mode_validator.param_set(N_WAVEFORMS, SEED)

    print("Generating true waveforms for the target mode...")
    true_waveforms, parameter_set = mode_validator.true_waveforms(parameter_set)
    true_waveforms.phases -= true_waveforms.phases[:, 0].reshape(-1, 1)
    predicted_waveforms = mode_validator.predicted_waveforms(parameter_set)
    mode_validator._apply_predicted_time_shifts(predicted_waveforms, parameter_set)

    cartesian_true, cartesian_predicted = mode_validator.waveforms(
        true_waveforms, predicted_waveforms
    )
    mismatches_all = np.array([
        mode_validator.mismatch(a, b)
        for a, b in tqdm(list(zip(cartesian_true, cartesian_predicted)),
                         desc="mode mismatches")
    ])
    # A sign change in the mode's own amplitude, from the same waveforms.
    sign_changes_all = np.array([
        bool(np.any(amp < 0)) for amp in true_waveforms.amplitudes
    ])

    # --- contribution, through the summed waveform ------------------------
    reference_model = model.mode_models[Mode(2, 2)]
    validator = ValidateModel(reference_model)
    frequencies = validator.frequencies
    psd = validator.psd_values

    def power(waveform: np.ndarray, mask: np.ndarray) -> float:
        return float(
            np.abs(
                np.trapezoid(
                    np.conj(waveform[mask]) * waveform[mask] / psd[mask],
                    x=frequencies[mask],
                )
            )
        )

    mismatches, fractions, mass_ratios, sign_changes = [], [], [], []

    for index in tqdm(range(len(parameter_set.parameter_array)), desc="power shares"):
        row = parameter_set.parameter_array[index]
        mass_ratio, lambda_1, lambda_2, chi_1, chi_2 = row
        params = ParametersWithExtrinsic(
            mass_ratio=mass_ratio,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            chi_1=chi_1,
            chi_2=chi_2,
            distance_mpc=DISTANCE_MPC,
            inclination=INCLINATION,
            total_mass=TOTAL_MASS,
        )

        try:
            true = model.get_teob_modes_dict(frequencies, params)
        except Exception as e:  # noqa: BLE001
            logging.warning("Skipping a parameter tuple: %s", e)
            continue

        # The EOB modes are zero-padded below their starting frequency.
        mask = np.ones(len(frequencies), dtype=bool)
        for mode_array in true.values():
            mask &= np.abs(mode_array) > 0
        if mask.sum() < 2:
            continue

        total_power = power(sum(true.values()), mask)
        if total_power <= 0:
            continue

        mismatches.append(mismatches_all[index])
        fractions.append(power(true[target_key], mask) / total_power)
        mass_ratios.append(mass_ratio)
        sign_changes.append(sign_changes_all[index])

    return (
        np.array(mismatches),
        np.array(fractions),
        np.array(mass_ratios),
        np.array(sign_changes),
    )


def plot(mismatches, fractions, mass_ratios, sign_changes) -> None:
    """Scatter accuracy against contribution, coloured by mass ratio."""
    fig, ax = plt.subplots(figsize=(9.5, 7))

    cmap = matplotlib.colormaps["viridis"]
    scatter = ax.scatter(
        fractions,
        mismatches,
        c=mass_ratios,
        cmap=cmap,
        s=16,
        alpha=0.75,
        linewidths=0,
    )

    # Lines of constant mismatch * power: equal first-order contribution
    # to the full-waveform error. Clipped to the axes, which are set by
    # the data --- letting the guides drive the limits leaves most of the
    # frame empty.
    x_limits = np.array([fractions.min() * 0.5, fractions.max() * 2])
    y_limits = (mismatches.min() * 0.4, mismatches.max() * 3)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)

    for product in np.geomspace(1e-10, 1e-4, 7):
        ax.plot(
            x_limits,
            product / x_limits,
            color="grey",
            linewidth=0.7,
            linestyle=":",
            zorder=0,
        )
        # label where the guide leaves the frame, so it stays visible
        y_at_right = product / x_limits[1]
        if y_limits[0] < y_at_right < y_limits[1]:
            ax.annotate(f"{product:.0e}", xy=(x_limits[1], y_at_right),
                        fontsize=7, color="grey", ha="right", va="bottom")
        else:
            x_at_top = product / y_limits[1]
            if x_limits[0] < x_at_top < x_limits[1]:
                ax.annotate(f"{product:.0e}", xy=(x_at_top, y_limits[1]),
                            fontsize=7, color="grey", ha="left", va="top")

    if sign_changes.any():
        ax.scatter(
            fractions[sign_changes],
            mismatches[sign_changes],
            facecolors="none",
            edgecolors="crimson",
            s=60,
            linewidths=0.9,
            label=f"amplitude changes sign ({sign_changes.sum()})",
        )
        ax.legend(loc="lower left", fontsize="small")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(
        r"share of PSD-weighted power carried by the mode, "
        r"$(h_{21}|h_{21}) \, / \, (h|h)$"
    )
    ax.set_ylabel(r"$(2,1)$ mode mismatch")
    ax.grid(True, which="major", alpha=0.4)

    correlation = np.corrcoef(np.log10(fractions), np.log10(mismatches))[0, 1]
    fig.suptitle(
        rf"$(\ell, m) = (2,1)$ accuracy against contribution, "
        rf"{len(mismatches)} waveforms, $\iota = {INCLINATION}$"
    )
    ax.set_title(
        f"Pearson r of the logs = {correlation:+.2f}; dotted lines are constant "
        "mismatch x power",
        fontsize="small",
    )

    fig.colorbar(scatter, ax=ax, label="Mass ratio $q$", pad=0.02)
    fig.tight_layout()
    fig.savefig(OUTFILE, dpi=150)
    print(f"Saved plot to {OUTFILE}")


if __name__ == "__main__":
    if DATA_CACHE.exists():
        print(f"Loading collected data from {DATA_CACHE}")
        cached = np.load(DATA_CACHE)
        mismatches = cached["mismatches"]
        fractions = cached["fractions"]
        mass_ratios = cached["mass_ratios"]
        sign_changes = cached["sign_changes"].astype(bool)
    else:
        model = load_or_train()
        mismatches, fractions, mass_ratios, sign_changes = collect(model)
        np.savez(
            DATA_CACHE,
            mismatches=mismatches,
            fractions=fractions,
            mass_ratios=mass_ratios,
            sign_changes=sign_changes,
        )
        print(f"Saved collected data to {DATA_CACHE}")

    print(f"\n{len(mismatches)} waveforms")
    print(f"  mismatch:     median {np.median(mismatches):.3e}, "
          f"worst {np.max(mismatches):.3e}")
    print(f"  power share:  median {np.median(fractions):.3e}, "
          f"max  {np.max(fractions):.3e}")
    print(f"  product:      median {np.median(mismatches * fractions):.3e}, "
          f"worst {np.max(mismatches * fractions):.3e}")
    print(f"  sign changes: {sign_changes.sum()}")
    print(f"  Pearson r of the logs: "
          f"{np.corrcoef(np.log10(fractions), np.log10(mismatches))[0, 1]:+.3f}")

    plot(mismatches, fractions, mass_ratios, sign_changes)
