"""Validate the shared time-shift predictor of the ``default_hom`` model.

A :class:`ModesModel` trains one regressor --- saved as
``default_hom_timeshifts.pkl`` --- mapping the intrinsic parameters
:math:`[q, \\Lambda_1, \\Lambda_2, \\chi_1, \\chi_2]` to the time shift
:math:`\\Delta t(\\theta)` in seconds, and reuses it for every mode. That
:math:`\\Delta t` is what :func:`mlgw_bns.model.remove_linear_trend` takes
out of the phase residuals before the PCA and the network ever see them,
and what the prediction adds back afterwards, so an error here is a pure
linear-in-frequency phase error in every predicted waveform.

The regression target is defined by
:meth:`mlgw_bns.data_management.Residuals.phase_timeshifts`: the slope of
the chord through the phase residual between its first sample point and
the one 20% of the way along the downsampling grid. This script
reproduces that target exactly --- same mode, same downsampling nodes,
same frequency grid --- on freshly drawn parameters, and histograms

1. the absolute error :math:`\\Delta t_{\\rm pred} - \\Delta t_{\\rm true}`,
   in seconds;
2. the relative error, which is what tells you whether the regressor is
   actually resolving the parameter dependence rather than predicting
   something near the mean;
3. the dephasing that error induces across the model's band,
   :math:`2 \\pi\\, \\delta(\\Delta t)\\, (f_{\\max} - f_{\\min})`, which is
   the quantity a mismatch calculation would have to marginalise away.

The parameters are drawn with a seed different from the one
:meth:`Dataset.generate_residuals` defaults to (2), so these are held-out
draws rather than a re-scoring of the training set.

Run with: python visualization/validate_timeshifts_predictor.py
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.modes_model import ModesModel
from mlgw_bns.neural_network import load_timeshifts_predictor_from_file

logging.basicConfig(level=logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Base name of the model whose predictor is under test; the predictor
#: itself lives at ``{MODEL_FILENAME}_timeshifts.pkl``.
MODEL_FILENAME = REPO_ROOT / "default_hom"

#: The shared predictor is trained from the (2,2) mode's residuals, so the
#: truth must be computed on that mode's downsampling grid.
REFERENCE_MODE = Mode(2, 2)

N_WAVEFORMS = 200

#: Not 2, which is what :meth:`Dataset.generate_residuals` falls back to
#: when the dataset has no parameter generator set --- i.e. what the
#: training set was drawn with. These are held-out parameters.
SEED = 4242

OUTFILE = REPO_ROOT / "timeshifts_validation.png"


def load_pieces():
    """Return the (2,2) model and the predictor being validated.

    The predictor is loaded from its checkpoint explicitly, rather than
    taken from the :class:`ModesModel`, so that it is unambiguous which
    file this script is scoring.
    """
    modes_model = ModesModel(modes=[REFERENCE_MODE], filename=str(MODEL_FILENAME))
    modes_model.load()

    model = modes_model.models[REFERENCE_MODE]
    if model.downsampling_indices is None:
        raise RuntimeError(
            f"{MODEL_FILENAME} has no downsampling indices; "
            "run make_default_dataset.py first."
        )

    predictor_file = Path(f"{MODEL_FILENAME}_timeshifts.pkl")
    if not predictor_file.exists():
        raise FileNotFoundError(
            f"No time-shift predictor at {predictor_file}; "
            "run make_default_dataset.py first."
        )

    return model, load_timeshifts_predictor_from_file(str(predictor_file))


def true_and_predicted_timeshifts(model, predictor):
    """Compute the regression target and the prediction, for fresh draws.

    Returns
    -------
    true_timeshifts : np.ndarray
        Target values, in seconds, exactly as
        :meth:`Model.generate` computes them for training.
    predicted_timeshifts : np.ndarray
        The predictor's output for the same parameters.
    band_hz : tuple[float, float]
        First and last phase sample point, in Hz.
    """
    dataset = model.dataset

    # `generate_residuals` reaches for `dataset.parameter_generator` when it
    # is set, and only falls back to seed 2 otherwise; setting it here is
    # what makes these draws held-out and reproducible.
    dataset.parameter_generator = dataset.make_parameter_generator(seed=SEED)

    frequencies, parameters, residuals = dataset.generate_residuals(
        N_WAVEFORMS,
        model.downsampling_indices,
        flatten_phase=False,
    )
    frequencies_hz = dataset.natural_units_to_hz(frequencies)

    true_timeshifts = residuals.phase_timeshifts(frequencies=frequencies_hz)
    predicted_timeshifts = predictor.predict(parameters.parameter_array)

    phase_frequencies_hz = dataset.frequencies_hz[
        model.downsampling_indices.phase_indices
    ]

    return (
        np.asarray(true_timeshifts),
        np.asarray(predicted_timeshifts),
        (phase_frequencies_hz[0], phase_frequencies_hz[-1]),
    )


def report(true_timeshifts, predicted_timeshifts, band_hz):
    """Print summary statistics; return the three error arrays to plot."""
    error = predicted_timeshifts - true_timeshifts
    relative_error = error / np.abs(true_timeshifts)
    dephasing = 2 * np.pi * error * (band_hz[1] - band_hz[0])

    print(f"{len(error)} held-out waveforms, seed {SEED}")
    print(
        f"  target spread:  mean {np.mean(true_timeshifts):+.4e} s, "
        f"std {np.std(true_timeshifts):.4e} s"
    )
    print(
        f"  absolute error: median |.| {np.median(abs(error)):.4e} s, "
        f"worst {np.max(abs(error)):.4e} s, bias {np.mean(error):+.4e} s"
    )
    print(
        f"  relative error: median |.| {np.median(abs(relative_error)):.4%}, "
        f"worst {np.max(abs(relative_error)):.4%}"
    )
    print(
        f"  dephasing over {band_hz[0]:.2f}-{band_hz[1]:.1f} Hz: "
        f"median |.| {np.median(abs(dephasing)):.3f} rad, "
        f"worst {np.max(abs(dephasing)):.3f} rad"
    )

    # How much of the target's variance the regressor actually explains; a
    # predictor stuck near the mean would score ~0 here even if its absolute
    # error looked small.
    residual_variance = np.sum(error**2)
    total_variance = np.sum((true_timeshifts - np.mean(true_timeshifts)) ** 2)
    print(f"  R^2: {1 - residual_variance / total_variance:.4f}")

    return error, relative_error, dephasing


def plot(error, relative_error, dephasing, band_hz):
    """Histogram the three error measures.

    The magnitudes are histogrammed on log-spaced bins, as
    ``validate_modes_model.py`` does for mismatches: the errors span
    several decades --- a tight core with a handful of outliers an order
    of magnitude or two out --- and linear bins collapse the core into a
    single column. The sign of the error carries no information the
    printed bias does not already give.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    panels = [
        (
            error,
            r"$|\Delta t_{\rm pred} - \Delta t_{\rm true}|$ [s]",
            "absolute error",
        ),
        (
            relative_error,
            r"$|\Delta t_{\rm pred} - \Delta t_{\rm true}| \, / \, |\Delta t_{\rm true}|$",
            "relative error",
        ),
        (
            dephasing,
            r"$2 \pi \, |\delta(\Delta t)| \, (f_{\max} - f_{\min})$ [rad]",
            f"dephasing over {band_hz[0]:.1f}-{band_hz[1]:.0f} Hz",
        ),
    ]

    for ax, (values, xlabel, title) in zip(axes, panels):
        magnitudes = np.abs(values)
        positive = magnitudes[magnitudes > 0]
        bins = np.geomspace(positive.min(), positive.max(), 30)

        ax.hist(magnitudes, bins=bins, histtype="step", linewidth=1.8, color="tab:blue")
        ax.axvline(
            np.median(magnitudes),
            color="tab:red",
            linewidth=1.2,
            label=f"median {np.median(magnitudes):.3g}",
        )
        ax.axvline(
            np.max(magnitudes),
            color="tab:orange",
            linewidth=1.2,
            linestyle="--",
            label=f"worst {np.max(magnitudes):.3g}",
        )
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True)
        ax.legend()

    axes[0].set_ylabel("Count")

    fig.suptitle(
        f"{MODEL_FILENAME.name}_timeshifts.pkl prediction error, "
        f"{len(error)} held-out waveforms"
    )
    fig.tight_layout()
    fig.savefig(OUTFILE, dpi=150)
    print(f"Saved plot to {OUTFILE}")


if __name__ == "__main__":
    model, predictor = load_pieces()
    true_timeshifts, predicted_timeshifts, band_hz = true_and_predicted_timeshifts(
        model, predictor
    )
    error, relative_error, dephasing = report(
        true_timeshifts, predicted_timeshifts, band_hz
    )
    plot(error, relative_error, dephasing, band_hz)
