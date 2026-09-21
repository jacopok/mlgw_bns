"""Train the model shipped with the package, and install it into `mlgw_bns/data/`.

Training writes `default_hom_*` in the current directory, with the training
residuals included so that the model can be retrained or validated without
regenerating them; `install()` then re-saves it into the package without
those residuals, which takes it from ~100 MB to ~2 MB.

The regressor is a kernel ridge regression rather than the multi-layer
perceptron the packaged model was originally trained with: under a fixed
training budget the surrogate is limited by the map from parameters to
principal-component coefficients, and on the (2,2) mode the kernel is worth
about three orders of magnitude there. `reference_amplitude` changes what the
(2,1) and (3,3) modes decompose, dividing by a fixed reference amplitude
rather than each waveform's own Post-Newtonian one, which is worth a further
factor of eighteen on the (3,3). Pass `--legacy` to reproduce the original
pipeline instead.

Run with: python make_default_dataset.py [--legacy] [--n-jobs N]

Every EOB sweep in `Model.generate` runs sequentially (`n_jobs=1`) unless
told otherwise -- parallelism is opt-in, not a hidden default -- so pass
`--n-jobs` to use multiple worker processes.
"""

import argparse
import logging

from mlgw_bns.model import DEFAULT_MODES, MODELS_AVAILABLE, PRETRAINED_MODEL_FOLDER, Model
from mlgw_bns.neural_network import KernelRidgeNetwork, SklearnNetwork

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)

#: Where training writes its output, relative to the current directory.
TRAINING_BASENAME = MODELS_AVAILABLE[0]

#: Where the packaged copy lives, relative to the repository root.
PACKAGED_BASENAME = f"mlgw_bns/{PRETRAINED_MODEL_FOLDER}{MODELS_AVAILABLE[0]}"


def train(legacy: bool = False, n_jobs: int = 1) -> None:
    model = Model(
        modes=list(DEFAULT_MODES),
        filename=TRAINING_BASENAME,
        initial_frequency_hz=5.0,
        nn_kind=SklearnNetwork if legacy else KernelRidgeNetwork,
        reference_amplitude=not legacy,
    )
    # Downsampling-index training saturates by ~16-32 waveforms (held-out
    # reconstruction error and the phase node count are both flat well before
    # 64; see downsampling_vs_ntrain.py), while generating that many on the
    # full ~5x10^5-point grid is the single largest memory spike in
    # `generate` (~17 GB at 1024, ~0.5 GB at 64). 64 keeps the margin.
    #
    # The per-mode NN dataset size is capped by KernelRidgeNetwork's exact
    # RBF solve, whose peak RSS is ~quadratic in its size (measured with
    # probe_kernel_ridge_memory.py on this 23 GB / ~16 GB-available box:
    # 8192 -> 1.3 GB, 16384 -> 4.5 GB, 24576 -> 9.8 GB, 32768 timed out,
    # almost certainly OOM); 24576 leaves a comfortable margin. PCA's
    # economy SVD is linear in its dataset size, so it can stay large. The
    # reference (time-shift + mode-phase) pre-pass now also fits an exact
    # KernelRidge (ModePhasesNN switched off Nystroem, see
    # compare_mode_phase_regressor.py) and so faces the same quadratic
    # limit -- kept at 16384, both safe and past the accuracy plateau
    # (reference-phase-regressor-floor memory note).
    model.generate(64, 2**15, 24576, reference_dataset_size=16384, n_jobs=n_jobs)
    model.set_hyper_and_train_nn()
    model.save(include_training_data=True)


def install() -> None:
    """Copy the trained model into the package, without the training data."""
    model = Model(modes=list(DEFAULT_MODES), filename=TRAINING_BASENAME)
    model.load()

    if not model.nn_available:
        raise RuntimeError(f"No trained networks found at {TRAINING_BASENAME}; run train() first.")

    model.base_filename = PACKAGED_BASENAME
    model.save(include_training_data=False)
    logging.info("Installed the default model into %s", PACKAGED_BASENAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="train with the original network and amplitude parametrization",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="worker processes for EOB sweeps; sequential (1) unless given",
    )
    args = parser.parse_args()

    train(legacy=args.legacy, n_jobs=args.n_jobs)
    install()
