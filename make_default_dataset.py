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

Run with: python make_default_dataset.py [--legacy]
"""

import argparse
import logging

from mlgw_bns.model import DEFAULT_MODES, MODELS_AVAILABLE, PRETRAINED_MODEL_FOLDER, Model
from mlgw_bns.neural_network import KernelRidgeNetwork, SklearnNetwork

logging.basicConfig(level=logging.INFO)

#: Where training writes its output, relative to the current directory.
TRAINING_BASENAME = MODELS_AVAILABLE[0]

#: Where the packaged copy lives, relative to the repository root.
PACKAGED_BASENAME = f"mlgw_bns/{PRETRAINED_MODEL_FOLDER}{MODELS_AVAILABLE[0]}"


def train(legacy: bool = False) -> None:
    model = Model(
        modes=list(DEFAULT_MODES),
        filename=TRAINING_BASENAME,
        initial_frequency_hz=5.0,
        nn_kind=SklearnNetwork if legacy else KernelRidgeNetwork,
        reference_amplitude=not legacy,
    )
    model.generate(2**9, 2**13, 2**13)
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
    args = parser.parse_args()

    train(legacy=args.legacy)
    install()
