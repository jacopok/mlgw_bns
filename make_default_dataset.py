"""Train the model shipped with the package, and install it into `mlgw_bns/data/`.

Training writes `default_hom_*` in the current directory, with the training
residuals included so that the model can be retrained or validated without
regenerating them; `install()` then re-saves it into the package without
those residuals, which takes it from ~100 MB to ~2 MB.

Run with: python make_default_dataset.py
"""

import logging

from mlgw_bns.model import DEFAULT_MODES, MODELS_AVAILABLE, PRETRAINED_MODEL_FOLDER, Model

logging.basicConfig(level=logging.INFO)

#: Where training writes its output, relative to the current directory.
TRAINING_BASENAME = MODELS_AVAILABLE[0]

#: Where the packaged copy lives, relative to the repository root.
PACKAGED_BASENAME = f"mlgw_bns/{PRETRAINED_MODEL_FOLDER}{MODELS_AVAILABLE[0]}"


def train() -> None:
    model = Model(
        modes=list(DEFAULT_MODES),
        filename=TRAINING_BASENAME,
        initial_frequency_hz=5.0,
    )
    model.generate(2**8, 2**12, 2**13)
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
    train()
    install()
