r"""Train matched models for the legacy and improved pipelines.

The two changes worth comparing are the regressor --- a kernel ridge
regression rather than the multi-layer perceptron --- and the definition
of the amplitude residual, which for the (2,1) and (3,3) modes can divide
by a fixed reference amplitude rather than each waveform's own
Post-Newtonian one.

Waveform generation dominates the cost, so the first two models share it:
the residuals are generated once, the network is trained and saved, and
then the same per-mode models are retrained with the kernel and saved
again. That makes the legacy/kernel comparison exact --- identical
training waveforms, identical PCA basis, only the regressor differs. The
reference-amplitude model changes what is being decomposed and so needs
its own generation pass.

Writes ``val_legacy*``, ``val_kernel*`` and ``val_kernel_ref*`` into the
current directory, ready for ``validate_model.py``.

Run with: python visualization/train_comparison_models.py [--n-train 2048]
"""

from __future__ import annotations

import argparse
import logging
import time

from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model import Model
from mlgw_bns.neural_network import KernelRidgeNetwork, SklearnNetwork

MODES = [Mode(2, 2), Mode(2, 1), Mode(3, 3), Mode(4, 4)]

#: Matches the packaged model, so that the comparison is against the
#: pipeline as it actually ships.
INITIAL_FREQUENCY_HZ = 5.0


def train(name: str, n_train: int, nn_kind, reference_amplitude: bool) -> Model:
    model = Model(
        modes=list(MODES),
        filename=name,
        initial_frequency_hz=INITIAL_FREQUENCY_HZ,
        nn_kind=nn_kind,
        reference_amplitude=reference_amplitude,
    )
    start = time.perf_counter()
    model.generate(2**8, n_train, n_train)
    logging.info("[%s] generation took %.0f s", name, time.perf_counter() - start)

    start = time.perf_counter()
    model.set_hyper_and_train_nn()
    logging.info("[%s] training took %.0f s", name, time.perf_counter() - start)

    model.save(include_training_data=True)
    return model


def retrain_in_place(model: Model, name: str, nn_kind) -> None:
    """Re-fit an already-generated model with a different regressor.

    Everything upstream of the regressor --- downsampling nodes, PCA
    basis, training residuals --- is left exactly as it was, so the only
    difference between the saved result and the original is the map from
    parameters to component coefficients.
    """
    # The setter propagates the new base name to every mode model that has
    # already been built, which after `generate` is all of them.
    model.base_filename = name
    for mode_model in model.mode_models.values():
        mode_model.nn_kind = nn_kind

    start = time.perf_counter()
    model.set_hyper_and_train_nn()
    logging.info("[%s] retraining took %.0f s", name, time.perf_counter() - start)

    model.save(include_training_data=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=2048)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    legacy = train("val_legacy", args.n_train, SklearnNetwork, False)
    retrain_in_place(legacy, "val_kernel", KernelRidgeNetwork)

    train("val_kernel_ref", args.n_train, KernelRidgeNetwork, True)

    logging.info("Done: val_legacy, val_kernel, val_kernel_ref")


if __name__ == "__main__":
    main()
