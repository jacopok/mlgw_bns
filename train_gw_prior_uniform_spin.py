"""Train a HOM model with the same settings as ``default_hom`` but with the
GW-analysis parameter prior: mass ratio uniform in 1/q, aligned spins uniform.

Writes ``gw_prior_uniform_spin_hom_*`` in the current directory, including the
training residuals (``include_training_data=True``) so the run can be
revalidated or retrained without regenerating the EOB data.

Dataset sizes default to the ones that actually produced the shipped
``default_hom`` model (see ``retrain_default_hom.log``: 512 / 8192 / 8192,
peak ~8.3 GB RSS), not the larger values written in ``make_default_dataset.py``
which OOM this machine (23 GB, shared with the editor/Zoom).

Run with: python train_gw_prior_uniform_spin.py [--downsampling N] [--pca N] [--nn N] [--reference N]
"""

import argparse
import logging

from mlgw_bns.model import DEFAULT_MODES, Model
from mlgw_bns.neural_network import KernelRidgeNetwork
from mlgw_bns.dataset_generation import GWPriorUniformSpinParameterGenerator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)

TRAINING_BASENAME = "gw_prior_uniform_spin_hom"


def train(downsampling: int, pca: int, nn: int, reference: int) -> None:
    model = Model(
        modes=list(DEFAULT_MODES),
        filename=TRAINING_BASENAME,
        initial_frequency_hz=5.0,
        nn_kind=KernelRidgeNetwork,
        reference_amplitude=True,
        parameter_generator_class=GWPriorUniformSpinParameterGenerator,
    )
    model.generate(downsampling, pca, nn, reference_dataset_size=reference)
    model.set_hyper_and_train_nn()
    model.save(include_training_data=True)
    logging.info("Done; wrote %s_*", TRAINING_BASENAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downsampling", type=int, default=2**9)
    parser.add_argument("--pca", type=int, default=2**13)
    parser.add_argument("--nn", type=int, default=2**13)
    parser.add_argument("--reference", type=int, default=2**13)
    args = parser.parse_args()
    train(args.downsampling, args.pca, args.nn, args.reference)
