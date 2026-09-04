"""Moderate-size full HOM retrain with the Nystroem phase regressor.

Same configuration as ``default_hom`` (5 Hz, ``KernelRidgeNetwork`` per-mode
PCA-coefficient regressor, ``reference_amplitude``), but:

* :class:`~mlgw_bns.neural_network.ModePhasesNN` now uses
  ``make_nystroem_ridge_pipeline`` (Nystroem landmarks + GCV ridge) instead
  of random Fourier features -- the ``compare_phase_regressors.py`` winner;
* the ``psi_lm`` ``np.angle`` branch-cut fix is in
  (:mod:`mlgw_bns.pn_modes`), so the per-mode PN phase -- and hence the
  regression target -- is continuous in parameter space;
* dataset sizes are moderate for a quick turnaround, not the shipped
  1024 / 32768 / 16384 / 32768.

Writes ``nystroem_hom_*`` in the current directory (with training data).
Validate with::

    python visualization/validate_model.py --model nystroem_hom --n-mismatches 100
"""

import argparse
import logging

from mlgw_bns.model import DEFAULT_MODES, Model
from mlgw_bns.neural_network import KernelRidgeNetwork

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BASENAME = "nystroem_hom"


def train(downsampling: int, pca: int, nn: int, reference: int, batch: int) -> None:
    model = Model(
        modes=list(DEFAULT_MODES),
        filename=BASENAME,
        initial_frequency_hz=5.0,
        nn_kind=KernelRidgeNetwork,
        reference_amplitude=True,
    )
    model.generate(
        downsampling, pca, nn,
        reference_dataset_size=reference,
        reference_batch_size=batch,
    )
    model.set_hyper_and_train_nn()
    model.save(include_training_data=True)
    logging.info("Done; wrote %s_*", BASENAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downsampling", type=int, default=512)
    parser.add_argument("--pca", type=int, default=4096)
    parser.add_argument("--nn", type=int, default=4096)
    parser.add_argument("--reference", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=2000)
    args = parser.parse_args()
    train(args.downsampling, args.pca, args.nn, args.reference, args.batch)
