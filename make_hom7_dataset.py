"""Train a 7-mode model with a wider tidal-deformability range.

Adds (3,1), (3,2), (4,3) to the shipped default's (2,2)/(2,1)/(3,3)/(4,4),
and widens `lambda1_range`/`lambda2_range` from the default (5, 5000) to
(5, 12000). Everything else (kernel-ridge regressor, reference amplitude,
odd-`m` power weighting, downsampling/training sizes) matches
`make_default_dataset.py`; power weighting activates generically on
`mode.m % 2 == 1` (see `ModeModel._training_power_weights`), so it applies
to (3,1) and (4,3) automatically, no per-mode listing needed.

Writes `hom7_*` in the current directory.

Run with: python make_hom7_dataset.py [--n-jobs N]
"""

import argparse
import logging

from mlgw_bns.data_management import ParameterRanges
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model import Model
from mlgw_bns.neural_network import KernelRidgeNetwork

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)

#: Where training writes its output, relative to the current directory.
TRAINING_BASENAME = "hom7"

MODES = [
    Mode(2, 2),
    Mode(2, 1),
    Mode(3, 1),
    Mode(3, 2),
    Mode(3, 3),
    Mode(4, 3),
    Mode(4, 4),
]

LAMBDA_MAX = 12000.0


def train(n_jobs: int = 1) -> None:
    model = Model(
        modes=list(MODES),
        filename=TRAINING_BASENAME,
        initial_frequency_hz=5.0,
        nn_kind=KernelRidgeNetwork,
        reference_amplitude=True,
        parameter_ranges=ParameterRanges(
            lambda1_range=(5.0, LAMBDA_MAX),
            lambda2_range=(5.0, LAMBDA_MAX),
        ),
    )
    # Sizes match `make_default_dataset.py`; see its comments for the
    # memory/accuracy tradeoffs behind each number.
    model.generate(64, 2**15, 24576, reference_dataset_size=16384, n_jobs=n_jobs)
    model.set_hyper_and_train_nn()
    model.save(include_training_data=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="worker processes for EOB sweeps; sequential (1) unless given",
    )
    args = parser.parse_args()

    train(n_jobs=args.n_jobs)
