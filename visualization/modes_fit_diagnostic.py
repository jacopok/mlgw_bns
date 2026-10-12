"""The "natural" loss of saved perceptrons of a mode on the part of their
training set they were fitted to and on the part held out (the split of
JaxMLP.fit), to tell overfitting from underfitting.

    python visualization/modes_fit_diagnostic.py DATASET --pca-size 16384 --mode l2_m2 \
        --n 16384 --model runs/a/model [runs/b/model ...]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from modes_scale import parse_mode  # noqa: E402

from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.modes_dataset import ShardedModesDataset, mode_name
from mlgw_bns.neural_network import JaxMLPNetwork

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
parser.add_argument("--pca-size", type=int, required=True)
parser.add_argument("--mode", type=parse_mode, required=True)
parser.add_argument("--n", type=int, required=True, help="the training set of the models")
parser.add_argument("--model", nargs="+", required=True, help="base filenames")
args = parser.parse_args()

dataset = ShardedModesDataset(args.dataset)
mode = Mode(*args.mode)
model = dataset.model()
mode_model = model.mode_models[mode]
mode_model.pca_data = dataset.principal_components(args.pca_size)[mode]
parameters, reduced, sample_weight = dataset.reduced(model, mode, args.n)
for base in args.model:
    nn = JaxMLPNetwork.from_file(f"{base}_{mode_name(mode)}_nn.pkl")
    config, network = nn.network.config, nn.network
    eigen = mode_model.pca_data.eigenvalues ** nn.hyper.pc_exponent
    y = reduced * eigen
    units = mode_model.pca_data.principal_components_scaling / eigen
    weights = np.var(y, axis=0) * units**2
    weights /= weights.mean()
    samples = np.ones(len(y)) if sample_weight is None else sample_weight / sample_weight.mean()
    order = np.random.default_rng(config.seed).permutation(len(y))
    n_held = int(round(config.validation_fraction * len(y)))
    errors = np.sum(weights * ((nn.predict(parameters) - y) / network.output_scale) ** 2, axis=1) / weights.size
    held, kept = order[:n_held], order[n_held:]
    loss = lambda i: np.mean(samples[i] * errors[i])  # noqa: E731
    best = network.history[np.argmin(network.history[:, 1])]
    print(f"{base}: training loss {loss(kept):.3g}, held-out {loss(held):.3g} "
          f"(best epoch {best[0]:.0f}/{config.epochs}, logged {best[1]:.3g}); "
          f"held-out errors: median {np.median(errors[held]):.3g}, max {errors[held].max():.3g}")
    worst = np.argsort(errors)[::-1][:10]
    share = errors[worst].sum() / errors.sum()
    print(f"  the 10 worst of all {len(errors)} carry {share:.0%} of the loss:")
    for i in worst:
        print(f"    #{i} ({'held out' if i in set(held) else 'trained'}): error {errors[i]:.3g}, "
              f"parameters {np.round(parameters[i], 3).tolist()}")
