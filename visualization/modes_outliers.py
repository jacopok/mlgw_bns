"""Waveforms of a sharded modes dataset whose principal components (in the
basis of its first PCA_SIZE) are outliers: the largest of each mode, in
units of the per-component spread, and in which shard they are.

    python visualization/modes_outliers.py DATASET --pca-size 16384 [--n N] [--modes l2_m2 ...]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from modes_scale import parse_mode  # noqa: E402

from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.modes_dataset import ShardedModesDataset

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
parser.add_argument("--pca-size", type=int, required=True)
parser.add_argument("--n", type=int)
parser.add_argument("--modes", type=parse_mode, nargs="+")
parser.add_argument("--top", type=int, default=8)
args = parser.parse_args()

dataset = ShardedModesDataset(args.dataset)
model = dataset.model()
pca = dataset.principal_components(args.pca_size)
for mode in dataset.modes:
    model.mode_models[mode].pca_data = pca[mode]
shard = dataset.config.shard_size
for mode in [Mode(*m) for m in args.modes] if args.modes else dataset.modes:
    parameters, reduced, _ = dataset.reduced(model, mode, args.n)
    units = pca[mode].principal_components_scaling
    # the natural loss of each waveform: its squared residual norm, as a fraction of the average
    power = np.sum((reduced * units) ** 2, axis=1)
    z = np.max(np.abs(reduced - np.median(reduced, axis=0)) / (1.4826 * np.median(np.abs(
        reduced - np.median(reduced, axis=0)), axis=0)), axis=1)
    order = np.argsort(power)[::-1][: args.top]
    print(f"{mode}: {len(power)} waveforms; residual power median {np.median(power):.3g}, "
          f"max/median {power.max() / np.median(power):.3g}; share of the top {args.top}: "
          f"{power[order].sum() / power.sum():.1%}")
    for i in order:
        print(f"  #{i} (shard {i // shard}): power {power[i] / np.median(power):.3g} x median, "
              f"robust z {z[i]:.3g}, parameters {np.round(parameters[i], 3)}")
