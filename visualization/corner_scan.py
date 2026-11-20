"""Aligned-spin full-waveform mismatch vs TEOBResumS on common draws in a
parameter box, for several models (the same points for each).
Usage: corner_scan.py N QMIN QMAX L1MIN L1MAX L2MIN L2MAX MODEL_DIR...
Spins uniform in the training range; prints per-model quantiles and the
worst points, and saves corner_scan_<label>.npz."""
import sys

import numpy as np

import validate_model as vm
from mlgw_bns.dataset_generation import WaveformParameters

n = int(sys.argv[1])
qmin, qmax, l1min, l1max, l2min, l2max = map(float, sys.argv[2:8])
names = sys.argv[8:]
rng = np.random.default_rng(498)
draws = np.column_stack([
    rng.uniform(qmin, qmax, n), rng.uniform(l1min, l1max, n), rng.uniform(l2min, l2max, n),
    rng.uniform(-0.5, 0.5, n), rng.uniform(-0.5, 0.5, n),
])
label = "_".join(sys.argv[2:8])
out = {"draws": draws}
for name in names:
    model = vm.load_model(name)
    params = [WaveformParameters(*row, model.dataset) for row in draws]
    opt, no_opt, _ = vm.full_waveform_mismatches(model, params)
    opt = np.asarray(opt)
    out[name] = opt
    worst = np.argsort(opt)[-3:][::-1]
    print(f"{name[-45:]}: median {np.median(opt):.1e}  90% {np.quantile(opt, .9):.1e}  max {opt.max():.1e}  "
          f">1e-5 {np.mean(opt > 1e-5):.0%}  worst (q, L1, L2, chi1, chi2) = {np.round(draws[worst], 2).tolist()}")
np.savez(f"corner_scan_{label}.npz", **{k.replace('/', '|'): v for k, v in out.items()})
