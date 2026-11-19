"""Merger-time errors of the shipped models against TEOBResumS: over uniform
draws of each model's box, and in the low-Lambda_1 corner of the tests.
MODEL is a shipped name (default: all of them) or a saved base path
(``.../runs/log60_32768/model``).

    python visualization/merger_time_release.py [N [MODEL ...]]
"""
import sys
import warnings
from multiprocessing import get_context

import numpy as np

from mlgw_bns import Model, ParametersWithExtrinsic
from mlgw_bns.dataset_generation import ParameterSet
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model import MODELS_AVAILABLE, ModelAccuracyWarning
from mlgw_bns.model_validation import ValidateModel

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "tests"))
from test_model import LOW_LAMBDA_1_CORNER  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
NAMES = sys.argv[2:] or MODELS_AVAILABLE
CHUNKS = 8


def errors(args):
    name, chunk = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ModelAccuracyWarning)
        if "/" in name:
            model = Model(modes=[Mode(2, 2)], filename=name)
            model.load()
        else:
            model = Model.default_for_testing(name, modes=[Mode(2, 2)])
    val = ValidateModel(model.mode_models[Mode(2, 2)])
    if chunk < 0:
        ds = val.model.dataset
        intr = [ParametersWithExtrinsic(mass_ratio=q, lambda_1=a, lambda_2=b, chi_1=c, chi_2=d, distance_mpc=100.,
                                        inclination=1., total_mass=ds.total_mass).intrinsic(ds)
                for q, a, b, c, d in LOW_LAMBDA_1_CORNER]
        ps = ParameterSet.from_list_of_waveform_parameters(intr)
    else:
        ps = val.param_set(N // CHUNKS, seed=100 + chunk)
    dt, _, valid = val.merger_reference_errors(ps)
    return name, chunk, dt, valid.parameter_array


if __name__ == "__main__":
    jobs = [(n, c) for n in NAMES for c in [-1, *range(CHUNKS)]]
    with get_context("fork").Pool(8) as pool:
        out = pool.map(errors, jobs)
    for name in NAMES:
        corner = np.concatenate([dt for n, c, dt, _ in out if n == name and c < 0])
        dt = np.concatenate([dt for n, c, dt, _ in out if n == name and c >= 0])
        par = np.concatenate([p for n, c, _, p in out if n == name and c >= 0])
        a = np.abs(dt) * 1e6
        worst = par[np.argmax(a)]
        print(f"{name[-48:]:>24}: |dt| median {np.median(a):5.1f}  90% {np.quantile(a, .9):6.1f}  99% {np.quantile(a, .99):6.1f}"
              f"  max {a.max():6.0f} us  >0.2 ms {np.mean(a > 200):.1%}  (n={len(a)}; worst q,L1,L2,chi1,chi2 = {np.round(worst, 2)})")
        print(f"{'':>24}  low-Lambda_1 corner: {np.round(corner * 1e6).astype(int)} us")
