"""Jitter of TEOBResumS' (2,2) time-frequency map between neighbouring
parameters, at fixed frequencies and at the merger (the tangent at the top
of the trained band).

Fine scans in Lambda_1 around random binaries; the times from the training
data's reference (value and slope fitted at f0) to each frequency are
detrended with a quartic, and the rms remainder is printed. The EOBRun_module
on the path is the one tested:

    PYTHONPATH=.:../teobresums-tf/.build-base python \
        visualization/teob_reference_jitter.py [N_BINARIES]
"""
import os
import sys
from multiprocessing import get_context

import numpy as np

import EOBRun_module
from mlgw_bns import Model, ParametersWithExtrinsic
from mlgw_bns.higher_order_modes import Mode

HZ = [30.0, 100.0, 400.0, 800.0, 1200.0, 1500.0, 1800.0, 2200.0, 2700.0, "top"]
STEPS = 41
STEP = 0.2  # in Lambda_1


def scan(seed):
    rng = np.random.default_rng(seed)
    model = Model.default_for_testing("default_hom_lambda2000", modes=[Mode(2, 2)])
    mode_model = model.mode_models[Mode(2, 2)]
    ds = mode_model.dataset
    q, l2 = rng.uniform(1, 2), rng.uniform(50, 1500)
    l1 = rng.uniform(50, 1500)
    c1, c2 = rng.uniform(-0.3, 0.3, 2)
    knots = ds.frequencies_hz[mode_model.downsampling_indices.phase_indices]
    f = np.array([knots[-1] * 0.999 if h == "top" else h for h in HZ])
    pairs = np.sort(np.concatenate([f * 0.9997, f * 1.0003]))
    start = np.linspace(20.5, 25, 200)
    # the training grid is fine enough for TEOBResumS not to slip by 2 pi
    grid = np.union1d(ds.frequencies_hz, np.union1d(start, pairs))
    generator = mode_model.waveform_generator
    times = []
    for i in range(STEPS):
        p = ParametersWithExtrinsic(q, l1 + STEP * i, l2, c1, c2, 100., 0., ds.total_mass).intrinsic(ds)
        _, _, phase = generator.all_modes_amplitude_phase(
            p, [Mode(2, 2)], grid * ds.mass_sum_seconds, settings_modes=[Mode(4, 4)])[Mode(2, 2)]
        pn = generator.post_newtonian_phase(p, grid * ds.mass_sum_seconds)
        res = phase - pn
        n = np.searchsorted(grid, start)
        slope0 = np.polyfit(start - 20.5, res[n], 2)[1]
        a, b = (np.searchsorted(grid, x) for x in (f * 0.9997, f * 1.0003))
        times.append(((phase[b] - phase[a]) / (grid[b] - grid[a]) - slope0) / (2 * np.pi))
    times = np.array(times) * 1e6
    x = np.linspace(-1, 1, STEPS)
    out = times - np.polynomial.polynomial.polyval(
        x, np.polynomial.polynomial.polyfit(x, times, 4)).T
    return np.sqrt(np.mean(out ** 2, axis=0)), np.abs(np.diff(out, axis=0)).max(axis=0)


if __name__ == "__main__":
    print(EOBRun_module.__file__)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    with get_context("fork").Pool(int(os.environ.get("N_JOBS", "8"))) as pool:
        rms, jump = map(np.array, zip(*pool.map(scan, range(n))))
    print("worst seeds", np.argsort(-rms[:, -1])[:5], np.sort(-rms[:, -1])[:5])
    print("      f   rms median    max   largest jump (us)")
    for i, h in enumerate(HZ):
        print(f"{h!s:>7}  {np.median(rms[:, i]):9.3f} {rms[:, i].max():8.3f}  {jump[:, i].max():8.3f}")
