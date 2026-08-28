r"""Crux check for the batched-EOB refactor.

Does one TEOBResumS call with ``use_mode_lm = [1, 4, 0, 8]`` (all of
(2,2),(3,3),(2,1),(4,4), ``initial_frequency`` lowered by 0.5 because of
the (4,4)) reproduce, mode by mode, what the current one-call-per-mode
path (``TEOBResumSModeGenerator.effective_one_body_waveform``, factor 1.0
for (2,2)/(2,1), 0.667 for (3,3)) produces on the requested grid?

Prints max |dphase| and max |damp/amp| per mode over a handful of random
parameter points. Physically these should be at interpolation-noise level
(~1e-8); a real in-band shift is a bug to localise.

Run: python visualization/multimode_eob_equivalence.py
"""

import numpy as np

from mlgw_bns.dataset_generation import Dataset, WaveformParameters
from mlgw_bns.higher_order_modes import (
    Mode,
    mode_to_k,
    start_integration_early,
    teob_mode_generator_factory,
)

MODES = [Mode(2, 2), Mode(3, 3), Mode(2, 1), Mode(4, 4)]


def multimode_call(gen, params, modes, frequencies):
    """One EOB call for every mode in ``modes``; mirrors the post-processing
    of ``effective_one_body_waveform`` (align to merger, negate)."""
    par_dict = params.teobresums()
    to_slice = start_integration_early(par_dict, frequencies, modes)
    par_dict["arg_out"] = "yes"
    par_dict["use_mode_lm"] = [mode_to_k(m) for m in modes]

    f_spa, _, _, _, _, hflm, htlm, dyn = gen.eobrun_callable(par_dict)
    f_spa = f_spa[to_slice]
    tc = gen._merger_time(dyn, htlm)

    out = {}
    for m in modes:
        k = str(mode_to_k(m))
        amp = hflm[k][0][to_slice] * params.eta
        phase = gen._align_mode_phase_to_merger(hflm[k][1][to_slice], f_spa, tc)
        out[m] = (f_spa, amp, -phase)
    return out


def main():
    dataset = Dataset(initial_frequency_hz=20.0, srate_hz=4096.0, multibanding=True)
    freqs = dataset.frequencies
    gens = {m: teob_mode_generator_factory(m) for m in MODES}
    gen_any = gens[Mode(2, 2)]

    pgen = dataset.make_parameter_generator(seed=11)
    worst = {m: [0.0, 0.0] for m in MODES}

    fhz = dataset.natural_units_to_hz(freqs)
    worst = {m: [0.0, 0.0, 0.0] for m in MODES}

    n = 12
    for _ in range(n):
        p = next(pgen)
        multi = multimode_call(gen_any, p, MODES, freqs)
        for m in MODES:
            f_s, amp_s, phi_s = gens[m].effective_one_body_waveform(p, freqs)
            f_m, amp_m, phi_m = multi[m]
            # weight by amplitude: only the in-band region matters
            finite = np.isfinite(amp_s) & np.isfinite(phi_s) & np.isfinite(phi_m)
            w = finite & (amp_s > 1e-3 * np.nanmax(amp_s))
            if w.sum() < 10:
                continue
            dphi_raw = np.max(np.abs(phi_s - phi_m)[w])
            # remove a constant + linear-in-f trend (absorbed by the
            # reference-phase and time-shift predictors anyway)
            A = np.vstack([np.ones(w.sum()), fhz[w]]).T
            resid = (phi_s - phi_m)[w]
            coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
            dphi_detrended = np.max(np.abs(resid - A @ coef))
            damp = np.max(np.abs((amp_s - amp_m) / amp_s)[w])
            worst[m][0] = max(worst[m][0], dphi_raw)
            worst[m][1] = max(worst[m][1], dphi_detrended)
            worst[m][2] = max(worst[m][2], damp)

    print(f"over {n} random parameter points, multiband grid ({len(freqs)} pts):")
    print("  (amplitude-weighted band; detrended = after removing const + linear-in-f)")
    for m in MODES:
        raw, det, damp = worst[m]
        flag = "  <-- LARGE" if (det > 1e-5 or damp > 1e-4) else ""
        print(f"  ({m.l},{m.m}):  max|dphase|_raw = {raw:.3e}   "
              f"detrended = {det:.3e} rad   max|damp/amp| = {damp:.3e}{flag}")


if __name__ == "__main__":
    main()
