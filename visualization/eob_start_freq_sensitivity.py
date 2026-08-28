r"""How sensitive is the in-band per-mode EOB waveform to where the ODE
integration starts?

The batched-EOB refactor would force ``initial_frequency`` down by the
(4,4) factor 0.5 for every mode, where the (2,2)/(2,1) single-mode path
uses factor 1.0 and (3,3) uses 0.667. This checks whether factor 0.5 is
near-converged (difference 0.5-vs-0.35 << 0.5-vs-1.0) or whether 1.0 is
special.

Run: python visualization/eob_start_freq_sensitivity.py
"""

import numpy as np

from mlgw_bns.dataset_generation import Dataset
from mlgw_bns.higher_order_modes import (
    Mode, mode_to_k, start_integration_early, teob_mode_generator_factory)

MODES = [Mode(2, 2), Mode(2, 1), Mode(3, 3)]
FACTORS = [1.0, 0.667, 0.5, 0.35]


def call(gen, params, mode, frequencies, factor):
    par_dict = params.teobresums()
    f_0 = par_dict["initial_frequency"]
    df = par_dict["df"]
    # replicate start_integration_early but with an explicit factor
    par_dict["freqs"] = list(np.insert(
        frequencies, 0, np.arange(f_0 - df * 256, f_0, step=df)))
    par_dict.pop("df")
    par_dict["interp_freqs"] = "yes"
    par_dict["initial_frequency"] = f_0 * factor - df * 256
    par_dict["arg_out"] = "yes"
    par_dict["use_mode_lm"] = [mode_to_k(mode)]
    to_slice = slice(-len(frequencies), None)
    f_spa, _, _, _, _, hflm, htlm, dyn = gen.eobrun_callable(par_dict)
    f_spa = f_spa[to_slice]
    k = str(mode_to_k(mode))
    amp = hflm[k][0][to_slice] * params.eta
    phase = gen._align_mode_phase_to_merger(hflm[k][1][to_slice], f_spa,
                                            gen._merger_time(dyn, htlm))
    return amp, -phase


def detrended_dphi(fhz, w, a, b):
    A = np.vstack([np.ones(w.sum()), fhz[w]]).T
    r = (a - b)[w]
    c, *_ = np.linalg.lstsq(A, r, rcond=None)
    return np.max(np.abs(r - A @ c))


def main():
    dataset = Dataset(initial_frequency_hz=20.0, srate_hz=4096.0, multibanding=True)
    freqs = dataset.frequencies
    fhz = dataset.natural_units_to_hz(freqs)
    gens = {m: teob_mode_generator_factory(m) for m in MODES}
    pgen = dataset.make_parameter_generator(seed=11)

    acc = {m: {f: [0.0, 0.0] for f in FACTORS} for m in MODES}
    n = 8
    for _ in range(n):
        p = next(pgen)
        for m in MODES:
            ref_amp, ref_phi = call(gens[m], p, m, freqs, 0.35)  # earliest = reference
            w = np.isfinite(ref_amp) & (ref_amp > 1e-3 * np.nanmax(ref_amp))
            for fac in FACTORS:
                a, ph = call(gens[m], p, m, freqs, fac)
                dphi = detrended_dphi(fhz, w, ph, ref_phi)
                damp = np.max(np.abs((a - ref_amp) / ref_amp)[w])
                acc[m][fac][0] = max(acc[m][fac][0], dphi)
                acc[m][fac][1] = max(acc[m][fac][1], damp)

    print(f"max diff vs factor 0.35 (earliest start), {n} points, detrended phase:")
    for m in MODES:
        print(f"  ({m.l},{m.m}):")
        for fac in FACTORS:
            d, a = acc[m][fac]
            print(f"     factor {fac:5.3f}:  dphi = {d:.3e} rad   damp/amp = {a:.3e}")


if __name__ == "__main__":
    main()
