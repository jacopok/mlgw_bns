"""Is the FD validation 'gap' the surrogate, or a TEOB config difference?

`validate_extrinsic_against_teob.py` builds its independent reference with a
bare ``EOBRunPy(initial_frequency=15, srate_interp=effective_srate_hz)`` call.
The generator the surrogate is *trained* on
(``all_modes_amplitude_phase``) instead

  * lowers the ODE start by ``initial_frequency_scaling`` (x0.5 for m_max=4),
  * raises ``srate_interp`` by ``srate_interp_scaling`` (x2 for m_max=4),

precisely to keep every HOM well-conditioned across the band. So the
"surrogate vs independent TEOB" mismatch folds in a TEOB-vs-TEOB config
difference. This probe removes the surrogate entirely and mismatches the
*reference constructions* against each other, as a function of mass ratio.

  A  training generator (all_modes_amplitude_phase), dense grid
  B  bare EOBRunPy(f0=15, srate_interp=srate)             <- the script's ref
  C  EOBRunPy(f0=15 * 0.5, srate_interp=srate * 2)        <- library's scalings

Run: python visualization/probe_teob_config_gap.py
"""

from __future__ import annotations

import copy

import numpy as np

from mlgw_bns.higher_order_modes import Mode, mode_to_k
from mlgw_bns.model import Model
from mlgw_bns.model_validation import ValidateModel
from mlgw_bns.mode_model import ParametersWithExtrinsic
from mlgw_bns.special_func import spinsphericalharm

MODES = [Mode(2, 2), Mode(2, 1), Mode(3, 3), Mode(4, 4)]
M_VALUES = np.array([m.m for m in MODES])
BAND_LO, BAND_HI = 20.0, 2048.0
F0 = 15.0
_MSUN_SECONDS = 4.925490947e-6


def eobrun_modes(params, frequencies, f0, srate, inclination):
    from EOBRun_module import EOBRunPy

    inside = (frequencies >= BAND_LO) & (frequencies <= BAND_HI)
    target = frequencies[inside]
    par = dict(
        q=params.mass_ratio, LambdaAl2=params.lambda_1, LambdaBl2=params.lambda_2,
        chi1=params.chi_1, chi2=params.chi_2, M=params.total_mass,
        distance=params.distance_mpc, inclination=inclination,
        initial_frequency=f0, srate_interp=srate, use_geometric_units="no",
        domain=1, interp_freqs="yes", freqs=list(target), coalescence_angle=0.0,
        output_hpc="no", arg_out="yes", use_spins=1,
        use_mode_lm=sorted({mode_to_k(m) for m in MODES}),
    )
    f_spa, *_, hflm, htlm, dyn = EOBRunPy(par)
    f_spa = np.asarray(f_spa)
    tc = float(dyn["tc"]) if "tc" in dyn else float(np.asarray(htlm["t"])[-1])
    tc_s = tc * params.total_mass * _MSUN_SECONDS
    out = {}
    for m in MODES:
        amp = np.asarray(hflm[str(mode_to_k(m))][0])
        phase = np.asarray(hflm[str(mode_to_k(m))][1]) - 2 * np.pi * tc_s * f_spa
        yr, yi = spinsphericalharm(-2, m.l, m.m, 0.0, inclination)
        out[(m.l, m.m)] = amp * np.exp(-1j * phase) * (yr + 1j * yi)
    return out, inside


def training_modes(model, frequencies, params):
    """Reference built exactly as get_teob_modes_dict, per mode, on `frequencies`."""
    d = model.dataset
    return model.get_teob_modes_dict(frequencies, params)


def mm(ref, other, freqs, psd, m_values=M_VALUES, max_dt=0.02):
    weight = np.gradient(freqs) / psd

    def prod(a, b):
        return np.sum(np.conj(a) * b * weight)

    n1 = np.sqrt(np.abs(prod(ref, ref)))
    rw = np.conj(ref) * weight
    stack = np.asarray(other)
    mv = m_values[:, None]
    phi_grid = np.linspace(-np.pi, np.pi, 361)
    t_grid = np.linspace(-max_dt, max_dt, 2001)
    tk = np.exp(2j * np.pi * np.outer(t_grid, freqs))
    best = -1.0
    for phi in phi_grid:
        h = (stack * np.exp(1j * mv * phi)).sum(axis=0)
        ov = tk @ (rw * h)
        nn = np.sqrt(np.abs(np.sum((np.abs(h) ** 2) * weight)))
        v = np.abs(ov).max() / (n1 * nn)
        best = max(best, v)
    return 1.0 - best


def main():
    model = Model(modes=MODES, filename="mlgw_bns/data/default_hom")
    model.load()
    srate = model.dataset.effective_srate_hz
    validator = ValidateModel(model.mode_models[Mode(2, 2)])

    dense = np.arange(BAND_LO, BAND_HI, 0.5)
    rng = np.random.default_rng(4)

    print(f"{'q':>5} {'A-vs-B':>10} {'A-C(f+s)':>10} {'A-C(f)':>10} {'A-C(s)':>10}")
    for _ in range(24):
        q = rng.uniform(1.0, 3.0)
        params = ParametersWithExtrinsic(
            mass_ratio=q, lambda_1=rng.uniform(5, 5000), lambda_2=rng.uniform(5, 5000),
            chi_1=rng.uniform(-0.5, 0.5), chi_2=rng.uniform(-0.5, 0.5),
            distance_mpc=100.0, inclination=np.arccos(rng.uniform(-1, 1)),
            total_mass=rng.uniform(2.0, 4.0), reference_phase=0.0,
        )
        try:
            A = training_modes(model, dense, params)
            inclA = params.inclination
            B, inside = eobrun_modes(params, dense, F0, srate, inclA)
            C, _ = eobrun_modes(params, dense, F0 * 0.5, srate * 2.0, inclA)
            Cf, _ = eobrun_modes(params, dense, F0 * 0.5, srate, inclA)
            Cs, _ = eobrun_modes(params, dense, F0, srate * 2.0, inclA)
        except Exception as e:  # noqa: BLE001
            print(f"{q:5.2f}  failed: {e}")
            continue
        fb = dense[inside]
        psd = validator.psd_at_frequencies(fb)
        A_s = sum(v[inside] for v in A.values())
        B_l = [B[(m.l, m.m)] for m in MODES]
        C_l = [C[(m.l, m.m)] for m in MODES]
        A_l = [A[(m.l, m.m)][inside] for m in MODES]
        ab = mm(A_s, B_l, fb, psd)
        ac = mm(A_s, C_l, fb, psd)
        acf = mm(A_s, [Cf[(m.l, m.m)] for m in MODES], fb, psd)
        acs = mm(A_s, [Cs[(m.l, m.m)] for m in MODES], fb, psd)
        print(f"{q:5.2f} {ab:10.2e} {ac:10.2e} {acf:10.2e} {acs:10.2e}")


if __name__ == "__main__":
    main()
