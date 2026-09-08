"""How self-consistent is a frequency-domain multi-mode TEOBResumS waveform?

Two studies, no surrogate in the loop --- both mismatch TEOBResumS against
*itself*, built with a different configuration, over the ``default_hom``
parameter box (q in [1, 3], M in [2, 4] Msun, Lambda in [5, 5000],
|chi| < 0.5), with a per-mode ``exp(i m phi_c)`` + time-shift optimisation.

``--abc`` (the original probe)
    ``validate_extrinsic_against_teob.py`` used to build its independent
    reference with a bare ``EOBRunPy(initial_frequency=15)``. The training
    generator ``all_modes_amplitude_phase`` instead lowers the ODE start by
    ``initial_frequency_scaling`` (x0.5 for m_max=4) and raises
    ``srate_interp`` by ``srate_interp_scaling`` (x2). This mismatches

      A  training generator (all_modes_amplitude_phase)
      B  bare EOBRunPy(f0=15, srate_interp=srate)
      C  EOBRunPy(f0=15 * 0.5, srate_interp=srate * 2)  (+ f-only, +s-only)

    and finds the whole A-vs-B gap (~1e-4, growing with q) is the ODE start:
    A-vs-C(f) ~3e-6, A-vs-C(s) == A-vs-B.

``--ladder`` (default)
    Sweeps the ODE start frequency ``f0`` (the (2,2) GW frequency at the
    start of the integration) and mismatches each against the training
    generator (ODE from ~1.8 Hz, taken as converged). Result, over the
    ``default_hom`` box:

      * over [40, 2048] Hz the FD multi-mode waveform is *independent* of
        ``f0`` from 3 to 18 Hz --- self-consistent to ~1e-9 median,
        ~1e-7 worst (the worst grows with mass ratio: more HOM content).
        No phase drift, no initial-data transient reaching the band.
      * over [20, 2048] Hz that same ~5e-9 floor holds only for
        ``f0 <= 7`` Hz. It climbs to ~2e-6 at ``f0 = 13`` and ~8e-5 at
        ``f0 = 15`` --- purely the kinematic cutoff: the ``(l, m)``
        multipole is identically zero below ``(m/2) f0``, so a 15 Hz
        start has no (3,3) below 22.5 Hz and no (4,4) below 30 Hz.

    ``--band-lo`` sets the comparison band's low edge (default 40; use 20
    to see the cutoff). Writes ``teob_self_consistency_<lo>_2048.{png,npz}``.

Run: python visualization/probe_teob_config_gap.py [--abc] [--band-lo HZ]
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import minimize

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

#: ODE start frequencies (Hz, the (2,2) GW frequency) for ``--ladder``.
F0_GRID = [3.0, 4.0, 5.0, 7.0, 10.0, 13.0, 15.0, 18.0]
#: The ladder mismatch is measured here: above (18/2) * (4/2) = 36 Hz every
#: mode has support for every ``f0`` in ``F0_GRID`` (the (l, m) mode is
#: identically zero below (m/2) f0), so the number is initial-data error,
#: not the kinematic band-edge cutoff.
LADDER_BAND_LO = 40.0


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
    """Reference built exactly as ``get_teob_modes_dict`` (ODE from ~1.8 Hz)."""
    return model.get_teob_modes_dict(frequencies, params)


def optimised_mm(ref, others, freqs, psd, m_values=M_VALUES, max_dt=0.02):
    """1 - max over (t_c, per-mode phi_c) of the normalised overlap."""
    weight = np.gradient(freqs) / psd
    n1 = np.sqrt(np.abs(np.sum(np.conj(ref) * ref * weight)))
    rw = np.conj(ref) * weight
    stack = np.asarray(others)
    mv = np.asarray(m_values)[:, None]

    def combined(phi):
        return (stack * np.exp(1j * mv * phi)).sum(axis=0)

    def match(t_c, phi_c):
        h = combined(phi_c) * np.exp(2j * np.pi * freqs * t_c)
        n2 = np.sqrt(np.abs(np.sum(np.conj(h) * h * weight)))
        if n2 <= 0:
            return 0.0
        return np.abs(np.sum(rw * h)) / (n1 * n2)

    phi_grid = np.linspace(-np.pi, np.pi, 241)
    t_grid = np.linspace(-max_dt, max_dt, 2001)
    t_kernels = np.exp(2j * np.pi * np.outer(t_grid, freqs))

    best, arg = -1.0, (0.0, 0.0)
    for phi_c in phi_grid:
        h = combined(phi_c)
        overlaps = t_kernels @ (rw * h)
        norm = np.sqrt(np.abs(np.sum((np.abs(h) ** 2) * weight)))
        k = int(np.argmax(np.abs(overlaps)))
        value = np.abs(overlaps[k]) / (n1 * norm)
        if value > best:
            best, arg = value, (t_grid[k], phi_c)

    res = minimize(
        lambda x: -match(x[0], x[1]), arg, method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 600},
    )
    return 1.0 - max(best, -res.fun)


def abc_probe(model, srate, validator, n_binaries):
    dense = np.arange(BAND_LO, BAND_HI, 0.5)
    rng = np.random.default_rng(4)

    print(f"{'q':>5} {'A-vs-B':>10} {'A-C(f+s)':>10} {'A-C(f)':>10} {'A-C(s)':>10}")
    for _ in range(n_binaries):
        q = rng.uniform(1.0, 3.0)
        params = ParametersWithExtrinsic(
            mass_ratio=q, lambda_1=rng.uniform(5, 5000), lambda_2=rng.uniform(5, 5000),
            chi_1=rng.uniform(-0.5, 0.5), chi_2=rng.uniform(-0.5, 0.5),
            distance_mpc=100.0, inclination=np.arccos(rng.uniform(-1, 1)),
            total_mass=rng.uniform(2.0, 4.0), reference_phase=0.0,
        )
        try:
            A = training_modes(model, dense, params)
            incl = params.inclination
            B, inside = eobrun_modes(params, dense, F0, srate, incl)
            C, _ = eobrun_modes(params, dense, F0 * 0.5, srate * 2.0, incl)
            Cf, _ = eobrun_modes(params, dense, F0 * 0.5, srate, incl)
            Cs, _ = eobrun_modes(params, dense, F0, srate * 2.0, incl)
        except Exception as e:  # noqa: BLE001
            print(f"{q:5.2f}  failed: {e}")
            continue
        fb = dense[inside]
        psd = validator.psd_at_frequencies(fb)
        A_s = sum(v[inside] for v in A.values())
        get = lambda d: [d[(m.l, m.m)] for m in MODES]  # noqa: E731
        print(
            f"{q:5.2f} {optimised_mm(A_s, get(B), fb, psd):10.2e} "
            f"{optimised_mm(A_s, get(C), fb, psd):10.2e} "
            f"{optimised_mm(A_s, get(Cf), fb, psd):10.2e} "
            f"{optimised_mm(A_s, get(Cs), fb, psd):10.2e}",
            flush=True,
        )


def ladder_study(model, srate, validator, n_binaries, band_lo=LADDER_BAND_LO):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dense = np.arange(BAND_LO, BAND_HI, 0.5)
    tag = f"{band_lo:g}_{int(BAND_HI)}"
    rng = np.random.default_rng(4)

    rows: list[dict] = []
    for i in range(n_binaries):
        q = rng.uniform(1.0, 3.0)
        params = ParametersWithExtrinsic(
            mass_ratio=q, lambda_1=rng.uniform(5, 5000), lambda_2=rng.uniform(5, 5000),
            chi_1=rng.uniform(-0.5, 0.5), chi_2=rng.uniform(-0.5, 0.5),
            distance_mpc=100.0, inclination=np.arccos(rng.uniform(-1, 1)),
            total_mass=rng.uniform(2.0, 4.0), reference_phase=0.0,
        )
        incl = params.inclination
        try:
            ref_modes = training_modes(model, dense, params)
            per_f0 = {
                f0: eobrun_modes(params, dense, f0, srate, incl)[0] for f0 in F0_GRID
            }
        except Exception as e:  # noqa: BLE001
            print(f"  binary {i} (q={q:.2f}) failed: {e}", flush=True)
            continue

        inside = (dense >= BAND_LO) & (dense <= BAND_HI)
        fsub = dense[inside]
        band = fsub >= band_lo
        fb = fsub[band]
        psd = validator.psd_at_frequencies(fb)
        ref_strain = sum(v[inside][band] for v in ref_modes.values())

        row = {"q": q, "total_mass": params.total_mass, "inclination": incl}
        for f0 in F0_GRID:
            arrs = [per_f0[f0][(m.l, m.m)][band] for m in MODES]
            row[f0] = optimised_mm(ref_strain, arrs, fb, psd)
        rows.append(row)
        print(
            f"  {len(rows)}/{n_binaries} q={q:.2f} M={params.total_mass:.2f}: "
            + " ".join(f"{f0:g}={row[f0]:.1e}" for f0 in F0_GRID),
            flush=True,
        )

    qs = np.array([r["q"] for r in rows])
    table = {f0: np.array([r[f0] for r in rows]) for f0 in F0_GRID}

    print(
        f"\n  mismatch vs the training generator (ODE ~1.8 Hz), "
        f"[{band_lo:g}, {BAND_HI:g}] Hz, {len(rows)} binaries\n"
        f"  {'f0 [Hz]':>8} {'median':>10} {'p90':>10} {'max':>10} {'r(log mm, q)':>13}"
    )
    for f0 in F0_GRID:
        v = table[f0]
        r = np.corrcoef(qs, np.log10(np.clip(v, 1e-12, None)))[0, 1]
        print(f"  {f0:8.1f} {np.median(v):10.2e} {np.percentile(v, 90):10.2e} "
              f"{v.max():10.2e} {r:13.2f}")

    plateau = np.concatenate([table[f0] for f0 in F0_GRID if f0 <= 5.0])
    print(
        f"\n  self-consistency floor (f0 <= 5 Hz): median {np.median(plateau):.2e}, "
        f"p90 {np.percentile(plateau, 90):.2e}, max {plateau.max():.2e}"
    )
    hi = table[15.0]
    print(
        f"  f0 = 15 Hz            : median {np.median(hi):.2e}, "
        f"p90 {np.percentile(hi, 90):.2e}, max {hi.max():.2e}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    for r in rows:
        axes[0].plot(F0_GRID, [r[f0] for f0 in F0_GRID], color="0.75", lw=0.8)
    axes[0].plot(F0_GRID, [np.median(table[f0]) for f0 in F0_GRID], "o-",
                 color="tab:red", lw=2.2, label="median")
    axes[0].plot(F0_GRID, [np.percentile(table[f0], 90) for f0 in F0_GRID], "s--",
                 color="tab:orange", lw=1.6, label="90th pct")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"ODE start frequency $f_0$ [Hz]  ($(2,2)$ GW frequency)")
    axes[0].set_ylabel(
        rf"mismatch vs converged TEOBResumS, $[{band_lo:g}, {BAND_HI:g}]$ Hz"
    )
    axes[0].set_title("FD multi-mode TEOBResumS self-consistency")
    axes[0].legend()

    edges = np.percentile(qs, [0, 33, 67, 100])
    for lo, hi_q, color in ((edges[0], edges[1], "tab:blue"),
                            (edges[1], edges[2], "tab:green"),
                            (edges[2], edges[3], "tab:red")):
        mask = (qs >= lo) & (qs <= hi_q)
        med = [np.median(table[f0][mask]) for f0 in F0_GRID]
        axes[1].plot(F0_GRID, med, "o-", color=color,
                     label=rf"$q \in [{lo:.2f}, {hi_q:.2f}]$")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"ODE start frequency $f_0$ [Hz]")
    axes[1].set_ylabel("median mismatch")
    axes[1].set_title("by mass-ratio tertile")
    axes[1].legend()

    axes[2].scatter(qs, table[15.0], s=18, color="tab:red", label=r"$f_0 = 15$ Hz")
    axes[2].scatter(qs, table[5.0], s=18, color="tab:blue", label=r"$f_0 = 5$ Hz")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("mass ratio $q$")
    axes[2].set_ylabel(r"mismatch vs converged TEOBResumS")
    axes[2].set_title("start-frequency error vs mass ratio")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(f"visualization/teob_self_consistency_{tag}.png", dpi=150)
    np.savez(
        f"visualization/teob_self_consistency_{tag}.npz",
        f0_grid=np.array(F0_GRID), q=qs,
        total_mass=np.array([r["total_mass"] for r in rows]),
        inclination=np.array([r["inclination"] for r in rows]),
        mismatch=np.stack([table[f0] for f0 in F0_GRID], axis=1),
    )
    print(f"\n  wrote visualization/teob_self_consistency_{tag}.{{png,npz}}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abc", action="store_true",
                        help="run the A/B/C config probe instead of the ladder")
    parser.add_argument("--n-binaries", type=int, default=26)
    parser.add_argument("--band-lo", type=float, default=LADDER_BAND_LO,
                        help="low edge of the ladder comparison band [Hz]")
    args = parser.parse_args()

    model = Model(modes=MODES, filename="mlgw_bns/data/default_hom")
    model.load()
    srate = model.dataset.effective_srate_hz
    validator = ValidateModel(model.mode_models[Mode(2, 2)])

    if args.abc:
        abc_probe(model, srate, validator, args.n_binaries)
    else:
        ladder_study(model, srate, validator, args.n_binaries, args.band_lo)


if __name__ == "__main__":
    main()
