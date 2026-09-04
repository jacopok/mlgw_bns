r"""Reproduce the (4,4) reference-phase 2*pi branch slip from bare TEOBResumS.

Bypasses every bit of mlgw_bns's residual / PCA / regression machinery.
For a fine ``q`` scan (other intrinsic parameters fixed) it calls
``EOBRunPy`` directly, pulls the raw ``hflm`` phase of one mode at the
reference frequency ``f0``, and walks the exact mlgw_bns post-processing
chain one step at a time:

  0. raw ``hflm[k_lm]`` phase at ``f0``                     (bare TEOBResumS)
  1. + merger-time alignment          ``- 2*pi*tc*f``       (_align_mode_phase_to_merger)
  2. + sign flip                      ``-> -phase``         (ModePhasesNN target convention)
  3. - Taylor-F2 (l,m) PN phase at f0                       (the actual regression target)

plus the merger time ``tc`` itself, which is the term the ``- 2*pi*tc*f``
correction depends on.

Each phase curve has a hardcoded smooth trend ``a*M_lm(q) + b*q + c*q^2 + d``
subtracted (``M_lm`` = :func:`reference_phase_backbone`, i.e. the same
analytic backbone the regressor uses; coefficients least-squares-fit on
the scan and printed); ``tc`` is detrended with a cubic in ``q``. An
isolated step of ~2*pi against an otherwise-smooth residual is the branch
slip. Whichever panel it first appears in is where it is born.

Run with::

    python repro_2pi_jump.py --mode 4,4 --q-lo 2.0 --q-hi 2.3 --n 400
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from mlgw_bns.model import DEFAULT_MODES, Model
from mlgw_bns.higher_order_modes import Mode, mode_to_k, start_integration_early
from mlgw_bns.pn_modes import Mode as PNMode, reference_phase_backbone
from mlgw_bns.dataset_generation import WaveformParameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
TWO_PI = 2.0 * np.pi


def one_q(q, lam, chi1, chi2, mode, k, f_ref_natural, f0_natural, ds, pn_phase):
    """Bare TEOBResumS call for one q; returns the chain stages at f0 + tc."""
    from EOBRun_module import EOBRunPy  # type: ignore

    params = WaveformParameters(q, lam, lam, chi1, chi2, dataset=ds)
    par_dict = params.teobresums()
    to_slice = start_integration_early(par_dict, f_ref_natural, [mode])
    par_dict["arg_out"] = "yes"
    par_dict["use_mode_lm"] = [k]

    try:
        f_spa, _, _, _, _, hflm, htlm, dyn = EOBRunPy(par_dict)
    except Exception:  # pragma: no cover
        return None

    f_spa = np.asarray(f_spa)[to_slice]
    raw = np.asarray(hflm[str(k)][1])[to_slice]
    if len(f_spa) != len(f_ref_natural) or not np.all(np.isfinite(raw)):
        return None

    tc = float(dyn["tc"]) if "tc" in dyn else float(np.asarray(htlm["t"])[-1])
    aligned = raw - TWO_PI * tc * f_spa                      # == _align_mode_phase_to_merger
    flipped = -aligned
    pn = pn_phase(params, f_spa)
    target = flipped - pn

    i0 = 0
    assert abs(f_spa[i0] - f0_natural) < 1e-6 * f0_natural
    return (float(raw[i0]), float(aligned[i0]), float(flipped[i0]),
            float(target[i0]), tc, float(f_spa[i0]))


def isolated_steps(y, min_abs=2.0, ratio=6.0):
    """Indices i where |y[i+1]-y[i]| is a large isolated spike."""
    dd = np.abs(np.diff(y))
    med = np.median(dd)
    return np.where((dd > min_abs) & (dd > ratio * med))[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=str, default="4,4")
    parser.add_argument("--q-lo", type=float, default=2.0)
    parser.add_argument("--q-hi", type=float, default=2.3)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--lambda", dest="lam", type=float, default=500.0)
    parser.add_argument("--chi1", type=float, default=0.0)
    parser.add_argument("--chi2", type=float, default=0.0)
    parser.add_argument("--initial-frequency-hz", type=float, default=5.0)
    parser.add_argument("--grid-points", type=int, default=64)
    parser.add_argument("--fmax-hz", type=float, default=512.0)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    l, m = (int(x) for x in args.mode.split(","))
    mode, k = Mode(l, m), mode_to_k(Mode(l, m))
    out = args.out or f"repro_2pi_jump_{l}{m}.png"

    model = Model(modes=list(DEFAULT_MODES),
                  initial_frequency_hz=args.initial_frequency_hz,
                  reference_amplitude=True)
    _grid_hz, f_ref_natural, f0_natural = model._reference_grid(
        args.grid_points, args.fmax_hz
    )
    ds = model.mode_models[Mode(2, 2)].dataset
    pn_phase = model.mode_models[mode].waveform_generator.post_newtonian_phase

    q = np.linspace(args.q_lo, args.q_hi, args.n)
    rows = Parallel(n_jobs=args.n_jobs)(
        delayed(one_q)(qi, args.lam, args.chi1, args.chi2, mode, k,
                       f_ref_natural, f0_natural, ds, pn_phase)
        for qi in q
    )
    ok = np.array([r is not None for r in rows])
    q = q[ok]
    raw, aligned, flipped, target, tc, f0v = np.array(
        [r for r in rows if r is not None]
    ).T
    logging.info("%d/%d valid EOB calls; f0=%.6e (spread %.2e)",
                 ok.sum(), len(ok), f0v.mean(), f0v.std())

    M_lm = reference_phase_backbone(
        np.column_stack([q, np.full_like(q, args.lam), np.full_like(q, args.lam),
                         np.full_like(q, args.chi1), np.full_like(q, args.chi2)]),
        f0_natural, PNMode(l, m),
    )
    basis = np.column_stack([M_lm, q, q ** 2, np.ones_like(q)])   # hardcoded trend basis

    def detrend_phase(sig):
        coef, *_ = np.linalg.lstsq(basis, sig, rcond=None)
        return sig - basis @ coef, coef

    panels = []
    for name, sig in [
        ("0: raw hflm phase", raw),
        ("1: + merger-time align  (- 2pi tc f)", aligned),
        ("2: + sign flip  (-> -phase)", flipped),
        (f"3: - Taylor-F2 ({l},{m}) PN  =  regression target", target),
    ]:
        det, coef = detrend_phase(sig)
        steps = isolated_steps(det)
        panels.append((name, det, steps))
        logging.info(
            "%-42s trend[M,q,q2,1]=%s  isolated 2pi-ish steps at q=%s",
            name, np.array2string(coef, precision=3),
            [round(float(q[i]), 4) for i in steps] or "none",
        )

    # tc detrended with a cubic in q
    tc_coef = np.polyfit(q, tc, 3)
    tc_det = tc - np.polyval(tc_coef, q)
    tc_steps = isolated_steps(tc_det, min_abs=0.0, ratio=8.0)
    # a tc jump of 1/f0 puts a full 2pi into (- 2pi tc f0)
    logging.info(
        "tc: isolated steps at q=%s ; 1/f0 = %.4e ; largest d(tc)=%.4e",
        [round(float(q[i]), 4) for i in tc_steps] or "none",
        1.0 / f0_natural, np.max(np.abs(np.diff(tc))),
    )

    fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    for ax, (name, det, steps) in zip(axes[:4], panels):
        ax.plot(q, det, ".-", ms=3, lw=0.8)
        for i in steps:
            ax.axvline(q[i], color="tab:red", lw=1.0)
        ax.set_ylabel("detrended [rad]")
        ax.set_title(name +
                     (f"   <-- step at q={q[steps[0]]:.4f}" if len(steps) else ""),
                     fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", lw=0.5)
    axes[4].plot(q, tc_det, ".-", ms=3, lw=0.8, color="tab:green")
    for i in tc_steps:
        axes[4].axvline(q[i], color="tab:red", lw=1.0)
    axes[4].set_ylabel(r"$t_c$ detrended [M]")
    axes[4].set_title(
        f"merger time $t_c$ (cubic-in-q detrended); $1/f_0$={1/f0_natural:.3e} M "
        f"puts $2\\pi$ into $-2\\pi t_c f_0$", fontsize=9)
    axes[4].grid(True, alpha=0.3)
    axes[4].set_xlabel(r"$q$")

    fig.suptitle(
        f"mode ({l},{m}) reference phase at $f_0$ from bare TEOBResumS "
        f"($\\Lambda$={args.lam:g}, $\\chi_1$={args.chi1:g}, $\\chi_2$={args.chi2:g}, "
        f"{args.initial_frequency_hz:g} Hz)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
