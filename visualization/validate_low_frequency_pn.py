r"""Validate the surrogate's post-Newtonian low-frequency extension.

The ``default_hom`` frequency grid starts at
``effective_initial_frequency_hz`` :math:`\approx 3.57` Hz (the 5 Hz dataset
start scaled by :math:`m_{\min}/m_{\mathrm{ref}} = 2.0/2.8`). When a waveform
is requested below that --- i.e. ``f_requested * total_mass / 2.8 < 3.57`` ---
:meth:`ModeModel.predict_amplitude_phase_optimized` splices on a *per-mode*
TaylorF2 segment (:math:`(m/2)\,\phi_{22}(2f/m) + \arg H_{\ell m}`, 3.5PN with
tidal terms) below the connection, with a :func:`smoothing_func` amplitude
blend across :math:`[f_{\mathrm{conn}}/2, f_{\mathrm{conn}}]` and a constant
phase shift that glues the PN piece onto the bottom of the model band
(``mode_model.py`` ~L1542; the shift goes on the PN piece, not the band, so
the per-mode inter-mode phase constant survives --- see 528c828).

Every other validation script starts at 20 Hz and never exercises this path.
Here we request from ``F_LOW = 2`` Hz so the extension fires for every mass in
the box, and compare against:

* an *independent* ``EOBRunPy`` call started at ``F_LOW * 0.5`` Hz
  (:func:`initial_frequency_scaling` for :math:`m_{\max}=4`, so every mode has
  support at ``F_LOW``) --- the physical reference;
* the model's own pure TaylorF2 (:meth:`Model.get_taylorf2_modes_dict`) ---
  below the connection the spliced surrogate *is* this, up to the blend and
  the connecting shift, so their difference isolates the splice itself.

Memory: the ``EOBRunPy`` call passes ``interp_freqs="yes"`` onto a bounded
geometric grid (~6000 points), so peak RSS stays ~50 MB even from a 1 Hz ODE
start (measured); the blow-up only happens if the output grid is left
unbounded (uniform ``df`` from 1 Hz to ``srate/2``). Binaries are processed
one at a time with an explicit ``gc.collect()`` between them, and only
low-band summaries (not the full 6000-point arrays) are kept.

Outputs (``visualization/validate_low_frequency_pn.{png,npz}``):

* per-mode amplitude ratio :math:`A_{\mathrm{sur}}/A_{\mathrm{EOB}}` and phase
  residual :math:`\phi_{\mathrm{sur}} - \phi_{\mathrm{EOB}}` (per mode
  detrended by an :math:`a + b f` fit over [10, 500] Hz and extrapolated
  down), median and 10--90 band over binaries, on a common low-band grid;
* full-strain mismatch (per-mode :math:`e^{i m \phi_c}` + :math:`t_c`
  optimised) over ``[F_LOW, 2048]``, ``[F_LOW, f_conn]`` (pure PN),
  ``[f_conn, 2048]`` (model band), with the ET PSD and, for the low band,
  flat weighting;
* the low-band phase-residual RMS and mismatch vs total mass and mass ratio;
* a splice self-consistency check: ``arg(S_lm conj TF2_lm)`` is flat over the
  PN band to ~1e-9 (the spliced surrogate *is* the TaylorF2 phase plus one
  constant there).

Results (24 binaries, ``q in [1,3]``, ``M in [2.0, 3.5]``, from 2 Hz):

    mismatch vs independent EOB, PN band [2, 3.57] Hz : median 1.7e-7
    mismatch vs independent EOB, full [2, 2048] Hz    : median 1.1e-6
    per-mode phase residual RMS over [2, 20] Hz       : (2,2) 0.006 rad,
      (2,1) 0.004, (3,3) 0.018, (4,4) 0.039 (growing with m, PN accuracy)
    amplitude shape ratio vs EOB at 2 Hz             : within 2e-5 .. 5e-4
    peak RSS                                          : ~1.1 GB

The extension is accurate and joins the model band without a kink; the
residual grows toward low frequency (longer PN reach) and with total-mass
(more accumulated phase), the expected TaylorF2 truncation behaviour.

Run: python visualization/validate_low_frequency_pn.py [--n-waveforms N]
     [--f-low HZ]
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mlgw_bns
from mlgw_bns.higher_order_modes import Mode, initial_frequency_scaling, mode_to_k
from mlgw_bns.model import Model
from mlgw_bns.mode_model import ParametersWithExtrinsic
from mlgw_bns.special_func import spinsphericalharm

from validate_extrinsic_against_teob import optimised_mismatches

logging.basicConfig(level=logging.WARNING)

MODES = [Mode(2, 2), Mode(2, 1), Mode(3, 3), Mode(4, 4)]
M_VALUES = [m.m for m in MODES]

SEED = 7
N_WAVEFORMS = 24

Q_RANGE = (1.0, 3.0)
CHI_RANGE = (-0.5, 0.5)
LAMBDA_RANGE = (5.0, 5000.0)
#: kept <= 3.5 so the top of the band stays inside the trained region
#: (2048 * 3.5 / 2.8 = 2560 Hz < 2925 Hz) --- isolates the *low*-frequency
#: extension from the high-frequency zero-padding.
TOTAL_MASS_RANGE = (2.0, 3.5)

F_LOW = 2.0
F_HI = 2048.0
#: geometric spacing through the fast-winding low band (where the extension
#: lives), uniform 0.5 Hz above 60 Hz so the inter-mode beat is not
#: undersampled in the model-band cross-check.
_GRID_SPLIT_HZ = 60.0
_N_LOG = 1500
#: common grid for the stacked per-mode residual plots.
LOW_GRID = np.geomspace(2.05, 40.0, 240)
#: detrend the per-mode phase difference on a well-measured, well-modelled band
#: and extrapolate the fit downward, so the low-band residual is genuine PN
#: drift rather than an artefact of fitting through it.
FIT_BAND = (10.0, 500.0)
#: normalise each mode's amplitude to its value here before taking the
#: surrogate/EOB ratio --- the two use different absolute amplitude
#: conventions (physical strain vs raw ``hflm``), so only the *shape* is
#: comparable.
AMP_REF_HZ = 15.0

_MSUN_SECONDS = 4.925490947e-6


def et_psd_interpolator():
    """Linear interpolator over the full ET PSD file (1 Hz -- 10 kHz), so the
    weight is defined down to ``F_LOW`` (``ValidateModel`` clips its PSD to the
    model band and cannot be queried below ~3.6 Hz)."""
    import os.path as _p

    data = np.loadtxt(_p.join(_p.dirname(mlgw_bns.__file__), "data", "ET_psd.txt"))
    return lambda f: np.interp(f, data[:, 0], data[:, 1])


FIGURE_PATH = "visualization/validate_low_frequency_pn.png"
DATA_PATH = "visualization/validate_low_frequency_pn.npz"


def teob_low_modes(params, frequencies, srate_hz, inclination):
    r"""Per-mode :math:`h_{\ell m}(f)\,{}_{-2}Y_{\ell m}(\iota, 0)` from an
    independent ``EOBRunPy`` call started at ``F_LOW * initial_frequency_scaling``
    (1 Hz for :math:`m_{\max} = 4`), on the sub-grid of ``frequencies`` inside
    ``[F_LOW, F_HI]``. Same conventions as
    ``validate_extrinsic_against_teob.teob_independent_modes``.
    """
    from EOBRun_module import EOBRunPy

    inside = (frequencies >= F_LOW) & (frequencies <= F_HI)
    target = frequencies[inside]
    par = dict(
        q=params.mass_ratio, LambdaAl2=params.lambda_1, LambdaBl2=params.lambda_2,
        chi1=params.chi_1, chi2=params.chi_2, M=params.total_mass,
        distance=params.distance_mpc, inclination=inclination,
        initial_frequency=F_LOW * initial_frequency_scaling(MODES),
        srate_interp=srate_hz, use_geometric_units="no", domain=1,
        interp_freqs="yes", freqs=list(target), coalescence_angle=0.0,
        output_hpc="no", arg_out="yes", use_spins=1,
        use_mode_lm=sorted({mode_to_k(m) for m in MODES}),
    )
    f_spa, *_, hflm, htlm, dyn = EOBRunPy(par)
    f_spa = np.asarray(f_spa)
    tc = float(dyn["tc"]) if "tc" in dyn else float(np.asarray(htlm["t"])[-1])
    tc_s = tc * params.total_mass * _MSUN_SECONDS

    modes: dict[tuple[int, int], np.ndarray] = {}
    for mode in MODES:
        amp = np.asarray(hflm[str(mode_to_k(mode))][0])
        phase = np.asarray(hflm[str(mode_to_k(mode))][1]) - 2 * np.pi * tc_s * f_spa
        yr, yi = spinsphericalharm(-2, mode.l, mode.m, 0.0, inclination)
        modes[(mode.l, mode.m)] = amp * np.exp(-1j * phase) * (yr + 1j * yi)
    del hflm, htlm, dyn, f_spa
    return modes, inside


def detrended_phase_residual(sur_lm, ref_lm, freqs):
    r"""Per-mode phase difference minus an ``a + b f`` fit over ``FIT_BAND``.

    Uses the *difference* phase :math:`\arg(S_{\ell m}\,\bar R_{\ell m})`, which
    winds slowly (the two waveforms are close), rather than differencing two
    separately-unwrapped phases --- at a few Hz the (2,2) phase turns by many
    radians per grid bin and ``np.unwrap`` of each one is meaningless.
    The linear term absorbs the merger-time-convention offset between the
    surrogate and this EOB call; the constant absorbs the per-mode phase
    origin. What is left is genuine phase-shape disagreement.
    """
    diff = np.unwrap(np.angle(sur_lm * np.conj(ref_lm)))
    band = (freqs >= FIT_BAND[0]) & (freqs <= FIT_BAND[1])
    design = np.vstack([np.ones(band.sum()), freqs[band]]).T
    coef, *_ = np.linalg.lstsq(design, diff[band], rcond=None)
    return diff - (coef[0] + coef[1] * freqs)


def splice_vs_pn_residual(sur_lm, tf2_lm, freqs, f_conn):
    r"""``arg(S_{lm} conj(TF2_{lm}))`` minus an ``a + b f`` fit *over the PN
    band* (:math:`f \le f_{\mathrm{conn}}`). Below the connection the spliced
    surrogate is the TaylorF2 phase plus a constant and the surrogate's
    predicted merger-time shift (linear in ``f``, absent from
    ``get_taylorf2_modes_dict``), so removing both leaves ~1e-9; above the
    connection it shows where the trained surrogate departs from PN.
    """
    d = np.unwrap(np.angle(sur_lm * np.conj(tf2_lm)))
    pn = freqs <= f_conn
    design = np.vstack([np.ones(pn.sum()), freqs[pn]]).T
    coef, *_ = np.linalg.lstsq(design, d[pn], rcond=None)
    return d - (coef[0] + coef[1] * freqs)


def shape_amplitude_ratio(sur_lm, ref_lm, freqs):
    """``(A_sur / A_sur(AMP_REF)) / (A_ref / A_ref(AMP_REF))`` --- the two
    conventions differ by an overall scale, so compare frequency dependence
    only. 1.0 means the extension has the right amplitude *shape*."""
    a_s = np.abs(sur_lm)
    a_r = np.abs(ref_lm)
    i_ref = int(np.argmin(np.abs(freqs - AMP_REF_HZ)))
    return (a_s / a_s[i_ref]) / (a_r / a_r[i_ref])


def loginterp(x_new, x, y):
    return np.interp(np.log(x_new), np.log(x), y)


def validate(model, n_waveforms):
    psd_of = et_psd_interpolator()
    srate_hz = model.dataset.effective_srate_hz
    f_conn = model.dataset.effective_initial_frequency_hz
    grid = np.union1d(
        np.geomspace(F_LOW * 1.01, _GRID_SPLIT_HZ, _N_LOG),
        np.arange(_GRID_SPLIT_HZ, F_HI, 0.5),
    )
    rng = np.random.default_rng(SEED)

    bands = {
        "full [F_LOW,2048]": lambda f: (f >= F_LOW) & (f <= F_HI),
        "PN [F_LOW,f_conn]": lambda f: f <= f_conn,
        "model [f_conn,2048]": lambda f: f >= f_conn,
        "low [F_LOW,20]": lambda f: f <= 20.0,
    }

    rec: dict = {
        "mass_ratio": [], "total_mass": [], "chi_eff": [], "inclination": [],
        "amp_ratio": {m: [] for m in MODES},      # on LOW_GRID
        "phase_resid": {m: [] for m in MODES},     # on LOW_GRID, rad
        "phase_resid_pn": {m: [] for m in MODES},  # sur vs pure TaylorF2, on LOW_GRID
        "splice_pn_maxdev": {m: [] for m in MODES},  # rad, over PN band (should be ~0)
        "rms_low": {m: [] for m in MODES},         # rad, over [F_LOW,20]
        "mm_et": {b: [] for b in bands},
        "mm_flat_low": [],
    }

    for index in range(n_waveforms):
        q = rng.uniform(*Q_RANGE)
        chi_1 = rng.uniform(*CHI_RANGE)
        chi_2 = rng.uniform(*CHI_RANGE)
        total_mass = rng.uniform(*TOTAL_MASS_RANGE)
        inclination = np.arccos(rng.uniform(-1.0, 1.0))
        params = ParametersWithExtrinsic(
            mass_ratio=q, lambda_1=rng.uniform(*LAMBDA_RANGE),
            lambda_2=rng.uniform(*LAMBDA_RANGE), chi_1=chi_1, chi_2=chi_2,
            distance_mpc=100.0, inclination=inclination, total_mass=total_mass,
            reference_phase=0.0,
        )
        try:
            teob, inside = teob_low_modes(params, grid, srate_hz, inclination)
        except Exception as error:  # noqa: BLE001
            logging.warning("TEOBResumS failed for draw %d: %s", index, error)
            continue

        fb = grid[inside]
        sur = model.predict_modes_dict(grid, params)
        tf2 = model.get_taylorf2_modes_dict(grid, params)
        sur = {k: v[inside] for k, v in sur.items()}
        tf2 = {k: v[inside] for k, v in tf2.items()}

        psd = psd_of(fb)

        for mode in MODES:
            key = (mode.l, mode.m)
            ratio = shape_amplitude_ratio(sur[key], teob[key], fb)
            resid = detrended_phase_residual(sur[key], teob[key], fb)
            resid_pn = splice_vs_pn_residual(sur[key], tf2[key], fb, f_conn)
            rec["amp_ratio"][mode].append(loginterp(LOW_GRID, fb, ratio))
            rec["phase_resid"][mode].append(loginterp(LOW_GRID, fb, resid))
            rec["phase_resid_pn"][mode].append(loginterp(LOW_GRID, fb, resid_pn))
            rec["splice_pn_maxdev"][mode].append(
                float(np.max(np.abs(resid_pn[fb <= f_conn])))
            )
            low = fb <= 20.0
            rec["rms_low"][mode].append(float(np.sqrt(np.mean(resid[low] ** 2))))

        sur_arrays = [sur[(m.l, m.m)] for m in MODES]
        teob_strain = sum(teob.values())
        for label, mask_fn in bands.items():
            mask = mask_fn(fb)
            if mask.sum() < 16:
                rec["mm_et"][label].append(np.nan)
                continue
            _, mm_tp, _, _ = optimised_mismatches(
                teob_strain[mask], [a[mask] for a in sur_arrays], M_VALUES,
                fb[mask], psd[mask], max_delta_t=0.1,
            )
            rec["mm_et"][label].append(mm_tp)

        low = fb <= 20.0
        if low.sum() >= 16:
            _, mm_flat, _, _ = optimised_mismatches(
                teob_strain[low], [a[low] for a in sur_arrays], M_VALUES,
                fb[low], np.ones(low.sum()), max_delta_t=0.1,
            )
            rec["mm_flat_low"].append(mm_flat)
        else:
            rec["mm_flat_low"].append(np.nan)

        rec["mass_ratio"].append(q)
        rec["total_mass"].append(total_mass)
        rec["chi_eff"].append((q * chi_1 + chi_2) / (q + 1.0))
        rec["inclination"].append(inclination)

        print(
            f"  {index + 1}/{n_waveforms} q={q:.2f} M={total_mass:.2f}: "
            f"mm[full]={rec['mm_et']['full [F_LOW,2048]'][-1]:.2e} "
            f"mm[PN]={rec['mm_et']['PN [F_LOW,f_conn]'][-1]:.2e} "
            f"mm[flat<20]={rec['mm_flat_low'][-1]:.2e} "
            f"rms22<20={rec['rms_low'][Mode(2, 2)][-1]:.3f}rad",
            flush=True,
        )
        del teob, sur, tf2, sur_arrays, teob_strain
        gc.collect()

    return rec, f_conn


def summarise(rec, f_conn):
    print(f"\n  connection frequency f_conn = {f_conn:.3f} Hz")
    print(f"  {len(rec['mass_ratio'])} binaries, F_LOW = {F_LOW} Hz\n")

    print("  full-strain mismatch vs independent EOB (per-mode phi_c + t_c):")
    for label in rec["mm_et"]:
        v = np.array(rec["mm_et"][label])
        print(f"    {label:<22} median {np.nanmedian(v):.2e}  "
              f"90th {np.nanpercentile(v, 90):.2e}  max {np.nanmax(v):.2e}")
    v = np.array(rec["mm_flat_low"])
    print(f"    {'flat-weight [F_LOW,20]':<22} median {np.nanmedian(v):.2e}  "
          f"90th {np.nanpercentile(v, 90):.2e}  max {np.nanmax(v):.2e}")

    print("\n  per-mode phase-residual RMS over [F_LOW, 20] Hz (sur - EOB, rad):")
    for mode in MODES:
        v = np.array(rec["rms_low"][mode])
        print(f"    {str(mode):<14} median {np.nanmedian(v):.3f}  "
              f"90th {np.nanpercentile(v, 90):.3f}  max {np.nanmax(v):.3f}")

    print("\n  per-mode |amplitude shape ratio - 1| at F_LOW "
          "(sur vs EOB, normalised at 15 Hz; median over binaries):")
    for mode in MODES:
        stack = np.array(rec["amp_ratio"][mode])
        print(f"    {str(mode):<14} {np.nanmedian(np.abs(stack[:, 0] - 1.0)):.3e}")

    print("\n  splice self-consistency: max|arg(S_lm conj TF2_lm) - const| over "
          "the PN band (should be ~0):")
    for mode in MODES:
        v = np.array(rec["splice_pn_maxdev"][mode])
        print(f"    {str(mode):<14} median {np.nanmedian(v):.2e}  max {np.nanmax(v):.2e}")

    qs = np.array(rec["mass_ratio"])
    for key, arr in (("mm full", np.array(rec["mm_et"]["full [F_LOW,2048]"])),
                     ("mm flat<20", np.array(rec["mm_flat_low"])),
                     ("rms22<20", np.array(rec["rms_low"][Mode(2, 2)]))):
        good = np.isfinite(arr)
        rq = np.corrcoef(qs[good], np.log10(np.clip(arr[good], 1e-12, None)))[0, 1]
        rm = np.corrcoef(np.array(rec["total_mass"])[good],
                         np.log10(np.clip(arr[good], 1e-12, None)))[0, 1]
        print(f"    r({key}, log): mass_ratio {rq:+.2f}  total_mass {rm:+.2f}")


def plot(rec, f_conn):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    colors = {Mode(2, 2): "tab:blue", Mode(2, 1): "tab:orange",
              Mode(3, 3): "tab:green", Mode(4, 4): "tab:red"}

    for mode in MODES:
        resid = np.array(rec["phase_resid"][mode])
        lo, med, hi = np.nanpercentile(resid, [10, 50, 90], axis=0)
        axes[0, 0].plot(LOW_GRID, med, color=colors[mode], label=str(mode))
        axes[0, 0].fill_between(LOW_GRID, lo, hi, color=colors[mode], alpha=0.15)
        ratio = np.array(rec["amp_ratio"][mode])
        rlo, rmed, rhi = np.nanpercentile(ratio, [10, 50, 90], axis=0)
        axes[0, 1].plot(LOW_GRID, rmed, color=colors[mode], label=str(mode))
        axes[0, 1].fill_between(LOW_GRID, rlo, rhi, color=colors[mode], alpha=0.15)
        residpn = np.array(rec["phase_resid_pn"][mode])
        plo, pmed, phi_ = np.nanpercentile(residpn, [10, 50, 90], axis=0)
        axes[0, 2].plot(LOW_GRID, pmed, color=colors[mode], label=str(mode))
        axes[0, 2].fill_between(LOW_GRID, plo, phi_, color=colors[mode], alpha=0.15)

    for ax in (axes[0, 0], axes[0, 1], axes[0, 2]):
        ax.set_xscale("log")
        ax.axvline(f_conn, color="0.4", ls=":", lw=1)
        ax.set_xlabel("frequency [Hz]")
    axes[0, 0].set_ylabel(r"$\phi_{\rm sur} - \phi_{\rm EOB}$ (detrended) [rad]")
    axes[0, 0].set_title("per-mode phase residual vs EOB")
    axes[0, 0].legend()
    axes[0, 1].set_ylabel(r"$A_{\rm sur} / A_{\rm EOB}$")
    axes[0, 1].set_title("per-mode amplitude ratio vs EOB")
    axes[0, 1].axhline(1.0, color="0.4", ls="-", lw=0.6)
    axes[0, 2].set_ylabel(r"$\phi_{\rm sur} - \phi_{\rm TF2}$ [rad]")
    axes[0, 2].set_title("surrogate vs pure TaylorF2 (flat at 0 below f_conn "
                         "= splice is PN there)")
    axes[0, 2].axhline(0.0, color="0.4", ls="-", lw=0.6)

    ax = axes[1, 0]
    for label, style in (("full [F_LOW,2048]", "tab:blue"),
                         ("PN [F_LOW,f_conn]", "tab:red"),
                         ("model [f_conn,2048]", "tab:green")):
        v = np.array(rec["mm_et"][label])
        v = v[np.isfinite(v) & (v > 0)]
        ax.hist(np.log10(v), bins=20, alpha=0.55, color=style,
                label=f"{label} ({np.nanmedian(v):.1e})")
    ax.set_xlabel(r"$\log_{10}$ mismatch vs EOB (ET PSD)")
    ax.set_ylabel("count")
    ax.legend()
    ax.set_title("full-strain mismatch by band")

    qs = np.array(rec["mass_ratio"])
    ms = np.array(rec["total_mass"])
    mm_full = np.array(rec["mm_et"]["full [F_LOW,2048]"])
    mm_flat = np.array(rec["mm_flat_low"])
    axes[1, 1].scatter(ms, mm_full, c=qs, cmap="viridis", s=26)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel(r"total mass [$M_\odot$]")
    axes[1, 1].set_ylabel("mismatch [F_LOW,2048], ET PSD")
    axes[1, 1].set_title("colour = mass ratio")

    sc = axes[1, 2].scatter(qs, mm_flat, c=ms, cmap="plasma", s=26)
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_xlabel("mass ratio $q$")
    axes[1, 2].set_ylabel("flat-weight mismatch [F_LOW,20]")
    axes[1, 2].set_title(r"colour = total mass; pure-PN-band accuracy")
    fig.colorbar(sc, ax=axes[1, 2], label=r"$M$ [$M_\odot$]")

    fig.suptitle(
        f"Surrogate PN low-frequency extension vs independent EOB "
        f"({len(qs)} binaries, from {F_LOW} Hz)"
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    print(f"\n  figure written to {FIGURE_PATH}")


def main():
    global N_WAVEFORMS, F_LOW  # noqa: PLW0603
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-waveforms", type=int, default=N_WAVEFORMS)
    parser.add_argument("--f-low", type=float, default=F_LOW)
    args = parser.parse_args()
    N_WAVEFORMS = args.n_waveforms
    F_LOW = args.f_low

    model = Model(modes=MODES, filename="mlgw_bns/data/default_hom")
    model.load()

    rec, f_conn = validate(model, N_WAVEFORMS)
    summarise(rec, f_conn)
    plot(rec, f_conn)

    np.savez(
        DATA_PATH,
        f_conn=f_conn, f_low=F_LOW, low_grid=LOW_GRID,
        mass_ratio=rec["mass_ratio"], total_mass=rec["total_mass"],
        chi_eff=rec["chi_eff"], inclination=rec["inclination"],
        mm_flat_low=rec["mm_flat_low"],
        **{f"mm_{b}": rec["mm_et"][b] for b in rec["mm_et"]},
        **{f"amp_ratio_{m.l}{m.m}": np.array(rec["amp_ratio"][m]) for m in MODES},
        **{f"phase_resid_{m.l}{m.m}": np.array(rec["phase_resid"][m]) for m in MODES},
        **{f"phase_resid_pn_{m.l}{m.m}": np.array(rec["phase_resid_pn"][m]) for m in MODES},
        **{f"splice_pn_maxdev_{m.l}{m.m}": np.array(rec["splice_pn_maxdev"][m]) for m in MODES},
        **{f"rms_low_{m.l}{m.m}": np.array(rec["rms_low"][m]) for m in MODES},
    )
    print(f"  data written to {DATA_PATH}")


if __name__ == "__main__":
    main()
