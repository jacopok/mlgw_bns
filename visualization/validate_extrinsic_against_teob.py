r"""Validate the surrogate's extrinsic-parameter handling against TEOBResumS.

The other model-validation scripts hold the extrinsic parameters fixed
(``visualization/validate_model.py`` uses one inclination and one total
mass) and vary only the intrinsic ones, and they take the ground truth
from the *same* TEOBResumS-backed generator the surrogate is trained on
--- so they never exercise the code that turns the trained co-precessing
multipoles into an observer-frame :math:`h_+, h_\times`: the
:math:`{}_{-2}Y_{\ell m}(\iota)` projection with its :math:`m < 0`
reconstruction, the total-mass rescaling of the frequency axis and the
amplitude, the distance scaling, the reference phase.

This script exercises exactly that, against an independent TEOBResumS
call: for random draws of *every* parameter --- intrinsic, plus
inclination, total mass, distance, reference phase and a random detector
projection --- it compares :meth:`~mlgw_bns.model.Model.predict` to the
:math:`h_+, h_\times` TEOBResumS itself returns (``domain = 1``) with the
same inclination and total mass set on its own side. If the two codes
treat the extrinsic parameters the same way, the mismatch stays at the
intrinsic-modelling floor and the time/phase shifts needed to align the
two waveforms stay near zero and uncorrelated with any parameter.

The comparison is between the full complex strain ``sum_lm S_lm``, of the
surrogate and of an independent TEOBResumS call (no detector projection
--- the mismatch is invariant under it). Both are optimised over a time
shift and a coalescence phase, the latter rotating mode ``m`` by
``exp(i m phi_c)`` (its actual effect), which needs the per-mode split
even though the metric is on the summed strain.

The independent TEOBResumS strain is reconstructed by projecting and
summing each ``hflm`` multipole (``arg_out="yes"``,
:math:`{}_{-2}Y_{\ell m}(\iota, 0)`) rather than taking the code's own
pre-summed ``h_+, h_\times``: on a coarse grid the summed output, which
oscillates at the inter-mode beat, is undersampled by TEOBResumS'
internal interpolation, while each multipole's amplitude and phase are
smooth. The two agree to ~1e-11 on a dense grid. The independent call
also lowers the ODE start by ``initial_frequency_scaling(MODES)`` (x0.5
here), matching ``all_modes_amplitude_phase`` --- see
``teob_independent_modes`` and the note below.

Three mismatches per draw:

* ``time_phase`` --- full strain over the whole band [20, 2048] Hz;
* ``inband`` --- full strain over [40, 2048] Hz (kept as a cross-check;
  now that the reference's ODE start is lowered, every mode has support
  across the full band and the two agree);
* ``on_grid`` --- surrogate per-mode contributions vs the TEOBResumS
  per-mode contributions on the model's own frequency grid, through
  :meth:`~mlgw_bns.model_validation.ValidateModel.full_waveform_mismatch`.
  This has a ~1e-7 floor and checks that the :math:`Y_{\ell m}(\iota)`
  projection and total-mass scaling stay internally consistent.

A ``time_phase`` optimum that sits at a systematically non-zero time or
phase, or any mismatch that trends with total mass or inclination, is
the signature of a treatment that disagrees with TEOBResumS. Both shift
distributions are histogrammed, and the mismatches are plotted against
inclination and total mass.

Findings (300 waveforms, seed 4, all parameters drawn over the
``default_hom`` training ranges plus total mass 2-4 Msun, isotropic
inclination; full-strain FD comparison on a dense 0.5 Hz grid, per-mode
``exp(i m phi_c)`` marginalisation):

    on-grid per-mode mismatch   : median 3.0e-7, 90th pct 1.3e-5, worst 2.9e-3
    FD full strain [20,2048] Hz : median 3.0e-7, 90th pct 1.1e-5, worst 3.1e-3
    FD full strain [40,2048] Hz : median 3.8e-7, 90th pct 1.5e-5, worst 3.4e-3
    aligning time shift         : |median| 3.6e-5 s (sub-sample)
    aligning phase shift        : spreads over (-pi, pi] (phase-origin conv.)

The extrinsic treatment agrees with TEOBResumS. All three mismatch
metrics sit at the same floor, distribution for distribution: the full
reconstructed strain compared against a fresh, independent EOB call
matches the surrogate's own per-mode-on-its-grid consistency check ---
same median, same tail, same (weak) parameter correlations (each tracks
inclination at r ~ 0.31 on log10, identically, from the Y_lm projection
sampling; |r| < 0.22 for mass ratio and total mass). The floor and its
~3e-3 tail (shared by the on-grid check) are intrinsic modelling
difficulty at high mass ratio + high spin, the same the fixed-extrinsic
scripts see; not extrinsic handling.

Two real problems were found and fixed along the way:

* ``total_mass`` below the dataset reference (2.8 Msun) fired the
  post-Newtonian low-frequency extension, which overwrote each mode's
  inter-mode phase constant --- a ~50x median degradation with a step
  exactly at 2.8, and a strong ``total_mass`` correlation (r ~ -0.7).
  Fixed by shifting the PN segment to the band rather than the band to
  the PN (528c828); ``mismatch_vs_total_mass.py`` plots the before/after.

* The "FD floor": for a long time the full-strain FD comparison sat at
  ~4e-4 median and grew with mass ratio (r ~ 0.5, worst ~0.5), while the
  on-grid per-mode check was at ~3e-7. This was **not** the surrogate. It
  was the reference: a bare ``EOBRunPy(initial_frequency = 15)`` call
  integrates the early inspiral from too high a frequency, and its
  higher-mode phasing drifts from the well-conditioned
  ``all_modes_amplitude_phase`` generator (which lowers the ODE start by
  ``initial_frequency_scaling``, x0.5 for ``m_max = 4``) by an amount
  that grows with mass ratio --- the "~1e-3 ODE-start sensitivity" of
  TEOBResumS. ``probe_teob_config_gap.py`` isolates it with no surrogate
  in the loop: the entire ~4e-4 gap is that one knob, dropping to ~3e-6
  once the reference's ODE start is matched (the ``srate_interp_scaling``
  x2 is irrelevant at these total masses). An earlier "band-edge HOM
  support" explanation (that a 15 Hz start lacks the (3,3)/(4,4) below
  ~30 Hz) was wrong: restricting to [40, 2048] Hz did not help, because
  the drift spans the whole band. Fixed here by lowering the reference's
  ODE start; the FD floor is now ~4e-7.

Getting a clean number also required reconstructing the reference strain
from the ``hflm`` multipoles rather than TEOBResumS' own pre-summed
``h_+, h_\times`` (undersampled at the inter-mode beat on a coarse grid),
a dense frequency grid, and a per-mode ``exp(i m phi_c)`` marginalisation
(a single global phase alone inflates the median to ~3e-3). The FD floor
is not resampling (``fd_grid_convergence.py``).

The aligning time shift is sub-sample (~3.6e-5 s) --- it carries a mild
``total_mass`` trend (r ~ 0.37) but stays ~7x below one sample at
4096 Hz. The
aligning phase spreads over (-pi, pi] because mlgw_bns anchors the phase
at the first grid node while TEOBResumS anchors at merger, and the (2,2)
phase accumulated between the two wraps many times over; a phase-origin
convention, marginalised away in any real use, not an inconsistency.

Run with: python visualization/validate_extrinsic_against_teob.py
"""

from __future__ import annotations

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from mlgw_bns.higher_order_modes import (
    Mode,
    initial_frequency_scaling,
    mode_to_k,
)
from mlgw_bns.model import Model
from mlgw_bns.model_validation import ValidateModel
from mlgw_bns.mode_model import ParametersWithExtrinsic

logging.basicConfig(level=logging.WARNING)

MODES = [Mode(2, 2), Mode(2, 1), Mode(3, 3), Mode(4, 4)]

N_WAVEFORMS = 300
SEED = 4

#: Intrinsic ranges (the training ranges of ``default_hom``).
Q_RANGE = (1.0, 3.0)
CHI_RANGE = (-0.5, 0.5)
LAMBDA_RANGE = (5.0, 5000.0)
#: Extrinsic ranges.
TOTAL_MASS_RANGE = (2.0, 4.0)
DISTANCE_RANGE = (40.0, 400.0)

F0_TEOB = 15.0
BAND_LO = 20.0
BAND_HI = 2048.0

#: TEOBResumS sampling rate, in Hz; set from the model's band in ``main``.
SRATE_HZ = 4096.0

FIGURE_PATH = "visualization/validate_extrinsic_against_teob.png"
DATA_PATH = "visualization/validate_extrinsic_against_teob.npz"


#: One solar mass in seconds, for the geometric -> SI merger time.
_MSUN_SECONDS = 4.925490947e-6


def teob_independent_modes(params: ParametersWithExtrinsic, frequencies):
    r"""Per-mode observer-frame contributions :math:`h_{\ell m}(f)\,
    {}_{-2}Y_{\ell m}(\iota, 0)` from an *independent* TEOBResumS call
    (``arg_out="yes"``), returned directly on the in-band sub-grid of
    ``frequencies`` (no resampling on our side).

    The ODE start is lowered by :func:`initial_frequency_scaling` of the
    requested mode set (x0.5 for :math:`m_{\max} = 4`), exactly as
    :meth:`~mlgw_bns.higher_order_modes.TEOBResumSModeGenerator.all_modes_amplitude_phase`
    does. This is *not* cosmetic: a bare ``initial_frequency = 15`` Hz call
    integrates the early inspiral from too high a frequency and its
    higher-mode phasing drifts from the well-conditioned generator by an
    amount that grows with mass ratio (``probe_teob_config_gap.py``: the
    whole ~4e-4 "FD floor" is this one knob, ~3e-6 once it is matched). It
    stays an *independent* call --- a fresh EOB evaluation, not the trained
    network.

    Comparing *per mode* -- rather than against the code's summed
    ``h_+, h_\times`` -- keeps the sparse decimated grid from undersampling
    the inter-mode beat: each mode's amplitude and phase are smooth, their
    sum on a coarse grid is not. Each ``hflm`` phase is re-referenced to
    the merger (``- 2 pi t_c f``). The remaining per-mode offset relative
    to ``mlgw_bns`` (which builds the modes at azimuth 0) is a constant
    ``exp(-i m pi/2)`` -- the ``pi/2 - coalescence_angle`` convention of
    TEOBResumS' ``compute_hpc`` -- absorbed by the coalescence-phase
    optimisation.
    """
    from EOBRun_module import EOBRunPy
    from mlgw_bns.special_func import spinsphericalharm

    inside = (frequencies >= BAND_LO) & (frequencies <= BAND_HI)
    target = frequencies[inside]

    par = dict(
        q=params.mass_ratio, LambdaAl2=params.lambda_1, LambdaBl2=params.lambda_2,
        chi1=params.chi_1, chi2=params.chi_2, M=params.total_mass,
        distance=params.distance_mpc, inclination=params.inclination,
        initial_frequency=F0_TEOB * initial_frequency_scaling(MODES),
        srate_interp=SRATE_HZ, use_geometric_units="no",
        domain=1, interp_freqs="yes", freqs=list(target), coalescence_angle=0.0,
        output_hpc="no", arg_out="yes", use_spins=1,
        use_mode_lm=sorted({mode_to_k(mode) for mode in MODES}),
    )
    f_spa, *_, hflm, htlm, dyn = EOBRunPy(par)
    f_spa = np.asarray(f_spa)
    tc = float(dyn["tc"]) if "tc" in dyn else float(np.asarray(htlm["t"])[-1])
    tc_s = tc * params.total_mass * _MSUN_SECONDS

    modes: dict[tuple[int, int], np.ndarray] = {}
    for mode in MODES:
        amp = np.asarray(hflm[str(mode_to_k(mode))][0])
        phase = np.asarray(hflm[str(mode_to_k(mode))][1]) - 2 * np.pi * tc_s * f_spa
        yr, yi = spinsphericalharm(-2, mode.l, mode.m, 0.0, params.inclination)
        # conjugate to match mlgw_bns' exp(-i phase) mode convention
        modes[(mode.l, mode.m)] = amp * np.exp(-1j * phase) * (yr + 1j * yi)
    return modes, inside


def optimised_mismatches(reference, mode_arrays, m_values, frequencies, psd,
                         max_delta_t=0.05):
    r"""Mismatch of ``reference`` against ``sum_k S_k exp(i(2 pi f t_c +
    m_k phi_c))`` where ``S_k = mode_arrays[k]`` has azimuthal index
    ``m_values[k]``, optimised over the coalescence phase ``phi_c`` (which
    rotates mode ``m`` by ``exp(i m phi_c)``, *not* the same phase for
    every mode) and, for ``mm_time_phase``, a time shift ``t_c``.

    Returns ``(mm_phase, mm_time_phase, t_shift, phase_shift)``.
    """
    weight = np.gradient(frequencies) / psd

    def product(a, b):
        return np.sum(np.conj(a) * b * weight)

    norm1 = np.sqrt(np.abs(product(reference, reference)))
    if norm1 <= 0:
        return np.nan, np.nan, np.nan, np.nan

    stack = np.asarray(mode_arrays)  # (n_modes, n_freq)
    m_values = np.asarray(m_values)[:, None]
    ref_weighted = np.conj(reference) * weight

    def combined(phi_c):
        return (stack * np.exp(1j * m_values * phi_c)).sum(axis=0)

    def match(t_c, phi_c):
        h2 = combined(phi_c) * np.exp(2j * np.pi * frequencies * t_c)
        n2 = np.sqrt(np.abs(product(h2, h2)))
        if n2 <= 0:
            return 0.0
        return np.abs(np.sum(ref_weighted * h2)) / (norm1 * n2)

    phi_grid = np.linspace(-np.pi, np.pi, 721)
    t_grid = np.linspace(-max_delta_t, max_delta_t, 4001)
    t_kernels = np.exp(2j * np.pi * np.outer(t_grid, frequencies))

    best_phase_only = max(match(0.0, phi) for phi in phi_grid)

    # exhaustive 2-D grid (t_c via the kernel bank, per phi_c), then a local
    # Nelder-Mead polish. The dense phi_c grid matters: the m > 2 modes make
    # the overlap sharp in phi_c, and a coarse grid leaves a ~1e-3 floor.
    coarse_best, coarse_arg = -1.0, (0.0, 0.0)
    for phi_c in phi_grid:
        h2 = combined(phi_c)
        overlaps = t_kernels @ (ref_weighted * h2)
        norms = np.sqrt(np.abs(np.sum((np.abs(h2) ** 2) * weight)))
        k = int(np.argmax(np.abs(overlaps)))
        value = np.abs(overlaps[k]) / (norm1 * norms)
        if value > coarse_best:
            coarse_best, coarse_arg = value, (t_grid[k], phi_c)

    result = minimize(
        lambda x: -match(x[0], x[1]), coarse_arg, method="Nelder-Mead",
        options={"xatol": 1e-9, "fatol": 1e-13, "maxiter": 800},
    )
    if -result.fun >= coarse_best:
        t_shift, phase_shift = float(result.x[0]), float(result.x[1])
        best = -result.fun
    else:
        t_shift, phase_shift = coarse_arg
        best = coarse_best

    return (
        1.0 - best_phase_only,
        1.0 - best,
        t_shift,
        float((phase_shift + np.pi) % (2 * np.pi) - np.pi),
    )


def validate(model: Model, n_waveforms: int) -> dict:
    validator = ValidateModel(model.mode_models[Mode(2, 2)])
    frequencies = validator.frequencies
    #: The FD strain comparison is done on a dense uniform grid, not the
    #: model's decimated one --- the summed strain oscillates at the
    #: inter-mode beat and needs finer sampling than any single mode.
    fd_frequencies = np.arange(BAND_LO, BAND_HI, 0.5)
    rng = np.random.default_rng(SEED)

    keys = (
        "mismatch_phase", "mismatch_time_phase", "mismatch_inband",
        "mismatch_on_grid", "time_shift", "phase_shift",
        "inclination", "total_mass", "mass_ratio", "chi_eff", "distance_mpc",
    )
    records: dict = {key: [] for key in keys}

    for index in range(n_waveforms):
        q = rng.uniform(*Q_RANGE)
        chi_1 = rng.uniform(*CHI_RANGE)
        chi_2 = rng.uniform(*CHI_RANGE)
        total_mass = rng.uniform(*TOTAL_MASS_RANGE)
        inclination = np.arccos(rng.uniform(-1.0, 1.0))
        params = ParametersWithExtrinsic(
            mass_ratio=q,
            lambda_1=rng.uniform(*LAMBDA_RANGE),
            lambda_2=rng.uniform(*LAMBDA_RANGE),
            chi_1=chi_1,
            chi_2=chi_2,
            distance_mpc=rng.uniform(*DISTANCE_RANGE),
            inclination=inclination,
            total_mass=total_mass,
            reference_phase=0.0,
        )
        try:
            teob_modes_ind, inside = teob_independent_modes(params, fd_frequencies)
        except Exception as error:  # noqa: BLE001
            logging.warning("TEOBResumS failed for draw %d: %s", index, error)
            continue
        if inside.sum() < 16:
            continue

        fb = fd_frequencies[inside]
        psd = validator.psd_at_frequencies(fb)
        teob_strain = sum(teob_modes_ind.values())  # full strain, per-mode sourced

        surrogate_fd = model.predict_modes_dict(fd_frequencies, params)
        mode_arrays = [surrogate_fd[(mode.l, mode.m)][inside] for mode in MODES]
        m_values = [mode.m for mode in MODES]
        mm_p, mm_tp, t_shift, phase_shift = optimised_mismatches(
            teob_strain, mode_arrays, m_values, fb, psd
        )

        # Cross-check on [40, 2048] Hz. With the reference's ODE start
        # lowered by initial_frequency_scaling every mode now has support
        # across the whole band, so this should track ``mm_tp``; kept as a
        # guard against a band-edge regression.
        band = fb >= 40.0
        if band.sum() >= 16:
            _, mm_inband, _, _ = optimised_mismatches(
                teob_strain[band], [a[band] for a in mode_arrays],
                m_values, fb[band], psd[band],
            )
        else:
            mm_inband = np.nan

        surrogate_modes = model.predict_modes_dict(frequencies, params)
        og = (frequencies >= BAND_LO) & (frequencies <= BAND_HI)
        teob_modes = model.get_teob_modes_dict(frequencies, params)
        mm_grid = validator.full_waveform_mismatch(
            {k: v[og] for k, v in teob_modes.items()},
            {k: v[og] for k, v in surrogate_modes.items()},
            frequencies=frequencies[og],
        )

        records["mismatch_phase"].append(mm_p)
        records["mismatch_time_phase"].append(mm_tp)
        records["mismatch_inband"].append(mm_inband)
        records["mismatch_on_grid"].append(mm_grid)
        records["time_shift"].append(t_shift)
        records["phase_shift"].append(phase_shift)
        records["inclination"].append(inclination)
        records["total_mass"].append(total_mass)
        records["mass_ratio"].append(q)
        records["chi_eff"].append((q * chi_1 + chi_2) / (q + 1.0))
        records["distance_mpc"].append(params.distance_mpc)

        if (index + 1) % 25 == 0:
            print(
                f"  {index + 1}/{n_waveforms}: "
                f"mm(phase) median {np.nanmedian(records['mismatch_phase']):.2e}, "
                f"mm(t,phase) median {np.nanmedian(records['mismatch_time_phase']):.2e}, "
                f"|t| median {np.nanmedian(np.abs(records['time_shift'])):.2e} s",
                flush=True,
            )

    return {key: np.array(value) for key, value in records.items()}


def plot(records: dict) -> None:
    mm_p = records["mismatch_phase"]
    mm_tp = records["mismatch_time_phase"]
    mm_grid = records["mismatch_on_grid"]
    finite = np.isfinite(mm_tp) & (mm_tp > 0)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    def logclean(values):
        v = values[np.isfinite(values) & (values > 0)]
        return np.log10(v)

    mm_ib = records["mismatch_inband"]
    allv = np.concatenate([logclean(mm_ib), logclean(mm_tp), logclean(mm_grid)])
    bins = np.linspace(np.floor(allv.min()), np.ceil(allv.max()), 44)
    axes[0, 0].hist(logclean(mm_grid), bins=bins, alpha=0.6, color="tab:green",
                    label=f"on-grid per-mode (median {np.nanmedian(mm_grid):.1e})")
    axes[0, 0].hist(logclean(mm_tp), bins=bins, alpha=0.6, color="tab:blue",
                    label=f"FD [20,2048], t+phase (median {np.nanmedian(mm_tp):.1e})")
    axes[0, 0].hist(logclean(mm_ib), bins=bins, alpha=0.6, color="tab:red",
                    label=f"FD [40,2048], t+phase (median {np.nanmedian(mm_ib):.1e})")
    axes[0, 0].set_xlabel(r"$\log_{10}$ mismatch vs TEOBResumS")
    axes[0, 0].set_ylabel("count")
    axes[0, 0].legend()

    axes[0, 1].hist(records["time_shift"][finite] * 1e3, bins=40, color="tab:blue")
    axes[0, 1].set_xlabel("recovered time shift [ms]")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].set_title(
        f"median {np.nanmedian(records['time_shift']) * 1e3:.2f} ms, "
        f"std {np.nanstd(records['time_shift']) * 1e3:.2f} ms"
    )

    axes[0, 2].hist(records["phase_shift"][finite], bins=40, color="tab:blue")
    axes[0, 2].set_xlabel("recovered phase shift [rad]")
    axes[0, 2].set_ylabel("count")
    axes[0, 2].set_title(
        f"median {np.nanmedian(records['phase_shift']):.2f} rad, "
        f"std {np.nanstd(records['phase_shift']):.2f} rad"
    )

    for ax, key, label in (
        (axes[1, 0], "inclination", r"inclination $\iota$ [rad]"),
        (axes[1, 1], "total_mass", r"total mass [$M_\odot$]"),
        (axes[1, 2], "mass_ratio", "mass ratio $q$"),
    ):
        ax.scatter(records[key][finite], mm_tp[finite], s=14, alpha=0.6,
                   color="tab:blue", label="FD [20,2048]")
        ax.scatter(records[key][finite], mm_ib[finite], s=14, alpha=0.6,
                   color="tab:red", label="FD [40,2048]")
        ax.scatter(records[key][finite], mm_grid[finite], s=14, alpha=0.6,
                   color="tab:green", label="on-grid per-mode")
        ax.set_yscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("mismatch")
        if ax is axes[1, 0]:
            ax.legend()

    fig.suptitle(
        f"Surrogate vs TEOBResumS over all extrinsic parameters "
        f"({finite.sum()} waveforms)"
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    print(f"figure written to {FIGURE_PATH}")


def main() -> None:
    global N_WAVEFORMS, SRATE_HZ  # noqa: PLW0603

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-waveforms", type=int, default=N_WAVEFORMS)
    args = parser.parse_args()
    N_WAVEFORMS = args.n_waveforms

    model = Model(modes=MODES, filename="mlgw_bns/data/default_hom")
    model.load()
    SRATE_HZ = model.dataset.effective_srate_hz

    records = validate(model, N_WAVEFORMS)

    print()
    for label, key in (
        ("FD [20,2048] (time+phase)", "mismatch_time_phase"),
        ("FD [40,2048] (time+phase)", "mismatch_inband"),
        ("on-grid per-mode mismatch", "mismatch_on_grid"),
    ):
        v = records[key]
        print(f"{label:>27}: median {np.nanmedian(v):.3e}, "
              f"90th pct {np.nanpercentile(v, 90):.3e}, worst {np.nanmax(v):.3e}")
    for label, key in (("time shift [s]", "time_shift"), ("phase shift [rad]", "phase_shift")):
        v = records[key]
        print(f"{label:>27}: median {np.nanmedian(v):+.3e}, "
              f"|median| {np.nanmedian(np.abs(v)):.3e}, std {np.nanstd(v):.3e}")

    # a treatment mismatch shows up as an aligning shift, or a mismatch,
    # that trends with an extrinsic parameter -- a non-zero correlation
    finite = np.isfinite(records["mismatch_time_phase"])
    for key in ("time_shift", "phase_shift", "mismatch_time_phase",
                "mismatch_inband", "mismatch_on_grid"):
        values = records[key][finite]
        if key.startswith("mismatch"):
            values = np.log10(np.clip(values, 1e-12, None))
        print(f"\n  {key} correlations:")
        for par_key in ("inclination", "total_mass", "mass_ratio", "chi_eff"):
            r = np.corrcoef(records[par_key][finite], values)[0, 1]
            print(f"    vs {par_key:<12}: r = {r:+.3f}")

    np.savez(DATA_PATH, **records)
    plot(records)


if __name__ == "__main__":
    main()
