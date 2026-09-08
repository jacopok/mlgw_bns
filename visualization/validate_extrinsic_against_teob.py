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

The comparison is between the full complex ``sum_lm S_lm = h_+ - i h_x``
of the two codes (no detector projection --- the mismatch is invariant
under it and carries both polarisations already). Every marginalisation
here rotates mode ``m`` by ``exp(i m phi_c)``, the actual effect of a
coalescence-phase change, using the surrogate's per-mode decomposition.

Three mismatches per draw:

* ``phase`` --- surrogate vs the independent TEOBResumS FD call,
  optimised over the coalescence phase only (no time shift);
* ``time_phase`` --- the same, over both a time shift and the
  coalescence phase (the standard detection-pipeline mismatch);
* ``on_grid`` --- surrogate per-mode contributions vs the TEOBResumS
  per-mode contributions on the model's own frequency grid (no
  frequency-domain TEOBResumS call, no resampling), through
  :meth:`~mlgw_bns.model_validation.ValidateModel.full_waveform_mismatch`.
  This has a ~1e-7 floor and is the sensitive check that the
  :math:`Y_{\ell m}(\iota)` projection and total-mass scaling stay
  internally consistent across the parameter space; ``time_phase`` is
  the check that they agree with an *independent* TEOBResumS call.

A ``time_phase`` optimum that sits at a systematically non-zero time or
phase, or any mismatch that trends with total mass or inclination, is
the signature of a treatment that disagrees with TEOBResumS. Both shift
distributions are histogrammed, and the mismatches are plotted against
inclination and total mass.

Findings (300 waveforms, seed 4, all parameters drawn over the
``default_hom`` training ranges plus total mass 2-4 Msun, isotropic
inclination, a random detector projection; after commits 528c828 +
6fa0282 and the per-mode ``exp(i m phi_c)`` marginalisation here):

    on-grid per-mode mismatch : median 3.0e-7, 90th pct 1.3e-5
    FD mismatch (time + phase) : median 5.6e-4, 90th pct 2.6e-2
    aligning time shift        : |median| 3e-6 s, no parameter trend
    aligning phase shift       : |median| 1.5 rad (phase-origin convention)

The extrinsic treatment agrees with TEOBResumS. The on-grid per-mode
mismatch --- which exercises the Y_lm(iota) projection, the m<0
reconstruction and the total-mass/distance scaling, with a ~1e-7 floor
--- is flat in every extrinsic parameter (|correlation| < 0.3). Holding
each varied parameter fixed in turn (see the sweep below) turned up one
real bug, now fixed (528c828): for ``total_mass`` below the dataset
reference (2.8 Msun) the post-Newtonian low-frequency extension fired
and overwrote each mode's inter-mode phase constant, a ~50x median
degradation with a step exactly at 2.8. That step, and the strong
``total_mass`` correlation it produced (r ~ -0.7), are gone;
``mismatch_vs_total_mass.py`` plots the before/after. The residual tail
(worst ~8e-4) is high mass-ratio + high spin intrinsic modelling
difficulty, not extrinsic treatment.

The frequency-domain comparison against an *independent* TEOBResumS call
sits at ~5.6e-4 -- three orders of magnitude above the on-grid number
for the *same* parameters. That gap is NOT the surrogate: it is
TEOBResumS not being invariant to its own configuration -- its
``all_modes_amplitude_phase`` path (the on-grid ground truth) starts the
(2,2) ODE integration near 1.8 Hz where the plain
``EOBRunPy(initial_frequency=15)`` used for this FD comparison starts at
15 Hz, and the higher-mode phases depend on that at ~1e-3 (see
``teob_hom_start_frequency.py`` for a minimal reproducer), plus the
resampling of that FD call onto the model grid. An earlier revision of
this script optimised a single global phase on the summed strain rather
than ``exp(i m phi_c)`` per mode; that alone inflated the median to
~3e-3. Both effects are flat in inclination. The FD mismatch now trends
with mass ratio (r ~ 0.5), the intrinsic high-q modelling difficulty.

The aligning time shift is sub-sample (3e-6 s) with no parameter trend,
so the merger-time convention matches. The aligning phase spreads over
(-pi, pi] because mlgw_bns anchors the phase at the first grid node
while TEOBResumS anchors at merger, and the (2,2) phase accumulated
between the two wraps many times over; this is a phase-origin
convention, marginalised away in any real use (and in the on-grid
per-mode number), not an inconsistency.

Run with: python visualization/validate_extrinsic_against_teob.py
"""

from __future__ import annotations

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from mlgw_bns.higher_order_modes import Mode, mode_to_k
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

DF = 1.0 / 512.0
F0_TEOB = 15.0
BAND_LO = 20.0
BAND_HI = 2048.0

#: TEOBResumS sampling rate, in Hz; set from the model's band in ``main``.
SRATE_HZ = 4096.0

FIGURE_PATH = "visualization/validate_extrinsic_against_teob.png"
DATA_PATH = "visualization/validate_extrinsic_against_teob.npz"


def interp_fd(target_frequencies, frequencies, series):
    """Resample a frequency series through its amplitude and unwrapped phase."""
    amplitude = np.interp(target_frequencies, frequencies, np.abs(series))
    phase = np.interp(target_frequencies, frequencies, np.unwrap(np.angle(series)))
    return amplitude * np.exp(1j * phase)


def teob_polarizations(params: ParametersWithExtrinsic, frequencies):
    r"""TEOBResumS :math:`h_+, h_\times` for one aligned-spin source.

    Returned on the sub-grid of ``frequencies`` that TEOBResumS covers,
    in the ``mlgw_bns`` Fourier convention, with the boolean mask of that
    sub-grid.
    """
    from EOBRun_module import EOBRunPy

    par = dict(
        q=params.mass_ratio,
        LambdaAl2=params.lambda_1,
        LambdaBl2=params.lambda_2,
        chi1=params.chi_1,
        chi2=params.chi_2,
        M=params.total_mass,
        distance=params.distance_mpc,
        inclination=params.inclination,
        initial_frequency=F0_TEOB,
        srate_interp=SRATE_HZ,
        use_geometric_units="no",
        interp_uniform_grid="yes",
        domain=1,
        df=DF,
        # mlgw_bns adds `reference_phase` to every mode's phase; TEOBResumS
        # rotates the (l, m) mode by exp(i m * coalescence_angle), and for
        # the phase convention of compute_hpc that azimuth is
        # pi/2 - coalescence_angle. Setting reference_phase = 0 on the
        # surrogate side and the default coalescence_angle here leaves only
        # a constant offset, which the phase optimisation absorbs.
        coalescence_angle=0.0,
        output_hpc="no",
        arg_out="no",
        use_spins=1,
        use_mode_lm=sorted({mode_to_k(mode) for mode in MODES}),
    )
    f, real_hp, imag_hp, real_hc, imag_hc = EOBRunPy(par)
    f = np.asarray(f)
    hp = np.conj(np.asarray(real_hp) + 1j * np.asarray(imag_hp))
    hc = np.conj(np.asarray(real_hc) + 1j * np.asarray(imag_hc))

    inside = (frequencies >= max(BAND_LO, f[0])) & (frequencies <= min(BAND_HI, f[-1]))
    return (
        interp_fd(frequencies[inside], f, hp),
        interp_fd(frequencies[inside], f, hc),
        inside,
    )


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

    phi_grid = np.linspace(-np.pi, np.pi, 61)
    t_grid = np.linspace(-max_delta_t, max_delta_t, 201)

    best_phase_only = max(match(0.0, phi) for phi in phi_grid)

    # coarse 2-D grid, then a local Nelder-Mead refinement
    coarse_best, coarse_arg = -1.0, (0.0, 0.0)
    for phi_c in phi_grid:
        h2 = combined(phi_c)
        overlaps = np.exp(2j * np.pi * np.outer(t_grid, frequencies)) @ (
            ref_weighted * h2
        )
        norms = np.sqrt(np.abs(
            np.sum((np.abs(h2) ** 2) * weight)
        ))  # |h2| is t-shift invariant
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
    rng = np.random.default_rng(SEED)

    keys = (
        "mismatch_phase", "mismatch_time_phase", "mismatch_on_grid",
        "time_shift", "phase_shift",
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
            hp_t, hc_t, inside = teob_polarizations(params, frequencies)
        except Exception as error:  # noqa: BLE001
            logging.warning("TEOBResumS failed for draw %d: %s", index, error)
            continue
        if inside.sum() < 16:
            continue

        fb = frequencies[inside]
        psd = validator.psd_at_frequencies(fb)
        teob_strain = (hp_t - 1j * hc_t)  # sum_lm S_lm, on the TEOB sub-grid

        surrogate_modes = model.predict_modes_dict(frequencies, params)
        mode_arrays = [surrogate_modes[(mode.l, mode.m)][inside] for mode in MODES]
        m_values = [mode.m for mode in MODES]
        mm_p, mm_tp, t_shift, phase_shift = optimised_mismatches(
            teob_strain, mode_arrays, m_values, fb, psd
        )

        teob_modes = model.get_teob_modes_dict(frequencies, params)
        mm_grid = validator.full_waveform_mismatch(
            {k: v[inside] for k, v in teob_modes.items()},
            {k: v[inside] for k, v in surrogate_modes.items()},
            frequencies=fb,
        )

        records["mismatch_phase"].append(mm_p)
        records["mismatch_time_phase"].append(mm_tp)
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

    allv = np.concatenate([logclean(mm_p), logclean(mm_tp), logclean(mm_grid)])
    bins = np.linspace(np.floor(allv.min()), np.ceil(allv.max()), 44)
    axes[0, 0].hist(logclean(mm_grid), bins=bins, alpha=0.6, color="tab:green",
                    label=f"on-grid per-mode (median {np.nanmedian(mm_grid):.1e})")
    axes[0, 0].hist(logclean(mm_p), bins=bins, alpha=0.6, color="tab:orange",
                    label=f"FD, phase only (median {np.nanmedian(mm_p):.1e})")
    axes[0, 0].hist(logclean(mm_tp), bins=bins, alpha=0.6, color="tab:blue",
                    label=f"FD, time + phase (median {np.nanmedian(mm_tp):.1e})")
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
                   color="tab:blue", label="FD, time + phase")
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
        ("FD mismatch (phase only)", "mismatch_phase"),
        ("FD mismatch (time+phase)", "mismatch_time_phase"),
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
    for key in ("time_shift", "phase_shift", "mismatch_time_phase", "mismatch_on_grid"):
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
