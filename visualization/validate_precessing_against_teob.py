r"""End-to-end validation of the precessing surrogate against TEOBResumS.

The two other precessing scripts each test one half of the pipeline in
isolation:

* :mod:`visualization.validate_twist_against_teob` checks the twist and
  the PN Euler angles against TEOBResumS' own inertial-frame multipoles,
  in the time domain, by scaling the in-plane spins to zero;
* :mod:`visualization.precessing_mismatches` checks the surrogate's
  co-precessing multipoles by twisting *both* the surrogate and an EOB
  reference with the *same* angles, so the angles cancel out.

This script closes the loop: it compares the full frequency-domain
polarizations of
:meth:`~mlgw_bns.precessing_model.PrecessingModel.predict` against the
:math:`h_+, h_\times` that TEOBResumS returns for the same precessing
binary, over random orientations.

It tracks three mismatches per observation:

* ``reanchored`` --- the shipping pipeline:
  :meth:`~mlgw_bns.precessing_model.PrecessingModel.predict` with the
  Euler angles re-tabulated against the surrogate's own (2,2) phase (see
  :meth:`~mlgw_bns.precessing_model.EulerAngles.reanchored`);
* ``plain`` --- the same with ``reanchor=False``, i.e. the angles looked
  up against the 3.5PN orbital-frequency track the integration marches
  with. The gap between the two is what the re-anchoring buys;
* ``network`` --- ``reanchored`` against the same pipeline fed
  TEOBResumS' own co-precessing multipoles instead of the surrogate's,
  on the model grid with no frequency-domain TEOBResumS call: the pure
  co-precessing reconstruction error, the quantity
  :mod:`visualization.precessing_mismatches` measures. Sampled once per
  binary (it barely depends on the line of sight).

TEOBResumS is called in the frequency domain (``domain = 1``) with
``df = 1 / 2048`` Hz so the inspiral does not wrap, resampled onto the
model grid through amplitude and unwrapped phase (never the complex
strain, which aliases), and conjugated for its opposite Fourier-sign
convention. Its frequency-domain output is sky-projected, so it is
re-run for each inclination.

Findings (12 binaries x 4 orientations, total mass 2.8, |chi_perp| < 0.4,
seed 20; 2 binaries skipped where TEOBResumS' root finder fails), after
the fixes of 2026-09 -- the twist's rotation (D*, not D), the spins given
at TEOBResumS' own reference point, the LO tidal term in the PN
dOmega/dt, the mode-subset reference-phase bug of Model, and this
script's band edge and azimuth mapping:

                       plain      reanchored
    all             :  2.0e-3     1.4e-2
    beta 0.05-0.10  :  2.5e-3     3.1e-3
    beta 0.10-0.20  :  1.6e-3     5.9e-3
    beta 0.20-0.50  :  2.1e-3     3.0e-2
    network         :  1.1e-5 (worst 1.3e-4)

(before them: 1.7e-2 plain, 8.1e-3 reanchored). The plain lookup is what
TEOBResumS itself does -- its twist reads the PN-integrated angles against
the PN orbital frequency, in both domains -- so re-anchoring moves *away*
from it, and ``reanchor`` is now off by default.

What is left does not grow with beta and does not depend on the surrogate
(``network``; precession_error_vs_inplane_spin.py shows it already there
for aligned spins), but on inclination: ~2e-4 face-on, ~4e-3 edge-on. It
is the co-precessing orbital phase at the reference frequency. Our
multipoles and those TEOBResumS twists agree mode by mode only up to a
rotation exp(i n phi0), phi0 binary-dependent; applying it brings the
precessing mismatch to the aligned-spin floor (4e-6, 1.7e-4 on two
binaries at chi_perp 0.35). The SPA orbital phase read off our (2,2) at
the reference frequency is ~0 (as for an orbit started there), that of
TEOBResumS' multipoles is not: its convention for the initial orbital
phase of the co-precessing multipoles (plausibly tied to its initial
Euler angles, alpha_0 = gamma_0) is not reproduced yet. Without
precession the same rotation is an observer rotation, which the
single-phase marginalisation absorbs only for the (2, 2).

The earlier explanations of the floor -- the stationary-phase twist, the
co-precessing approximation, the N4LO PN equations, the orbital-frequency
evolution -- were ruled out along the way: our twist is the same formula
as TEOBResumS' twist_hlm_FD (compare_fd_twist_with_teob_formula.py), and
with the reference point and tidal term fixed our angles match its own
(``output_dynamics``) to 2e-4 rad from 15 Hz to 1.5 kHz.

Run with: python visualization/validate_precessing_against_teob.py
"""

from __future__ import annotations

import argparse
import logging
import time

import matplotlib.pyplot as plt
import numpy as np

from mlgw_bns.higher_order_modes import Mode, mode_to_k
from mlgw_bns.model import Model
from mlgw_bns.model_validation import ValidateModel
from mlgw_bns.precessing_model import (
    PrecessingModel,
    PrecessingParametersWithExtrinsic,
)

logging.basicConfig(level=logging.WARNING)

MODES = [Mode(2, 2), Mode(2, 1), Mode(3, 3), Mode(4, 4)]

N_BINARIES = 12
N_ORIENTATIONS = 6

#: Largest in-plane spin magnitude for either body. Neutron-star spins in
#: a BNS are small; the range is pushed a little past the astrophysical
#: expectation on purpose, to see where the precession model breaks down.
MAX_IN_PLANE_SPIN = 0.4

TOTAL_MASS = 2.8
DISTANCE_MPC = 100.0

#: TEOBResumS frequency-domain resolution, in Hz. Fine enough that
#: ``1 / DF`` exceeds the full inspiral length from ``F0_TEOB``.
DF = 1.0 / 2048.0
#: Where to start the TEOBResumS integration, in Hz --- low enough that
#: every multipole is present throughout the comparison band. TEOBResumS'
#: (l, m) multipole is identically zero below (m / 2) * 0.95 * F0_TEOB
#: (see SPIN_REFERENCE_FREQUENCY_HZ for the 0.95): starting at 15 Hz, as
#: this script used to, left the reference with no (4, 4) below 28.5 Hz
#: and no (3, 3) below 21.4 Hz, against a surrogate that has both --- a
#: ~1e-2 mismatch on its own for edge-on views, where those multipoles
#: weigh most.
F0_TEOB = 10.0
#: Comparison band, in Hz.
BAND_LO = 20.0
BAND_HI = 2048.0
#: Frequency at which PrecessingModel takes the spin vectors to be given.
#: In the frequency domain (``domain = 1``) TEOBResumS starts both its EOB
#: and its PN spin dynamics at 0.95 * initial_frequency, and imposes the
#: input spins (and Lhat = z) *there* -- read off its own ``dynspin.txt``
#: (``output_dynamics``); in the time domain it is initial_frequency
#: itself. ``None`` is PrecessingModel's own default, the start of its PN
#: integration, a few Hz. Either of those is a different binary as far as
#: the precession goes, by an amount that grows with the opening angle.
SPIN_REFERENCE_FREQUENCY_HZ = 0.95 * F0_TEOB

SEED = 20

#: TEOBResumS sampling rate, in Hz; set from the model's band in ``main``.
SRATE_HZ = 4096.0

FIGURE_PATH = "visualization/validate_precessing_against_teob.png"
DATA_PATH = "visualization/validate_precessing_against_teob.npz"


def antenna_patterns(theta: float, phi: float, psi: float) -> tuple[float, float]:
    r"""Antenna patterns :math:`(F_+, F_\times)` of an L-shaped detector."""
    cos_theta = np.cos(theta)
    a = 0.5 * (1.0 + cos_theta**2) * np.cos(2.0 * phi)
    b = cos_theta * np.sin(2.0 * phi)
    f_plus = a * np.cos(2.0 * psi) - b * np.sin(2.0 * psi)
    f_cross = a * np.sin(2.0 * psi) + b * np.cos(2.0 * psi)
    return f_plus, f_cross


def random_orientation(rng: np.random.Generator) -> dict:
    """Isotropic inclination and sky position, uniform azimuth and polarization."""
    return {
        "inclination": np.arccos(rng.uniform(-1.0, 1.0)),
        "azimuth": rng.uniform(0.0, 2.0 * np.pi),
        "theta": np.arccos(rng.uniform(-1.0, 1.0)),
        "phi": rng.uniform(0.0, 2.0 * np.pi),
        "psi": rng.uniform(0.0, np.pi),
    }


def random_spin_vector(rng: np.random.Generator, chi_z: float) -> tuple:
    """An aligned spin with a random in-plane component bolted on."""
    magnitude = rng.uniform(0.0, MAX_IN_PLANE_SPIN)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    return (magnitude * np.cos(angle), magnitude * np.sin(angle), chi_z)


def interp_fd(target_frequencies, frequencies, series):
    """Resample a frequency series through its amplitude and unwrapped phase."""
    amplitude = np.interp(target_frequencies, frequencies, np.abs(series))
    phase = np.interp(target_frequencies, frequencies, np.unwrap(np.angle(series)))
    return amplitude * np.exp(1j * phase)


def teob_polarizations(params, chi_1, chi_2, inclination, azimuth, frequencies):
    r"""TEOBResumS :math:`h_+, h_\times` for one binary and line of sight.

    Returned on the sub-grid of ``frequencies`` that TEOBResumS covers,
    in the ``mlgw_bns`` Fourier convention (``h_+ - i h_\times`` is the
    multipole sum), together with the boolean mask of that sub-grid.
    """
    from EOBRun_module import EOBRunPy

    par = dict(
        q=params.mass_ratio,
        LambdaAl2=params.lambda_1,
        LambdaBl2=params.lambda_2,
        M=TOTAL_MASS,
        distance=DISTANCE_MPC,
        initial_frequency=F0_TEOB,
        srate_interp=SRATE_HZ,
        use_geometric_units="no",
        interp_uniform_grid="yes",
        domain=1,
        df=DF,
        inclination=inclination,
        # PrecessingModel at azimuth phi is TEOBResumS at coalescence_angle
        # pi/2 + phi with h_x negated -- the mirror image of the pi/2 - phi
        # of twist_waveform.compute_hpc, which the multipoles' conjugate
        # Fourier convention introduces. Invisible without precession
        # (an aligned binary is symmetric under it), unambiguous with it:
        # checked against the Fourier transform of TEOBResumS' time-domain
        # h+, hx (1e-3 at beta ~ 0.36, against 0.16 with pi/2 - phi).
        coalescence_angle=np.pi / 2.0 + azimuth,
        output_hpc="no",
        arg_out="no",
        use_spins=2,
        chi1x=chi_1[0], chi1y=chi_1[1], chi1z=chi_1[2],
        chi2x=chi_2[0], chi2y=chi_2[1], chi2z=chi_2[2],
        use_mode_lm=sorted({mode_to_k(mode) for mode in MODES}),
    )
    f, real_hp, imag_hp, real_hc, imag_hc = EOBRunPy(par)
    f = np.asarray(f)
    hp = np.conj(np.asarray(real_hp) + 1j * np.asarray(imag_hp))
    hc = -np.conj(np.asarray(real_hc) + 1j * np.asarray(imag_hc))

    inside = (frequencies >= max(BAND_LO, f[0])) & (frequencies <= min(BAND_HI, f[-1]))
    hp_out = interp_fd(frequencies[inside], f, hp)
    hc_out = interp_fd(frequencies[inside], f, hc)
    return hp_out, hc_out, inside


def validate(model: Model, n_binaries: int, n_orientations: int):
    """Run the comparison and return the per-observation records."""
    precessing = PrecessingModel(model)
    validator = ValidateModel(model.mode_models[Mode(2, 2)])
    frequencies = validator.frequencies

    rng = np.random.default_rng(SEED)
    parameter_generator = model.dataset.make_parameter_generator(SEED)

    keys = (
        "reanchored_mismatch",
        "plain_mismatch",
        "network_mismatch",
        "inclination",
        "opening_angle",
        "in_plane_spin",
    )
    records: dict = {key: [] for key in keys}

    for index in range(n_binaries):
        intrinsic = next(parameter_generator)
        chi_1 = random_spin_vector(rng, intrinsic.chi_1)
        chi_2 = random_spin_vector(rng, intrinsic.chi_2)
        in_plane = max(np.hypot(*chi_1[:2]), np.hypot(*chi_2[:2]))

        precessing_params = PrecessingParametersWithExtrinsic(
            mass_ratio=intrinsic.mass_ratio,
            lambda_1=intrinsic.lambda_1,
            lambda_2=intrinsic.lambda_2,
            chi_1=chi_1,
            chi_2=chi_2,
            distance_mpc=DISTANCE_MPC,
            inclination=0.0,
            total_mass=TOTAL_MASS,
            # the spins are drawn as TEOBResumS' own, at its initial_frequency
            reference_frequency_hz=SPIN_REFERENCE_FREQUENCY_HZ,
        )
        angles = precessing.euler_angles(precessing_params, float(frequencies[0]))

        start = time.time()
        for orientation_index in range(n_orientations):
            orientation = random_orientation(rng)
            iota, azimuth = orientation["inclination"], orientation["azimuth"]
            f_plus, f_cross = antenna_patterns(
                orientation["theta"], orientation["phi"], orientation["psi"]
            )

            precessing_params.inclination = iota
            precessing_params.azimuth = azimuth

            try:
                hp_teob, hc_teob, inside = teob_polarizations(
                    intrinsic, chi_1, chi_2, iota, azimuth, frequencies
                )
            except RuntimeError as error:  # TEOBResumS' root finder, occasionally
                print(f"  binary {index + 1}: TEOBResumS failed ({error}), skipped")
                break
            strain_teob = f_plus * hp_teob + f_cross * hc_teob

            strains = {}
            for name, kwargs in (
                ("reanchored", dict(source="surrogate", reanchor=True)),
                ("plain", dict(source="surrogate", reanchor=False)),
            ):
                hp, hc = precessing.predict(
                    frequencies, precessing_params, angles=angles, **kwargs
                )
                strains[name] = (f_plus * hp + f_cross * hc)[inside]

            if inside.sum() < 2:
                reanchored_mismatch = plain_mismatch = network_mismatch = np.nan
            else:
                fb = frequencies[inside]
                reanchored_mismatch = validator.full_waveform_mismatch(
                    {(2, 2): strain_teob}, {(2, 2): strains["reanchored"]},
                    frequencies=fb,
                )
                plain_mismatch = validator.full_waveform_mismatch(
                    {(2, 2): strain_teob}, {(2, 2): strains["plain"]}, frequencies=fb,
                )
                # The network error barely depends on the line of sight;
                # one sample per binary is enough and the optimisation is
                # the expensive part of the loop.
                if orientation_index == 0:
                    hp_e, hc_e = precessing.predict(
                        frequencies, precessing_params, source="eob",
                        angles=angles, reanchor=False,
                    )
                    strain_eob = (f_plus * hp_e + f_cross * hc_e)[inside]
                    network_mismatch = validator.full_waveform_mismatch(
                        {(2, 2): strain_eob}, {(2, 2): strains["plain"]},
                        frequencies=fb,
                    )
                else:
                    network_mismatch = np.nan

            records["reanchored_mismatch"].append(reanchored_mismatch)
            records["plain_mismatch"].append(plain_mismatch)
            records["network_mismatch"].append(network_mismatch)
            records["inclination"].append(iota)
            records["opening_angle"].append(float(angles.beta.max()))
            records["in_plane_spin"].append(in_plane)

        print(
            f"  binary {index + 1}/{n_binaries}: "
            f"max beta {angles.beta.max():.3f} rad, "
            f"in-plane spin {in_plane:.2f}, "
            f"reanchored median {np.nanmedian(records['reanchored_mismatch']):.2e}, "
            f"plain median {np.nanmedian(records['plain_mismatch']):.2e}, "
            f"network median {np.nanmedian(records['network_mismatch']):.2e}  "
            f"({time.time() - start:.0f}s)",
            flush=True,
        )

    return {key: np.array(value) for key, value in records.items()}


def plot(records: dict) -> None:
    reanchored = records["reanchored_mismatch"]
    plain = records["plain_mismatch"]
    network = records["network_mismatch"]
    inclination = records["inclination"]
    opening_angle = records["opening_angle"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    def clean(values):
        return np.log10(values[np.isfinite(values) & (values > 0)])

    finite = np.concatenate([clean(reanchored), clean(plain), clean(network)])
    bins = np.linspace(np.floor(finite.min()), np.ceil(finite.max()), 44)
    axes[0, 0].hist(clean(network), bins=bins, color="tab:green", alpha=0.7,
                    label=f"network only (median {np.nanmedian(network):.1e})")
    axes[0, 0].hist(clean(plain), bins=bins, color="tab:orange", alpha=0.5,
                    label=f"plain PN angles (median {np.nanmedian(plain):.1e})")
    axes[0, 0].hist(clean(reanchored), bins=bins, color="tab:blue", alpha=0.5,
                    label=f"re-anchored (median {np.nanmedian(reanchored):.1e})")
    axes[0, 0].set_xlabel(r"$\log_{10}$ mismatch vs TEOBResumS $h_+, h_\times$")
    axes[0, 0].set_ylabel("count")
    axes[0, 0].legend()

    axes[0, 1].scatter(plain, reanchored, s=16, alpha=0.6, color="tab:blue")
    lims = [0.5 * min(plain.min(), reanchored.min()),
            2 * max(plain.max(), reanchored.max())]
    axes[0, 1].plot(lims, lims, color="grey", lw=0.8, ls="--")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlim(lims)
    axes[0, 1].set_ylim(lims)
    axes[0, 1].set_xlabel("plain PN-angle mismatch")
    axes[0, 1].set_ylabel("re-anchored mismatch")
    axes[0, 1].set_title("re-anchoring to the (2,2) phase, per observation")

    axes[1, 0].scatter(opening_angle, plain, s=16, alpha=0.5, color="tab:orange",
                       label="plain PN angles")
    axes[1, 0].scatter(opening_angle, reanchored, s=16, alpha=0.6, color="tab:blue",
                       label="re-anchored")
    net = np.isfinite(network)
    axes[1, 0].scatter(opening_angle[net], network[net], s=16, alpha=0.6,
                       color="tab:green", label="network only (vs EOB modes)")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel(r"$\max_t \beta$ [rad]  (precession strength)")
    axes[1, 0].set_ylabel("mismatch")
    axes[1, 0].legend()

    axes[1, 1].scatter(inclination, reanchored, s=16, alpha=0.6, color="tab:blue")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel(r"inclination $\iota$ [rad]")
    axes[1, 1].set_ylabel("re-anchored mismatch vs TEOBResumS")

    fig.suptitle(
        "Precessing surrogate vs TEOBResumS "
        f"({N_BINARIES} binaries $\\times$ {N_ORIENTATIONS} orientations, "
        f"$|\\chi_\\perp| < {MAX_IN_PLANE_SPIN}$)"
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    print(f"figure written to {FIGURE_PATH}")


def main() -> None:
    global N_BINARIES, N_ORIENTATIONS, SRATE_HZ  # noqa: PLW0603

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-binaries", type=int, default=N_BINARIES)
    parser.add_argument("--n-orientations", type=int, default=N_ORIENTATIONS)
    args = parser.parse_args()
    N_BINARIES = args.n_binaries
    N_ORIENTATIONS = args.n_orientations

    model = Model.default_for_testing(modes=MODES)
    SRATE_HZ = model.dataset.effective_srate_hz

    records = validate(model, N_BINARIES, N_ORIENTATIONS)

    print()
    for label, key in (
        ("re-anchored", "reanchored_mismatch"),
        ("plain PN angles", "plain_mismatch"),
        ("network only", "network_mismatch"),
    ):
        values = records[key]
        print(
            f"{label:>18}: median {np.nanmedian(values):.3e}, "
            f"90th pct {np.nanpercentile(values, 90):.3e}, "
            f"worst {np.nanmax(values):.3e}"
        )
    ratio = records["plain_mismatch"] / records["reanchored_mismatch"]
    print(
        f"{'re-anchoring gain':>18}: median x{np.nanmedian(ratio):.2f}"
    )

    np.savez(DATA_PATH, **records)
    plot(records)


if __name__ == "__main__":
    main()
