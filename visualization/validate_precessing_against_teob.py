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

To attribute the mismatch it runs the comparison for two sources of the
co-precessing multipoles:

* ``"surrogate"`` --- the trained networks, twisted along the PN angles
  integrated here: the whole pipeline;
* ``"eob"`` --- TEOBResumS' own co-precessing multipoles (its
  frequency-domain ``hlm``, which carry no precession: TEOBResumS twists
  them internally and only the summed :math:`h_+, h_\times` see the
  precession), twisted along the *same* PN angles.

The ``"eob"`` run has the surrogate taken out of the loop entirely, so
whatever mismatch it still shows against TEOBResumS' :math:`h_+,
h_\times` is the cost of *modelling* the precession here --- the PN
spin-precession angles versus TEOBResumS' internal dynamics, the
stationary-phase co-precessing multipoles versus the Fourier transform
of the time-domain ones, and the resampling onto the model's grid. The
``"surrogate"`` run adds the network reconstruction error on top. If the
two are equal, the networks are not what limits a precessing waveform.

A third number, ``network``, is the mismatch between the two surrogate
sources directly, on the model grid with no TEOBResumS frequency-domain
call: the pure co-precessing reconstruction error, the quantity
:mod:`visualization.precessing_mismatches` measures.

TEOBResumS is called in the frequency domain (``domain = 1``) with
``df = 1 / 512`` s so the inspiral does not wrap, resampled onto the
model grid through amplitude and unwrapped phase (never the complex
strain, which aliases), and conjugated for its opposite Fourier-sign
convention. Its frequency-domain output is sky-projected, so it is
re-run for each inclination.

Findings (12 binaries x 6 orientations, total mass 2.8, |chi_perp| < 0.4,
seed 20). The ``"surrogate"`` and ``"eob"`` mismatches against
TEOBResumS' h+, hx agree to 0.1% of each other, observation by
observation: the trained networks add nothing measurable to a precessing
waveform. Their own co-precessing reconstruction error (``network``) is
~1e-6, worst 6e-5 -- the same as in the aligned-spin validation.

What a precessing waveform costs instead is the precession model, and it
climbs steeply with the opening angle beta:

    beta < 0.05 rad :  median 4e-3   (2.9e-4 - 1.3e-2)
    0.05 - 0.10     :  median 9e-3   (2.6e-3 - 5.7e-2)
    0.10 - 0.20     :  median 3e-2   (2.6e-2 - 2.0e-1)
    0.20 - 0.30     :  median 1.3e-1 (7.9e-2 - 2.6e-1)

The near-aligned floor (~4e-3, occasionally 3e-4) is the stationary-phase
co-precessing multipoles versus the Fourier transform of the time-domain
ones, plus the resampling of TEOBResumS' frequency-domain output onto the
model grid. On top of that, the mismatch is set by the PN
spin-precession angles integrated here versus TEOBResumS' internal
dynamics -- consistent with the per-cent-level beta error that
validate_twist_against_teob measures. The astrophysically expected BNS
range is beta <~ 0.1 rad, where a precessing waveform is good to ~1e-2;
by beta ~ 0.25 rad it is a poor match. None of this is the surrogate.

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
DF = 1.0 / 512.0
#: Where to start the TEOBResumS integration, in Hz --- below the
#: comparison band so its own low-frequency taper stays out of it.
F0_TEOB = 15.0
#: Comparison band, in Hz.
BAND_LO = 20.0
BAND_HI = 2048.0

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
        # PrecessingModel projects with exp(i m azimuth); TEOBResumS uses
        # coalescence_angle = pi/2 - azimuth (see twist_waveform.compute_hpc).
        coalescence_angle=np.pi / 2.0 - azimuth,
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
    hc = np.conj(np.asarray(real_hc) + 1j * np.asarray(imag_hc))

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
        "surrogate_mismatch",
        "eob_mismatch",
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

            hp_teob, hc_teob, inside = teob_polarizations(
                intrinsic, chi_1, chi_2, iota, azimuth, frequencies
            )
            strain_teob = f_plus * hp_teob + f_cross * hc_teob

            strains = {}
            for source in ("surrogate", "eob"):
                hp, hc = precessing.predict(
                    frequencies, precessing_params, source=source, angles=angles
                )
                strains[source] = (f_plus * hp + f_cross * hc)[inside]

            if inside.sum() < 2:
                surrogate_mismatch = eob_mismatch = network_mismatch = np.nan
            else:
                fb = frequencies[inside]
                surrogate_mismatch = validator.full_waveform_mismatch(
                    {(2, 2): strain_teob}, {(2, 2): strains["surrogate"]},
                    frequencies=fb,
                )
                eob_mismatch = validator.full_waveform_mismatch(
                    {(2, 2): strain_teob}, {(2, 2): strains["eob"]}, frequencies=fb,
                )
                # The network error barely depends on the line of sight and
                # is ~1e-6; one sample per binary is enough and the
                # optimisation is the expensive part of the loop.
                if orientation_index == 0:
                    network_mismatch = validator.full_waveform_mismatch(
                        {(2, 2): strains["eob"]}, {(2, 2): strains["surrogate"]},
                        frequencies=fb,
                    )
                else:
                    network_mismatch = np.nan

            records["surrogate_mismatch"].append(surrogate_mismatch)
            records["eob_mismatch"].append(eob_mismatch)
            records["network_mismatch"].append(network_mismatch)
            records["inclination"].append(iota)
            records["opening_angle"].append(float(angles.beta.max()))
            records["in_plane_spin"].append(in_plane)

        print(
            f"  binary {index + 1}/{n_binaries}: "
            f"max beta {angles.beta.max():.3f} rad, "
            f"in-plane spin {in_plane:.2f}, "
            f"surrogate median {np.nanmedian(records['surrogate_mismatch']):.2e}, "
            f"network median {np.nanmedian(records['network_mismatch']):.2e}  "
            f"({time.time() - start:.0f}s)",
            flush=True,
        )

    return {key: np.array(value) for key, value in records.items()}


def plot(records: dict) -> None:
    surrogate = records["surrogate_mismatch"]
    eob = records["eob_mismatch"]
    network = records["network_mismatch"]
    inclination = records["inclination"]
    opening_angle = records["opening_angle"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    def clean(values):
        return np.log10(values[np.isfinite(values) & (values > 0)])

    finite = np.concatenate([clean(surrogate), clean(eob), clean(network)])
    bins = np.linspace(np.floor(finite.min()), np.ceil(finite.max()), 44)
    axes[0, 0].hist(clean(network), bins=bins, color="tab:green", alpha=0.7,
                    label=f"network only (median {np.nanmedian(network):.1e})")
    axes[0, 0].hist(clean(eob), bins=bins, color="tab:orange", alpha=0.5,
                    label=f"EOB modes + PN twist (median {np.nanmedian(eob):.1e})")
    axes[0, 0].hist(clean(surrogate), bins=bins, color="tab:blue", alpha=0.5,
                    label=f"full surrogate (median {np.nanmedian(surrogate):.1e})")
    axes[0, 0].set_xlabel(r"$\log_{10}$ mismatch vs TEOBResumS $h_+, h_\times$")
    axes[0, 0].set_ylabel("count")
    axes[0, 0].legend()

    axes[0, 1].scatter(surrogate, eob, s=16, alpha=0.6, color="tab:blue")
    lims = [0.5 * min(surrogate.min(), eob.min()), 2 * max(surrogate.max(), eob.max())]
    axes[0, 1].plot(lims, lims, color="grey", lw=0.8, ls="--")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlim(lims)
    axes[0, 1].set_ylim(lims)
    axes[0, 1].set_xlabel("full surrogate mismatch")
    axes[0, 1].set_ylabel("EOB modes + PN twist mismatch")
    axes[0, 1].set_title("the networks add nothing on top of the PN twist")

    axes[1, 0].scatter(opening_angle, surrogate, s=16, alpha=0.6, color="tab:blue",
                       label="full surrogate vs TEOBResumS")
    net = np.isfinite(network)
    axes[1, 0].scatter(opening_angle[net], network[net], s=16, alpha=0.6,
                       color="tab:green", label="network only (vs EOB modes)")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel(r"$\max_t \beta$ [rad]  (precession strength)")
    axes[1, 0].set_ylabel("mismatch")
    axes[1, 0].legend()

    axes[1, 1].scatter(inclination, surrogate, s=16, alpha=0.6, color="tab:blue")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel(r"inclination $\iota$ [rad]")
    axes[1, 1].set_ylabel("full surrogate mismatch vs TEOBResumS")

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
        ("full surrogate", "surrogate_mismatch"),
        ("EOB modes + PN twist", "eob_mismatch"),
        ("network only", "network_mismatch"),
    ):
        values = records[key]
        print(
            f"{label:>22}: median {np.nanmedian(values):.3e}, "
            f"90th pct {np.nanpercentile(values, 90):.3e}, "
            f"worst {np.nanmax(values):.3e}"
        )
    excess = records["surrogate_mismatch"] - records["eob_mismatch"]
    print(
        f"{'surrogate - EOB twist':>22}: median {np.nanmedian(excess):+.2e} "
        "(the networks' contribution to a precessing waveform)"
    )

    np.savez(DATA_PATH, **records)
    plot(records)


if __name__ == "__main__":
    main()
