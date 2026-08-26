r"""Mismatches for full precessing waveforms built on the surrogate.

:class:`~mlgw_bns.precessing_model.PrecessingModel` twists co-precessing
multipoles into the inertial frame along the PN spin-precession
dynamics. The twist itself is exact (a rotation), and the Euler angles
are the same whichever multipoles one feeds it, so comparing the
*surrogate*-twisted waveform against the *EOB*-twisted one for the same
binary isolates the surrogate's own reconstruction error --- which is
the question this script asks. It does not test the accuracy of the PN
angles against a precessing EOB evolution: TEOBResumS does not expose
its Euler angles, so that comparison is not available here.

What is randomized is the *orientation* of each binary relative to the
observer, which for a precessing source is not a triviality. For an
aligned-spin binary the inclination only reweights a fixed set of
multipoles, so a mismatch computed face-on says almost everything there
is to say. Under precession the multipoles themselves are mixed by the
Wigner-:math:`D` matrices, the relative weight of that mixing depends on
where the observer sits, and the detector then projects the two
polarizations onto a single strain,

.. math::
    h = F_+(\theta, \varphi, \psi) h_+ + F_\times(\theta, \varphi, \psi) h_\times \,,

so the surrogate error can be amplified or suppressed depending on the
line of sight. Each binary is therefore observed from many random
orientations: the inclination is drawn isotropically, as are the sky
position and the polarization angle. This is cheap, because the
expensive parts --- integrating the PN dynamics, generating the
co-precessing multipoles and twisting them --- depend only on the
intrinsic parameters and are done once per binary.

The mismatch marginalises over a time shift and an overall phase, as
:meth:`~mlgw_bns.model_validation.ValidateModel.full_waveform_mismatch`
does, weighted by the Einstein Telescope PSD.

The cost is dominated by the PN spin-precession integration, which
starts at the bottom of the model's band and so has to cover the whole
inspiral: about a minute per binary, and everything else is negligible
beside it. Expect the script to take of order half an hour.

Run with: python visualization/precessing_mismatches.py
"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model import Model
from mlgw_bns.model_validation import ValidateModel
from mlgw_bns.precessing_model import (
    PrecessingModel,
    PrecessingParametersWithExtrinsic,
    polarizations_from_inertial_modes,
)

logging.basicConfig(level=logging.WARNING)

MODES = [Mode(2, 2), Mode(2, 1), Mode(3, 3), Mode(4, 4)]

N_BINARIES = 20
N_ORIENTATIONS = 25

# largest in-plane spin magnitude for either body; the aligned component
# comes from the training distribution, which is what the surrogate
# knows about
MAX_IN_PLANE_SPIN = 0.1

DISTANCE_MPC = 100.0
TOTAL_MASS = 2.8

SEED = 17

FIGURE_PATH = "visualization/precessing_mismatches.png"


def antenna_patterns(theta, phi, psi):
    r"""Antenna patterns of a right-angled interferometer.

    Parameters
    ----------
    theta, phi : float
        Position of the source in the detector frame, in radians:
        polar angle from the normal to the detector plane, and azimuth
        from the first arm.
    psi : float
        Polarization angle, in radians.

    Returns
    -------
    tuple[float, float]
        :math:`(F_+, F_\times)`.
    """
    cos_theta = np.cos(theta)
    a = 0.5 * (1.0 + cos_theta**2) * np.cos(2.0 * phi)
    b = cos_theta * np.sin(2.0 * phi)

    f_plus = a * np.cos(2.0 * psi) - b * np.sin(2.0 * psi)
    f_cross = a * np.sin(2.0 * psi) + b * np.cos(2.0 * psi)
    return f_plus, f_cross


def random_orientation(rng):
    """Draw an isotropic inclination and sky position, and a polarization.

    Returns
    -------
    dict
        ``inclination``, ``azimuth``, ``theta``, ``phi``, ``psi``, all
        in radians.
    """
    return {
        "inclination": np.arccos(rng.uniform(-1.0, 1.0)),
        "azimuth": rng.uniform(0.0, 2.0 * np.pi),
        "theta": np.arccos(rng.uniform(-1.0, 1.0)),
        "phi": rng.uniform(0.0, 2.0 * np.pi),
        "psi": rng.uniform(0.0, np.pi),
    }


def random_spin_vector(rng, chi_z):
    """Add a random in-plane component to an aligned spin."""
    magnitude = rng.uniform(0.0, MAX_IN_PLANE_SPIN)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    return (magnitude * np.cos(angle), magnitude * np.sin(angle), chi_z)


def load_model() -> Model:
    """Load the pretrained model shipped with the package."""
    return Model.default_for_testing(modes=MODES)


def precessing_mismatches(model: Model):
    """Mismatch between the surrogate- and EOB-twisted precessing waveforms.

    Returns
    -------
    mismatches : np.ndarray
        One entry per (binary, orientation) pair.
    inclinations : np.ndarray
        The inclination each mismatch was computed at, in radians.
    opening_angles : np.ndarray
        The largest :math:`\\beta` reached by each binary, in radians:
        a measure of how strongly it precesses.
    """
    precessing = PrecessingModel(model)
    validator = ValidateModel(model.mode_models[Mode(2, 2)])
    frequencies = validator.frequencies

    rng = np.random.default_rng(SEED)
    parameter_generator = model.dataset.make_parameter_generator(SEED)

    mismatches = []
    inclinations = []
    opening_angles = []

    for index in range(N_BINARIES):
        intrinsic = next(parameter_generator)
        params = PrecessingParametersWithExtrinsic(
            mass_ratio=intrinsic.mass_ratio,
            lambda_1=intrinsic.lambda_1,
            lambda_2=intrinsic.lambda_2,
            chi_1=random_spin_vector(rng, intrinsic.chi_1),
            chi_2=random_spin_vector(rng, intrinsic.chi_2),
            distance_mpc=DISTANCE_MPC,
            inclination=0.0,
            total_mass=TOTAL_MASS,
        )

        # the angles and the twisted multipoles depend only on the
        # intrinsic parameters, so they are computed once per binary and
        # reused for every line of sight
        angles = precessing.euler_angles(params, float(frequencies[0]))
        predicted = precessing.predict_modes_dict(
            frequencies, params, source="surrogate", angles=angles
        )
        true = precessing.predict_modes_dict(
            frequencies, params, source="eob", angles=angles
        )

        for _ in range(N_ORIENTATIONS):
            orientation = random_orientation(rng)
            f_plus, f_cross = antenna_patterns(
                orientation["theta"], orientation["phi"], orientation["psi"]
            )

            strains = []
            for modes in (true, predicted):
                h_plus, h_cross = polarizations_from_inertial_modes(
                    *modes,
                    inclination=orientation["inclination"],
                    azimuth=orientation["azimuth"],
                )
                strains.append(f_plus * h_plus + f_cross * h_cross)

            # the EOB waveform is zero below the frequency at which it
            # starts; restrict the integral to where it is defined
            support = np.abs(strains[0]) > 0
            if support.sum() < 2:
                continue

            mismatches.append(
                validator.full_waveform_mismatch(
                    {(2, 2): strains[0][support]},
                    {(2, 2): strains[1][support]},
                    frequencies=frequencies[support],
                )
            )
            inclinations.append(orientation["inclination"])
            opening_angles.append(angles.beta.max())

        print(
            f"  binary {index + 1}/{N_BINARIES}: "
            f"median mismatch so far {np.median(mismatches):.3e}",
            flush=True,
        )

    return (
        np.array(mismatches),
        np.array(inclinations),
        np.array(opening_angles),
    )


def plot(mismatches, inclinations, opening_angles):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].hist(np.log10(mismatches), bins=40, color="tab:blue", alpha=0.8)
    axes[0].axvline(
        np.log10(np.median(mismatches)),
        c="black",
        ls="--",
        label=f"median = {np.median(mismatches):.1e}",
    )
    axes[0].set_xlabel(r"$\log_{10}$ mismatch")
    axes[0].set_ylabel("counts")
    axes[0].legend()

    axes[1].scatter(inclinations, mismatches, s=6, alpha=0.4)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"inclination $\iota$ [rad]")
    axes[1].set_ylabel("mismatch")

    axes[2].scatter(opening_angles, mismatches, s=6, alpha=0.4, c="tab:orange")
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"$\max_t \beta$ [rad]")
    axes[2].set_ylabel("mismatch")

    fig.suptitle(
        f"Precessing-waveform mismatches, surrogate vs EOB co-precessing "
        f"multipoles ({N_BINARIES} binaries "
        f"$\\times$ {N_ORIENTATIONS} orientations)"
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    print(f"figure written to {FIGURE_PATH}")


if __name__ == "__main__":
    mismatches, inclinations, opening_angles = precessing_mismatches(load_model())
    print(
        f"precessing mismatches: median {np.median(mismatches):.3e}, "
        f"90th percentile {np.percentile(mismatches, 90):.3e}, "
        f"worst {np.max(mismatches):.3e}"
    )
    np.savez(
        "visualization/precessing_mismatches.npz",
        mismatches=mismatches,
        inclinations=inclinations,
        opening_angles=opening_angles,
    )
    plot(mismatches, inclinations, opening_angles)
