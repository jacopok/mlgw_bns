r"""Validate the twisting code against TEOBResumS' own precessing waveform.

:mod:`~mlgw_bns.twist_waveform` is a reimplementation, in Python, of two
things TEOBResumS does internally: the PN spin-precession evolution that
produces the Euler angles :math:`\alpha(t)`, :math:`\beta(t)`,
:math:`\gamma(t)`, and the rotation ("twist") that carries the
co-precessing multipoles into the inertial frame. This script checks
that reimplementation end to end, against the C code itself.

The check exploits the fact that TEOBResumS, with its default
``project_spins = 0``, evolves the *same* co-precessing dynamics whether
or not the spins have in-plane components: the aligned-spin run with
:math:`\chi_{1z}, \chi_{2z}` and the precessing run with the full spin
vectors share the orbital dynamics exactly (the script asserts it). So
one can

1. run TEOBResumS in generic-spin mode and keep its inertial-frame
   multipoles :math:`h^{\rm I}_{\ell m}` --- the ground truth;
2. run it again in aligned-spin mode with only the :math:`z` components,
   and keep its multipoles as the co-precessing ones;
3. integrate the PN spin precession here, twist the multipoles from (2)
   with :func:`~mlgw_bns.twist_waveform.twist_modes`, and compare against
   (1), multipole by multipole and then in the polarizations.

Both runs are interpolated onto the same uniform time grid, so the
comparison is pointwise. The Euler angles are attached to the
co-precessing dynamics through :math:`M\Omega_{\rm orb}`, which is how
TEOBResumS itself associates the PN precession to the EOB evolution ---
and demonstrably so: matching the two evolutions by time instead
degrades the agreement by four orders of magnitude.

Agreement is not expected to be exact. The angles come from an
independent ODE integration, with its own tolerances and its own
interpolation onto the waveform grid, and a per-cent-level difference in
:math:`\beta` is enough to account for what is left. What the test is
really sensitive to is a wrong *convention* --- a sign, a transposed
Wigner index, a swapped :math:`\alpha \leftrightarrow \gamma` --- which
would show up as an :math:`\mathcal{O}(1)` discrepancy rather than a
small one. Two things are therefore checked beyond the raw numbers:

* the errors must vanish as the in-plane spins are turned off, since the
  twist then degenerates to the identity;
* they must vanish *at the right rate*: the leading mixing induced by
  the twist is first order in the rotation, so an error in the angles
  shows up linearly in :math:`\beta` in every multipole. What separates
  them is the prefactor --- :math:`(2, 2)` is contaminated only through
  the small co-precessing :math:`(2, 1)`, and comes out roughly fifty
  times cleaner than :math:`(2, 1)` itself, which is contaminated by the
  dominant multipole.

Where the residual actually lives is then settled directly, by
inverting: at each time the three Euler angles are fitted so that the
twist implemented here reproduces TEOBResumS' multipoles as well as it
can. The fit succeeds to machine precision, which says that TEOBResumS'
inertial multipoles are exactly a rigid rotation of the aligned-spin
ones and that the rotation is the one implemented here --- leaving
nothing but the angles --- and it hands back the angles the C code used,
to compare with the ones integrated here in radians rather than through
their effect on a waveform. Loosening the tolerances of the integration
done here by four orders of magnitude then changes nothing, so what is
left is a property of the spin-precession system itself, not of how
accurately it is solved here.

Normalisation matters when reading the output. A multipole that only
exists because of precession, such as :math:`(\ell, 0)`, is suppressed
by powers of :math:`\beta`, so the same absolute error looks alarming
against its own maximum and negligible against the dominant multipole's.
Both are printed.

Run with: python visualization/validate_twist_against_teob.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from mlgw_bns.higher_order_modes import Mode, mode_to_k
from mlgw_bns.precessing_model import newtonian_time_to_merger
from mlgw_bns.twist_waveform import (
    amp_phase_to_complex,
    compute_hpc,
    integrate_pn_spin_precession,
    twist_modes,
)

MASS_RATIO = 1.2
LAMBDA_1 = 300.0
LAMBDA_2 = 300.0

CHI_1 = (0.05, 0.0, 0.1)
CHI_2 = (0.0, -0.05, 0.05)

TOTAL_MASS = 2.8
# initial (2,2) frequency in geometric units, M f
INITIAL_FREQUENCY = 0.002
# spacing of the uniform time grid both runs are interpolated onto, in M
DELTA_T = 1.0

INCLINATION = 0.5
# azimuth of the line of sight; TEOBResumS parametrises the same freedom
# as pi/2 - coalescence_angle, so its default of zero is pi/2 --- see
# mlgw_bns.twist_waveform.compute_hpc
AZIMUTH = np.pi / 2.0

MODES = [Mode(2, 1), Mode(2, 2), Mode(3, 3), Mode(4, 4)]

# factors the in-plane spins are scaled down by, to check that the
# disagreement vanishes with the opening angle and at the right rate
CONVERGENCE_FACTORS = (1.0, 0.5, 0.25, 0.125)

# one in every ANGLE_RECOVERY_STRIDE samples is used when solving for the
# Euler angles TEOBResumS actually used; they vary on the precession
# timescale, so there is nothing to gain from a fine grid
ANGLE_RECOVERY_STRIDE = 200

FIGURE_PATH = "visualization/twist_validation.png"


def scaled_spins(factor):
    """The spins of the reference binary, with the in-plane parts scaled."""
    chi_1, chi_2 = np.array(CHI_1), np.array(CHI_2)
    chi_1[:2] *= factor
    chi_2[:2] *= factor
    return chi_1, chi_2


def teob_parameters(**overrides):
    """Common TEOBResumS parameters, in geometric units."""
    parameters = {
        "q": MASS_RATIO,
        "LambdaAl2": LAMBDA_1,
        "LambdaBl2": LAMBDA_2,
        "M": TOTAL_MASS,
        "distance": 1.0,
        "initial_frequency": INITIAL_FREQUENCY,
        "use_geometric_units": "yes",
        "interp_uniform_grid": "yes",
        "dt_interp": DELTA_T,
        "domain": 0,  # time domain
        "inclination": INCLINATION,
        "output_hpc": "no",
        "arg_out": "yes",
        "ode_tmax": 1e12,
        "use_mode_lm": sorted({mode_to_k(mode) for mode in MODES}),
    }
    parameters.update(overrides)
    return parameters


def teob_runs(chi_1, chi_2):
    """Run TEOBResumS twice: with the full spins, and with only their z parts.

    Returns
    -------
    times : np.ndarray
        Uniform time grid, in units of the total mass, shared by both runs.
    inertial : dict[tuple[int, int], np.ndarray]
        Inertial-frame multipoles of the precessing run, keyed by
        :math:`(\\ell, m)` with all signs of :math:`m` present.
    coprecessing : dict[tuple[int, int], np.ndarray]
        Multipoles of the aligned-spin run, which are the co-precessing
        ones of the precessing binary; only :math:`m > 0`.
    polarizations : np.ndarray
        :math:`h_+ - i h_\\times` of the precessing run.
    dynamics : dict
        The co-precessing dynamics, as returned by TEOBResumS.
    """
    from EOBRun_module import EOBRunPy

    times, h_plus, h_cross, teob_inertial, dynamics = EOBRunPy(
        teob_parameters(
            use_spins=2,
            chi1x=chi_1[0],
            chi1y=chi_1[1],
            chi1z=chi_1[2],
            chi2x=chi_2[0],
            chi2y=chi_2[1],
            chi2z=chi_2[2],
        )
    )
    aligned_times, _, _, teob_aligned, aligned_dynamics = EOBRunPy(
        teob_parameters(use_spins=1, chi1=chi_1[2], chi2=chi_2[2])
    )

    # the premise of the whole comparison: with project_spins = 0 the
    # co-precessing dynamics does not know about the in-plane spins
    assert np.array_equal(times, aligned_times)
    np.testing.assert_allclose(
        dynamics["MOmega_orb"], aligned_dynamics["MOmega_orb"], rtol=1e-12
    )

    inertial = {}
    for mode in MODES:
        key = str(mode_to_k(mode))
        inertial[(mode.l, mode.m)] = amp_phase_to_complex(*teob_inertial[key])
        inertial[(mode.l, -mode.m)] = amp_phase_to_complex(*teob_inertial["-" + key])
        inertial[(mode.l, 0)] = amp_phase_to_complex(*teob_inertial[f"{mode.l}0"])

    coprecessing = {
        (mode.l, mode.m): amp_phase_to_complex(*teob_aligned[str(mode_to_k(mode))])
        for mode in MODES
    }

    return times, inertial, coprecessing, h_plus - 1j * h_cross, dynamics


def euler_angles_on_grid(times, dynamics, chi_1, chi_2, rtol=1e-11):
    r"""Euler angles of this binary, on the waveform's time grid.

    The PN precession is integrated as a function of its own time
    variable, and attached to the EOB evolution through the shared
    :math:`M\Omega_{\rm orb}`: for every point of the co-precessing
    dynamics the angles are read off at the matching orbital frequency,
    and then resampled onto the uniform waveform grid.
    """
    eta = MASS_RATIO / (1.0 + MASS_RATIO) ** 2

    solution = integrate_pn_spin_precession(
        eta,
        chi_1,
        chi_2,
        INITIAL_FREQUENCY,
        t_max=2.0 * newtonian_time_to_merger(eta, np.pi * INITIAL_FREQUENCY),
        rtol=rtol,
    )

    angles = []
    for name in ("alpha", "beta", "gamma"):
        on_dynamics = np.interp(
            dynamics["MOmega_orb"], solution["Momega"], np.unwrap(solution[name])
        )
        angles.append(np.interp(times, dynamics["t"], on_dynamics))

    return tuple(angles)


def twisted_modes(coprecessing, alpha, beta, gamma):
    """Twist the co-precessing multipoles with the code under test."""
    positive, negative, zero = twist_modes(
        coprecessing,
        alpha,
        beta,
        gamma,
        # the m = 0 multipole is produced automatically for each ell
        lm_inertial=[(mode.l, mode.m) for mode in MODES],
    )
    return {**positive, **negative, **zero}


def twist_here(chi_1, chi_2):
    """The whole chain, for one choice of spins.

    Returns the time grid, our twisted multipoles and polarizations,
    TEOBResumS' own, and the opening angle.
    """
    times, inertial, coprecessing, polarizations, dynamics = teob_runs(chi_1, chi_2)
    alpha, beta, gamma = euler_angles_on_grid(times, dynamics, chi_1, chi_2)

    ours = twisted_modes(coprecessing, alpha, beta, gamma)
    our_polarizations = compute_hpc(
        {key: value for key, value in ours.items() if key[1] > 0},
        {key: value for key, value in ours.items() if key[1] < 0},
        {key: value for key, value in ours.items() if key[1] == 0},
        AZIMUTH,
        INCLINATION,
    )
    return (
        times,
        ours,
        inertial,
        our_polarizations,
        polarizations,
        (alpha, beta, gamma),
        coprecessing,
        dynamics,
    )


def relative_error(ours, theirs, scale=None):
    r"""Pointwise :math:`|h - h^{\rm TEOB}|`, divided by `scale`.

    `scale` defaults to :math:`\max |h^{\rm TEOB}|` for the multipole at
    hand; pass the dominant multipole's amplitude for the comparison
    that is usually the meaningful one.
    """
    if scale is None:
        scale = np.max(np.abs(theirs))
    return np.abs(ours - theirs) / scale


def report(times, ours, theirs, our_polarizations, polarizations, angles):
    beta = angles[1]
    dominant = np.max(np.abs(theirs[(2, 2)]))

    print(f"grid: {len(times)} points, {times[-1]:.3g} M long")
    print(f"max opening angle: beta = {np.max(beta):.3e} rad\n")
    print("multipole    max |dh| / max|h_lm|    max |dh| / max|h_22|")
    for key in sorted(theirs):
        own = np.max(relative_error(ours[key], theirs[key]))
        scaled = np.max(relative_error(ours[key], theirs[key], scale=dominant))
        print(f"  ({key[0]}, {key[1]:2d})       {own:.3e}              {scaled:.3e}")

    error = relative_error(our_polarizations, polarizations)
    print(f"\npolarizations  {np.max(error):.3e} (max), {np.mean(error):.3e} (mean)")


def convergence_check():
    r"""Turn the in-plane spins off, and watch the disagreement vanish.

    The twist becomes the identity as :math:`\beta \to 0`, so any
    residual must go with it, and linearly: the mixing between
    multipoles is first order in the rotation. The two multipoles
    printed differ in what they are contaminated *by* --- :math:`(2, 1)`
    by the dominant multipole, :math:`(2, 2)` only by the much smaller
    co-precessing :math:`(2, 1)` --- which sets the prefactor, not the
    slope.
    """
    print("\nin-plane spins scaled down:\n")
    print("factor   max beta      (2,1) / h22    (2,2) / h22")

    results = []
    for factor in CONVERGENCE_FACTORS:
        _, ours, theirs, _, _, (_, beta, _), _, _ = twist_here(*scaled_spins(factor))
        dominant = np.max(np.abs(theirs[(2, 2)]))
        errors = [
            np.max(relative_error(ours[key], theirs[key], scale=dominant))
            for key in ((2, 1), (2, 2))
        ]
        results.append((factor, np.max(beta), *errors))
        print(
            f"{factor:6.3f}   {np.max(beta):.3e}     "
            f"{errors[0]:.3e}      {errors[1]:.3e}"
        )

    results = np.array(results)
    for name, column in (("(2,1)", 2), ("(2,2)", 3)):
        slope = np.polyfit(np.log(results[:, 1]), np.log(results[:, column]), 1)[0]
        print(f"  {name} error ~ beta^{slope:.2f}")

    return results


def recover_euler_angles(times, coprecessing, inertial, guess, stride=ANGLE_RECOVERY_STRIDE):
    r"""Solve for the Euler angles TEOBResumS actually used.

    The twist is a rotation, so at every instant the inertial multipoles
    are a three-parameter function of the co-precessing ones. Rather than
    compare waveforms, then, one can invert: fit
    :math:`(\alpha, \beta, \gamma)` at each time so that the twist
    implemented here reproduces TEOBResumS' multipoles as closely as it
    can, and look at what is left.

    This splits the disagreement in two. If the fit residual comes out at
    machine precision, TEOBResumS' inertial multipoles *are* a rigid
    rotation of the aligned-spin ones and the rotation is the one
    implemented here --- no convention, sign or index error survives, and
    everything that remains is in the angles. The recovered angles can
    then be compared to the ones integrated here directly, in radians.

    One caveat on reading that comparison. The multipoles depend on
    :math:`\alpha - \gamma` at leading order and on :math:`\alpha +
    \gamma` only through the mixing induced by the rotation, suppressed
    by :math:`\beta`. The first combination is therefore pinned down
    sharply and the second only loosely, and a discrepancy in
    :math:`\beta` is partly absorbed into :math:`\alpha + \gamma` by the
    fit --- which is why its recovered offset grows as :math:`\beta`
    shrinks. Read :math:`\beta` and :math:`\alpha - \gamma`.

    Returns
    -------
    indices : np.ndarray
        Positions in `times` at which the angles were recovered.
    recovered : np.ndarray, shape (n, 3)
        The fitted ``(alpha, beta, gamma)``, unwrapped.
    residuals : np.ndarray
        Largest residual of each fit, relative to the amplitude of the
        dominant multipole.
    """
    keys = sorted(inertial)
    dominant = np.max(np.abs(inertial[(2, 2)]))
    indices = np.arange(0, len(times), stride)

    recovered = np.zeros((len(indices), 3))
    residuals = np.zeros(len(indices))
    # walk along the inspiral, warm-starting each fit from the last
    current = np.array([angle[indices[0]] for angle in guess])

    for position, index in enumerate(indices):
        here = {key: np.array([value[index]]) for key, value in coprecessing.items()}
        target = np.concatenate(
            [[inertial[key][index].real, inertial[key][index].imag] for key in keys]
        )

        def residual(angles):
            twisted = twisted_modes(here, *(np.array([angle]) for angle in angles))
            ours = np.concatenate(
                [[twisted[key][0].real, twisted[key][0].imag] for key in keys]
            )
            return (ours - target) / dominant

        # beta is bounded away from the mirror solution
        # (beta, alpha, gamma) -> (-beta, alpha + pi, gamma + pi), which
        # describes the same rotation and which the fit otherwise drifts
        # into whenever the opening angle passes near a minimum
        solution = least_squares(
            residual,
            current,
            bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.pi, np.inf]),
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
        )
        recovered[position] = current = solution.x
        residuals[position] = np.max(np.abs(solution.fun))

    return indices, recovered, residuals


def wrapped(angle):
    """An angle difference, brought back into :math:`(-\pi, \pi]`."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def report_angle_recovery(times, indices, recovered, residuals, alpha, beta, gamma):
    """Print how far the recovered angles are from the ones integrated here."""
    print("\nEuler angles recovered from TEOBResumS' own multipoles:\n")
    print(f"  fit residual / max|h_22|   {np.max(residuals):.3e} (max), "
          f"{np.median(residuals):.3e} (median)")

    delta_beta = recovered[:, 1] - beta[indices]
    delta_difference = wrapped(
        (recovered[:, 0] - recovered[:, 2]) - (alpha[indices] - gamma[indices])
    )
    delta_sum = wrapped(
        (recovered[:, 0] + recovered[:, 2]) - (alpha[indices] + gamma[indices])
    )
    print(f"  max |d beta|               {np.max(np.abs(delta_beta)):.3e} rad "
          f"({np.max(np.abs(delta_beta)) / np.max(beta):.2%} of max beta)")
    print(f"  max |d (alpha - gamma)|    {np.max(np.abs(delta_difference)):.3e} rad")
    print(f"  max |d (alpha + gamma)|    {np.max(np.abs(delta_sum)):.3e} rad "
          "(weakly constrained; see docstring)")

    return delta_beta, delta_difference


def check_our_integration_is_converged(chi_1, chi_2, times, coprecessing, inertial, dynamics):
    r"""Re-integrate the PN precession sloppily, and see nothing change.

    If the disagreement with TEOBResumS were down to the accuracy of the
    integration done *here*, loosening its tolerances by four orders of
    magnitude would move it. It does not, which leaves the difference in
    the spin-precession system itself --- its initial conditions, or
    whatever accuracy TEOBResumS integrates its own copy to.
    """
    dominant = np.max(np.abs(inertial[(2, 2)]))
    print("\nsensitivity to our own integration accuracy:\n")
    print("rtol      (2,1) / h22")

    for rtol in (1e-7, 1e-9, 1e-11):
        angles = euler_angles_on_grid(times, dynamics, chi_1, chi_2, rtol=rtol)
        twisted = twisted_modes(coprecessing, *angles)
        error = np.max(relative_error(twisted[(2, 1)], inertial[(2, 1)], scale=dominant))
        print(f"{rtol:.0e}   {error:.6e}")


def plot(
    times,
    ours,
    theirs,
    our_polarizations,
    polarizations,
    angles,
    convergence,
    recovery,
):
    alpha, beta, gamma = angles
    indices, recovered, delta_beta, delta_difference = recovery

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    dominant = np.max(np.abs(theirs[(2, 2)]))
    for key in sorted(theirs):
        if key[1] < 0:
            continue
        axes[0, 0].semilogy(
            times / 1e3,
            relative_error(ours[key], theirs[key], scale=dominant),
            lw=0.8,
            label=rf"$({key[0]}, {key[1]})$",
        )
    axes[0, 0].set_ylabel(
        r"$|h_{\ell m} - h^{\rm TEOB}_{\ell m}| / \max |h^{\rm TEOB}_{22}|$"
    )
    axes[0, 0].set_xlabel(r"$t / 10^3 M$")
    axes[0, 0].legend(ncol=4, fontsize="small")
    axes[0, 0].set_title("Twisted multipoles against TEOBResumS")

    axes[0, 1].loglog(convergence[:, 1], convergence[:, 2], "o-", label=r"$(2, 1)$")
    axes[0, 1].loglog(convergence[:, 1], convergence[:, 3], "s-", label=r"$(2, 2)$")
    reference = convergence[:, 1] / convergence[0, 1]
    for column, style in ((2, "--"), (3, ":")):
        axes[0, 1].loglog(
            convergence[:, 1],
            convergence[0, column] * reference,
            ls=style,
            c="grey",
            lw=0.8,
            label=r"$\propto \beta$" if column == 2 else None,
        )
    axes[0, 1].set_xlabel(r"$\max_t \beta$ [rad]")
    axes[0, 1].set_ylabel(r"error $/ \max |h^{\rm TEOB}_{22}|$")
    axes[0, 1].legend(fontsize="small")
    axes[0, 1].set_title("The residual vanishes with the twist")

    window = times > times[-1] - 2_000.0
    axes[0, 2].plot(
        times[window], polarizations[window].real, lw=1.2, label="TEOBResumS"
    )
    axes[0, 2].plot(
        times[window],
        our_polarizations[window].real,
        lw=1.0,
        ls="--",
        label="twisted here",
    )
    axes[0, 2].set_ylabel(r"$h_+$")
    axes[0, 2].set_xlabel(r"$t / M$")
    axes[0, 2].legend()
    axes[0, 2].set_title(f"Polarization, last $2000\\,M$ ($\\iota = {INCLINATION}$)")

    axes[1, 0].plot(times / 1e3, beta, lw=0.8, c="tab:green", label="integrated here")
    axes[1, 0].plot(
        times[indices] / 1e3,
        recovered[:, 1],
        ls="none",
        marker=".",
        ms=4,
        c="black",
        label="recovered from TEOBResumS",
    )
    axes[1, 0].set_ylabel(r"$\beta(t)$ [rad]")
    axes[1, 0].set_xlabel(r"$t / 10^3 M$")
    axes[1, 0].legend(fontsize="small")
    axes[1, 0].set_title("Opening angle of the orbital plane")

    axes[1, 1].plot(times[indices] / 1e3, delta_beta, lw=0.8, c="tab:red")
    axes[1, 1].axhline(0.0, c="grey", lw=0.6)
    axes[1, 1].set_ylabel(r"$\beta - \beta^{\rm TEOB}$ [rad]")
    axes[1, 1].set_xlabel(r"$t / 10^3 M$")
    axes[1, 1].set_title("...and how far off it is")

    axes[1, 2].semilogy(
        times[indices] / 1e3,
        np.abs(delta_difference),
        lw=0.8,
        c="tab:purple",
    )
    axes[1, 2].set_ylabel(r"$|\Delta(\alpha - \gamma)|$ [rad]")
    axes[1, 2].set_xlabel(r"$t / 10^3 M$")
    axes[1, 2].set_title("The combination the multipoles pin down")

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    print(f"\nfigure written to {FIGURE_PATH}")


if __name__ == "__main__":
    chi_1, chi_2 = scaled_spins(1.0)
    (
        times,
        ours,
        inertial,
        our_polarizations,
        polarizations,
        angles,
        coprecessing,
        dynamics,
    ) = twist_here(chi_1, chi_2)

    report(times, ours, inertial, our_polarizations, polarizations, angles)

    indices, recovered, residuals = recover_euler_angles(
        times, coprecessing, inertial, angles
    )
    delta_beta, delta_difference = report_angle_recovery(
        times, indices, recovered, residuals, *angles
    )

    check_our_integration_is_converged(
        chi_1, chi_2, times, coprecessing, inertial, dynamics
    )

    convergence = convergence_check()
    plot(
        times,
        ours,
        inertial,
        our_polarizations,
        polarizations,
        angles,
        convergence,
        (indices, recovered, delta_beta, delta_difference),
    )
