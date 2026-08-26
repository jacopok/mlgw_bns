"""
Standalone reproduction of the TEOBResumS waveform "twisting" procedure.

TEOBResumS generates the precessing waveform by first evolving the
dynamics in a co-precessing frame (where only the m=+-2 multipoles
are large) and then rotating ("twisting") the co-precessing multipoles
h_{l,n}(t) = A_{l,n}(t) * exp(-i*phi_{l,n}(t))  (n = 1 .. l)
into the fixed inertial frame using three time-dependent Euler angles
alpha(t), beta(t), gamma(t):

    h^inertial_{l,m}(t) = sum_n D^l_{m,n}(alpha,beta,gamma) h^coprec_{l,n}(t)

with the D-matrix built out of Wigner-d functions d^l_{m,n}(beta) and the
usual m=<0 symmetry of the co-precessing multipoles,
H_{l,-n} = (-1)^l H_{l,n}^*.

This module reimplements, line for line, the algebra of:

  - rmap_twist()          in C/src/TEOBResumSUtils.c
  - twist_hlm_TD()        in C/src/TEOBResumSWaveform.c
  - compute_hpc()         in C/src/TEOBResumSWaveform.c  (bonus: h+, hx)

as well as the PN spin-precession dynamics used to *produce* alpha(t),
beta(t), gamma(t) in the EOBPars->project_spins == 0 (default), the
EOBPars->spin_flx == SPIN_FLX_PN (default) branch:

  - eob_spin_dyn_rhs_PN() in C/src/TEOBResumSDynamics.c  (SPIN_FLX_PN case)
  - eob_spin_dyn()               "                 "     (initial data)
  - eob_spin_dyn_alpha/beta()    "                 "
  - alpha_initial_condition()    "                 "
  - eob_mrg_momg()        in C/src/TEOBResumSFits.c       (stop frequency)


(The Wigner d-function and the spin-weighted spherical harmonics of
C/src/SpecialFuns.c live in mlgw_bns.special_func, which is shared with
the rest of the package.)

Convention (matches the C code throughout):
    h_{l,m} = A_{l,m} * exp(-i * phi_{l,m}),   phi_{l,m} increasing in time.

For the surrogate-facing, frequency-domain use of all this --- taking
the aligned-spin mlgw_bns multipoles as the co-precessing ones and
twisting them into a full precessing waveform --- see
mlgw_bns.precessing_model.
"""

import math

import numpy as np
from scipy.integrate import solve_ivp  # type: ignore

from .special_func import spinsphericalharm, wigner_d_function

_EULER_GAMMA = 0.5772156649015328606065121  # matches EulerGamma in TEOBResumS.h


# ---------------------------------------------------------------------------
# Amplitude/phase <-> real/imaginary  (C/src/TEOBResumSUtils.c: rmap_twist)
# ---------------------------------------------------------------------------

def amp_phase_to_complex(a, p):
    """(Amplitude, phase) -> complex h = A*exp(-i*phi). rmap_twist(mode=0)."""
    return a * np.cos(p) - 1j * a * np.sin(p)


def complex_to_amp_phase(h):
    """complex h = A*exp(-i*phi) -> (Amplitude, phase). rmap_twist(mode=1)."""
    a = np.abs(h)
    p = -np.angle(h)
    return a, p


# ---------------------------------------------------------------------------
# The twist itself  (C/src/TEOBResumSWaveform.c: twist_hlm_TD)
# ---------------------------------------------------------------------------

def twist_modes(hlm_coprec, alpha, beta, gamma, lm_inertial,
                include_positive_n=True, include_negative_n=True):
    """
    Twist co-precessing multipoles into the inertial frame.

    Parameters
    ----------
    hlm_coprec : dict[(l, n)] -> complex ndarray
        Co-precessing multipoles for n = 1 .. l (m>0 only; the m<0
        co-precessing multipoles are reconstructed from the standard
        symmetry H_{l,-n} = (-1)^l H_{l,n}^*, exactly as the C code does).
        h = A*exp(-i*phi) convention. All arrays share the same grid.
    alpha, beta, gamma : ndarray
        Euler angles (radians) on that same grid.
    lm_inertial : iterable of (l, m), m > 0
        Inertial-frame multipoles to produce.
    include_positive_n, include_negative_n : bool
        Whether to include, in the sum over the co-precessing index n,
        the n>0 multipoles and their n<0 partners respectively. Both
        default to True, which is what the time-domain twist of the C
        code does. Splitting the two halves apart is what lets the same
        routine be used in the frequency domain, where the n>0 and n<0
        co-precessing multipoles have their support at opposite signs of
        the frequency and so must be twisted separately --- see
        :func:`~mlgw_bns.precessing_model.twist_modes_frequency_domain`.

    Returns
    -------
    hTlm, hTlm_neg, hTl0 : dict[(l, m)] -> complex ndarray
        Twisted inertial multipoles for m>0, m<0 (keyed by the negative m)
        and m=0 respectively, h = A*exp(-i*phi) convention.
    """
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    size = alpha.size

    hTlm, hTlm_neg, hTl0 = {}, {}, {}
    m0_done_for_ell = set()

    for (ell, m) in sorted(lm_inertial):
        eps = (-1.0) ** ell

        for q, emm in enumerate((m, -m, 0)):  # m>0, m<0, m=0
            if q == 2 and ell in m0_done_for_ell:
                continue

            sumr = np.zeros(size)
            sumi = np.zeros(size)

            for n in range(1, ell + 1):
                if (ell, n) not in hlm_coprec:
                    continue  # co-precessing mode not available, skip

                real, imag = hlm_coprec[(ell, n)].real, hlm_coprec[(ell, n)].imag

                cosng = np.cos(n * gamma)
                sinng = np.sin(n * gamma)

                # d^l_{n,emm}(-beta) and d^l_{-n,emm}(-beta)
                dl_mn = wigner_d_function(ell, n, emm, -beta)
                dl_mnn = wigner_d_function(ell, -n, emm, -beta)

                # n>0 co-precessing multipole
                hln_real_p, hln_imag_p = real, imag
                # n<0 co-precessing multipole via H_{l,-n} = (-1)^l H_{l,n}^*
                hln_real_n, hln_imag_n = eps * real, -eps * imag

                if include_positive_n:
                    sumr += dl_mn * (cosng * hln_real_p - sinng * hln_imag_p)
                    sumi += dl_mn * (sinng * hln_real_p + cosng * hln_imag_p)
                if include_negative_n:
                    sumr += dl_mnn * (cosng * hln_real_n + sinng * hln_imag_n)
                    sumi += dl_mnn * (-sinng * hln_real_n + cosng * hln_imag_n)

            hTlm_real = sumr * np.cos(emm * alpha) + sumi * np.sin(emm * alpha)
            hTlm_imag = -sumr * np.sin(emm * alpha) + sumi * np.cos(emm * alpha)
            h_out = hTlm_real + 1j * hTlm_imag

            if q == 0:
                hTlm[(ell, m)] = h_out
            elif q == 1:
                hTlm_neg[(ell, -m)] = h_out
            else:
                hTl0[(ell, 0)] = h_out
                m0_done_for_ell.add(ell)

    return hTlm, hTlm_neg, hTl0


# ---------------------------------------------------------------------------
# PN spin-precession dynamics  (project_spins = 0, spin_flx = SPIN_FLX_PN)
# (C/src/TEOBResumSDynamics.c: eob_spin_dyn_rhs_PN, eob_spin_dyn,
#  eob_spin_dyn_integrate, eob_spin_dyn_alpha/beta, alpha_initial_condition;
#  C/src/TEOBResumSFits.c: eob_mrg_momg)
# ---------------------------------------------------------------------------

def nu_to_X1(nu):
    """Mass fraction M1/M from the symmetric mass ratio nu. Mirrors
    nu_to_X1() in C/src/TEOBResumSUtils.c."""
    if nu < 0.0 or nu > 0.25:
        raise ValueError("symmetric mass ratio must be 0 <= nu <= 1/4")
    return 0.5 * (1.0 + math.sqrt(1.0 - 4.0 * nu))


def eob_mrg_momg(nu, X1, X2, chi1, chi2):
    """
    NR fit for M*Omega_orb at merger. Mirrors eob_mrg_momg() in
    C/src/TEOBResumSFits.c. `chi1`, `chi2` are the (z-component,
    dimensionless) spins of the two bodies.
    """
    nu2 = nu * nu
    X12 = math.sqrt(1.0 - 4.0 * nu)
    a1 = X1 * chi1
    a2 = X2 * chi2
    a0 = a1 + a2
    a12 = a1 - a2
    Shat = 0.5 * (a0 + X12 * a12)
    Shat2 = Shat * Shat
    b0, b1, b2, b3 = 0.066045, -0.23876, 0.76819, -0.9201
    return (0.273356 * (1.0 + 0.84074 * nu + 1.6976 * nu2)
            * (1.0 + ((-0.42311 + b0 * X12) / (1.0 + b1 * X12)) * Shat
                    + (-0.066699) * Shat2)
            / (1.0 + ((-0.83053 + b2 * X12) / (1.0 + b3 * X12)) * Shat))


def alpha_initial_condition(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, f0):
    """
    NLO initial value of the alpha (and, per the C code, gamma) Euler
    angle. Eq. A5 of arXiv:2111.03675. Mirrors alpha_initial_condition()
    in C/src/TEOBResumSDynamics.c. Geometric units (G=c=M=1); `f0` is the
    initial GW (2,2) frequency (so that the initial M*Omega_orb = pi*f0).
    """
    v = (math.pi * f0) ** (1.0 / 3.0)
    alpha_x_NLO = (-3.0 * q * (chi1y + chi2y * q) * (chi1z + chi2z * q) * v
                   + q * (chi1y * (4.0 + 3.0 * q) + chi2y * q * (3.0 + 4.0 * q)))
    alpha_y_NLO = (3.0 * q * (chi1x + chi2x * q) * (chi1z + chi2z * q) * v
                   - q * (chi1x * (4.0 + 3.0 * q) + chi2x * q * (3.0 + 4.0 * q)))
    return math.atan2(alpha_y_NLO, alpha_x_NLO)


def _pn_spin_precession_rhs(nu, q, y):
    """
    RHS of the PN spin-precession ODE system, EOBPars->spin_flx ==
    SPIN_FLX_PN branch: N4LO spin-orbit + spin-spin precession of SA, SB,
    Lhat (arXiv:2005.05338), and 3.5PN energy-balance frequency evolution
    (Eq. A1 of arXiv:1307.4418). Mirrors the SPIN_FLX_PN branch of
    eob_spin_dyn_rhs_PN() in C/src/TEOBResumSDynamics.c *exactly*,
    including which higher-order (4PN+) coefficients it computes but never
    actually adds to the sum (the C code's `for (i=2;i<8;i++)` loop caps
    the frequency evolution at 3.5PN even though a[8..11]/b[8..11] and
    beta8A/B are computed) -- those unused terms are simply omitted here.

    State y = [SAx,SAy,SAz, SBx,SBy,SBz, Lx,Ly,Lz, gamma, Momega].
    (alpha, beta are NOT part of the integrated state here -- see the note
    in integrate_pn_spin_precession().)
    """
    nu2, nu3 = nu * nu, nu ** 3
    Pi = math.pi
    Pi2 = Pi * Pi
    oothree = 1.0 / 3.0
    eleven_o_three = 11.0 / 3.0

    MA = nu_to_X1(nu)
    MB = 1.0 - MA
    dm = MA - MB
    ma_o_mb = MA / MB
    mb_o_ma = MB / MA

    SA = np.asarray(y[0:3], dtype=float)
    SB = np.asarray(y[3:6], dtype=float)
    Lh = np.asarray(y[6:9], dtype=float)
    omg = y[10]

    lnomg = math.log(omg)
    v = omg ** oothree
    v2, v3 = v * v, v ** 3
    v5, v6, v7, v9 = v ** 5, v ** 6, v ** 7, v ** 9

    qSAB = SA / q + SB
    SABq = SA + SB * q
    qSABLh = np.dot(qSAB, Lh)
    SABqLh = np.dot(SABq, Lh)

    v5_cA = nu * (2.0 + 1.5 / q)
    v5_cB = nu * (2.0 + 1.5 * q)
    v7_cA = 0.5625 + 1.25 * nu - 0.041666666666666664 * nu2 + dm * (-0.5625 + 0.625 * nu)
    v7_cB = 0.5625 + 1.25 * nu - 0.041666666666666664 * nu2 - dm * (-0.5625 + 0.625 * nu)
    v9_cA = (0.84375 + 0.1875 * nu - 3.28125 * nu2 - 0.02083333333333 * nu3
             + dm * (-0.84375 + 4.875 * nu - 0.15625 * nu2))
    v9_cB = (0.84375 + 0.1875 * nu - 3.28125 * nu2 - 0.02083333333333 * nu3
             - dm * (-0.84375 + 4.875 * nu - 0.15625 * nu2))

    csA = -0.25 * (3.0 + 1.0 / MA)
    csB = -0.25 * (3.0 + 1.0 / MB)
    csAL = -0.08333333333333333 * (1.0 + 27.0 / MA)
    csBL = -0.08333333333333333 * (1.0 + 27.0 / MB)

    L2PN_v2 = 1.5 + 0.1666666666666667 * nu
    L2PN_v4 = 3.375 - 2.375 * nu + 0.041666666666666664 * nu2
    v4 = v2 * v2

    OmgANLO  = (v5 * v5_cA - 1.5 * v6 * qSABLh) * Lh + 0.5 * v6 * SB
    OmgBNLO  = (v5 * v5_cB - 1.5 * v6 * SABqLh) * Lh + 0.5 * v6 * SA
    OmgANNLO = OmgANLO + v7 * v7_cA * Lh
    OmgBNNLO = OmgBNLO + v7 * v7_cB * Lh
    OmgAN4LO = OmgANNLO + v9 * v9_cA * Lh
    OmgBN4LO = OmgBNNLO + v9 * v9_cB * Lh

    SdotANLO  = np.cross(OmgANLO,  SA)
    SdotANNLO = np.cross(OmgANNLO, SA)
    SdotAN4LO = np.cross(OmgAN4LO, SA)
    SdotBNLO  = np.cross(OmgBNLO,  SB)
    SdotBNNLO = np.cross(OmgBNNLO, SB)
    SdotBN4LO = np.cross(OmgBN4LO, SB)

    dSA = SdotAN4LO
    dSB = SdotBN4LO

    L2PN = 1.0 + v2 * L2PN_v2 + v4 * L2PN_v4
    v_o_nu = v / nu

    SALh = np.dot(SA, Lh)
    SBLh = np.dot(SB, Lh)
    dSBNLOSA  = np.dot(SdotBNLO,  SA)
    dSANLOSB  = np.dot(SdotANLO,  SB)
    dSANNLOLh = np.dot(SdotANNLO, Lh)
    dSBNNLOLh = np.dot(SdotBNNLO, Lh)

    # Eq. (4c) of arXiv:2005.05338
    LNdotN4LO = (
        v_o_nu * (-SdotAN4LO - SdotBN4LO)
        - v3 * (csA * SdotANNLO + csB * SdotBNNLO)
        - v3 * (csAL * (-v_o_nu * (SdotANLO + SdotBNLO) * SALh
                         + Lh * (-v_o_nu * dSBNLOSA + dSANNLOLh))
                + csBL * (-v_o_nu * (SdotBNLO + SdotANLO) * SBLh
                          + Lh * (-v_o_nu * dSANLOSB + dSBNNLOLh)))
    ) / L2PN

    # Eq. (7) of arXiv:2005.05338 -- keep only the component perp. to Lh
    LNdotN4LOLh = np.dot(LNdotN4LO, Lh)
    dLh = LNdotN4LO - LNdotN4LOLh * Lh

    # dot{gamma} = + dot{alpha} * cos(beta), with cos(beta) = Lh_z
    div = Lh[0] ** 2 + Lh[1] ** 2
    if div == 0.0:
        dgamma = 0.0
    else:
        dgamma = Lh[2] * (Lh[0] * dLh[1] - Lh[1] * dLh[0]) / div

    # --- Eq. (A1) of arXiv:1307.4418: PN energy-balance domega/dt (3.5PN) ---
    sigma4_SASB      =  247.0 / (48.0 * nu)
    sigma4_SALh_SBLh = -721.0 / (48.0 * nu)
    sigma4_SA2       =  233.0 / (96.0 * MA ** 2)
    sigma4_SALh2     = -719.0 / (96.0 * MA ** 2)
    sigma4_SB2       =  233.0 / (96.0 * MB ** 2)
    sigma4_SBLh2     = -719.0 / (96.0 * MB ** 2)

    beta3A = 113.0 / 12 + 25.0 / 4 * mb_o_ma
    beta3B = 113.0 / 12 + 25.0 / 4 * ma_o_mb
    beta5A = (31319.0 / 1008 - 1159.0 / 24 * nu) + mb_o_ma * (809.0 / 84 - 281.0 / 8 * nu)
    beta5B = (31319.0 / 1008 - 1159.0 / 24 * nu) + ma_o_mb * (809.0 / 84 - 281.0 / 8 * nu)
    beta6A = Pi * (75.0 / 2 + 151.0 / 6 * mb_o_ma)
    beta6B = Pi * (75.0 / 2 + 151.0 / 6 * ma_o_mb)
    beta7A = ((130325.0 / 756 - 796069.0 / 2016 * nu + 100019.0 / 864 * nu2)
              + mb_o_ma * (1195759.0 / 18144 - 257023.0 / 1008 * nu + 2903.0 / 32 * nu2))
    beta7B = ((130325.0 / 756 - 796069.0 / 2016 * nu + 100019.0 / 864 * nu2)
              + ma_o_mb * (1195759.0 / 18144 - 257023.0 / 1008 * nu + 2903.0 / 32 * nu2))

    b6 = -1712.0 / 315

    a0 = 96.0 / 5 * nu
    a2 = -743.0 / 336 - 11.0 / 4 * nu
    a4_nosigma = 34103.0 / 18144 + 13661.0 / 2016 * nu + 59.0 / 18 * nu2
    a5_nobeta  = -4159.0 / 672 * Pi - 189.0 / 8 * Pi * nu
    a6_nobeta  = (16447322263.0 / 139708800 + 16.0 / 3 * Pi2 - 856.0 / 105 * math.log(16.0)
                  - 1712.0 / 105 * _EULER_GAMMA
                  + nu * (451.0 / 48 * Pi2 - 56198689.0 / 217728) + nu2 * 541.0 / 896 - nu3 * 5605.0 / 2592)
    a7_nobeta  = -4415.0 / 4032 * Pi + 358675.0 / 6048 * Pi * nu + 91495.0 / 1512 * Pi * nu2

    SAdotLh, SBdotLh = SALh, SBLh
    SAdotSB = np.dot(SA, SB)
    SA2 = np.dot(SA, SA)
    SB2 = np.dot(SB, SB)

    sigma4 = (sigma4_SASB * SAdotSB + sigma4_SALh_SBLh * SAdotLh * SBdotLh
              + sigma4_SA2 * SA2 + sigma4_SALh2 * SAdotLh ** 2
              + sigma4_SB2 * SB2 + sigma4_SBLh2 * SBdotLh ** 2)

    beta3 = beta3A * SAdotLh + beta3B * SBdotLh
    beta5 = beta5A * SAdotLh + beta5B * SBdotLh
    beta6 = beta6A * SAdotLh + beta6B * SBdotLh
    beta7 = beta7A * SAdotLh + beta7B * SBdotLh

    a = [0.0] * 8
    b = [0.0] * 8
    a[0], a[2] = a0, a2
    a[3] = 4.0 * Pi - beta3
    a[4] = a4_nosigma - sigma4
    a[5] = a5_nobeta - beta5
    a[6] = a6_nobeta - beta6
    a[7] = a7_nobeta - beta7
    b[6] = b6

    domg = 0.0
    for i in range(2, 8):
        domg += (a[i] + b[i] * lnomg) * omg ** (i * oothree)
    domg += 1.0
    domg *= a[0] * omg ** eleven_o_three

    return np.concatenate([dSA, dSB, dLh, [dgamma, domg]])


def integrate_pn_spin_precession(nu, chi1vec, chi2vec, f0, t_max=1.0e6, max_step=None,
                                 refinement=32, rtol=1.0e-11, atol=1.0e-13):
    """
    Integrate the closed PN spin-precession ODE system that TEOBResumS
    solves in eob_spin_dyn()/eob_spin_dyn_integrate() for
    EOBPars->use_spins == MODE_SPINS_GENERIC, EOBPars->project_spins == 0
    (the default) and EOBPars->spin_flx == SPIN_FLX_PN (the default).

    Note on alpha, beta: in the C code these are formally ODE state
    variables, but their r.h.s. is identically zero (see the "alpha and
    beta are not actually evolved" comment in eob_spin_dyn_rhs_PN); after
    every accepted forward step they are instead overwritten algebraically
    from the new Lhat: alpha = atan2(Lhat_y, Lhat_x), beta = acos(Lhat_z).
    That is mathematically identical to treating them as *derived*
    quantities of Lhat(t) rather than as independent state, which is what
    is done here; only SA, SB, Lhat, gamma and Momega are integrated.

    Parameters
    ----------
    nu : symmetric mass ratio (sets MA = m1/M >= MB = m2/M via
         MA = nu_to_X1(nu))
    chi1vec, chi2vec : length-3 sequences
        Dimensionless spin vectors of body 1 (mass fraction MA) and body 2
        (mass fraction MB).
    f0 : initial GW (2,2) frequency, M*f0 in geometric units (so the
         initial M*Omega_orb = pi*f0, matching EOBPars->initial_frequency).
    t_max : safety cutoff on the integration time (geometric units, M=1).
    refinement : int
        Number of sub-intervals each accepted step of the integrator is
        split into, using its dense output, before the angles are
        returned. The integrator is accurate but takes long steps --- a
        whole inspiral can be covered in a couple of hundred of them ---
        and the returned arrays are meant to be *interpolated*, so the
        grid has to resolve the precession, not merely carry it. Without
        this the angles are perfectly accurate at the returned points
        and visibly wrong between them. Costs only polynomial
        evaluations, no extra right-hand sides.
    rtol, atol : float
        Tolerances passed to the integrator.

    Returns
    -------
    dict with keys 't', 'alpha', 'beta', 'gamma', 'Momega', 'SA', 'SB', 'Lh'
    (ndarrays; SA/SB/Lh have shape (N,3)).
    """
    MA = nu_to_X1(nu)
    MB = 1.0 - MA
    q = MA / MB

    chi1vec = np.asarray(chi1vec, dtype=float)
    chi2vec = np.asarray(chi2vec, dtype=float)

    SA0 = chi1vec * MA ** 2
    SB0 = chi2vec * MB ** 2
    Lh0 = np.array([0.0, 0.0, 1.0])
    gamma0 = alpha_initial_condition(q, *chi1vec, *chi2vec, f0)  # == alpha0 in the C code
    Momg0 = math.pi * f0

    y0 = np.concatenate([SA0, SB0, Lh0, [gamma0, Momg0]])

    omg_stop = 1.1 * eob_mrg_momg(nu, MA, MB, chi1vec[2], chi2vec[2])

    def rhs(t, y):
        return _pn_spin_precession_rhs(nu, q, y)

    def event_reach_merger(t, y):
        return y[10] - omg_stop
    event_reach_merger.terminal = True
    event_reach_merger.direction = 1

    def event_domega_negative(t, y):
        return _pn_spin_precession_rhs(nu, q, y)[10]
    event_domega_negative.terminal = True
    event_domega_negative.direction = -1

    sol = solve_ivp(rhs, (0.0, t_max), y0, method="DOP853",
                     rtol=rtol, atol=atol,
                     events=[event_reach_merger, event_domega_negative],
                     dense_output=True,
                     max_step=(max_step if max_step is not None else np.inf))

    if not sol.success:
        raise RuntimeError(f"PN spin-precession integration failed: {sol.message}")

    if refinement > 1 and sol.t.size > 1:
        steps = np.diff(sol.t)[:, np.newaxis]
        offsets = np.arange(refinement)[np.newaxis, :] / refinement
        t = np.append((sol.t[:-1, np.newaxis] + steps * offsets).ravel(), sol.t[-1])
        y = sol.sol(t)
    else:
        t, y = sol.t, sol.y

    SA, SB, Lh = y[0:3].T, y[3:6].T, y[6:9].T
    gamma, Momega = y[9], y[10]

    alpha = np.arctan2(Lh[:, 1], Lh[:, 0])
    beta = np.arccos(np.clip(Lh[:, 2], -1.0, 1.0))

    return {"t": t, "alpha": alpha, "beta": beta, "gamma": gamma,
            "Momega": Momega, "SA": SA, "SB": SB, "Lh": Lh}


# ---------------------------------------------------------------------------
# Bonus: polarizations from the twisted (inertial) multipoles
# (C/src/TEOBResumSWaveform.c: compute_hpc, generic-spin/precessing branch)
# ---------------------------------------------------------------------------

def compute_hpc(hTlm, hTlm_neg, hTl0, phi, iota):
    """
    Combine twisted inertial multipoles into h+, hx for a given azimuth
    `phi` and inclination `iota`, mirroring compute_hpc() in the
    "generic spins" (precessing) branch: no +m/-m symmetry is assumed,
    m<0 and m=0 multipoles are used directly.

    `phi` is the azimuth of the line of sight, entering the harmonics as
    exp(i m phi). TEOBResumS parametrises the same freedom through its
    `coalescence_angle` parameter, with phi = pi/2 - coalescence_angle,
    so its default of 0 corresponds to phi = pi/2. With that
    identification this function reproduces the C code's h+ and hx to
    machine precision; see
    visualization/validate_twist_against_teob.py.

    Returns h = h+ - i*hx as a complex ndarray.
    """
    size = next(iter(hTlm.values())).size
    h = np.zeros(size, dtype=complex)

    for (l, m), hlm in hTlm.items():
        rY, iY = spinsphericalharm(-2, l, m, phi, iota)
        Y = rY + 1j * iY
        h += hlm * Y

    for (l, mneg), hlm in hTlm_neg.items():
        rY, iY = spinsphericalharm(-2, l, mneg, phi, iota)
        Y = rY + 1j * iY
        h += hlm * Y

    for (l, _), hlm in hTl0.items():
        rY, iY = spinsphericalharm(-2, l, 0, phi, iota)
        Y = rY + 1j * iY
        h += hlm * Y

    # compute_hpc() in the C code sets hplus = sumr and hcross = -sumi,
    # so h+ - i*hx is the sum itself
    return h
