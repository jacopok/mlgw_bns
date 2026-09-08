r"""How a TEOBResumS multi-mode frequency-domain waveform depends on the
ODE integration start frequency.

Self-contained (only ``EOBRun_module``; a spin-2 spherical harmonic is
inlined). Every call returns the waveform on the same explicit 40--1500 Hz
grid via ``interp_freqs``/``freqs``, so there is no interpolation here and
every :math:`(\ell, m)` up to :math:`(4,4)` has full support.

Findings for a q = 1.6, M = 2.8 :math:`M_\odot` BNS at :math:`\iota = 1`:

1. The :math:`(2,2)` is start-frequency independent to a mismatch of
   ``~3e-8`` (numerical noise).

2. With the higher modes on, the *summed* waveform moves by ``~2e-3``
   under a change of ``initial_frequency`` if only a single global phase
   and a time shift are allowed. That is a convention, not an
   inconsistency: a coalescence-phase change rotates mode
   :math:`(\ell, m)` by :math:`e^{i m \phi_c}`, and once that per-mode
   rotation is marginalised the mismatch drops back to ``~5e-6``.
   TEOBResumS just assigns a different coalescence-phase zero for each
   start frequency.

3. TEOBResumS computes each multipole independently: a mode extracted
   from a joint ``use_mode_lm`` call is byte-identical to the same mode
   from a single-mode call, at every start frequency. Each higher mode
   on its own is start-frequency invariant to ``~1e-7`` --- except the
   :math:`(4,4)`, which carries a fixed ``~1e-4`` phase-shape residual
   (the same at every start frequency, so a modelling difference, not a
   convergence problem).

So a summed multi-mode TEOBResumS FD waveform is self-consistent under a
change of start frequency to ``~5e-6`` once the coalescence phase is
marginalised per mode.

The reason the surrogate's frequency-domain mismatch against an
*independent* ``EOBRunPy(initial_frequency=15)`` is larger over a wide
band is **band-edge support**, not the overlap region: a 15 Hz-start
call only produces the :math:`(3,3)` above 22.5 Hz and the :math:`(4,4)`
above 30 Hz, whereas ``mlgw_bns`` trains from ``all_modes_amplitude_phase``
(which starts the :math:`(2,2)` ODE near 1.8 Hz) and so has the higher
modes down to 20 Hz. Restricted to the band where both have every mode,
the surrogate agrees with an independent TEOBResumS call to ``~3e-6``.

Run with: python visualization/teob_hom_start_frequency.py
"""

from __future__ import annotations

import math

import numpy as np

try:
    from EOBRun_module import EOBRunPy
except ImportError as error:  # pragma: no cover
    raise SystemExit(f"needs EOBRun_module: {error}")

#: 40 Hz is above the (4,4) onset for every start frequency used below.
GRID = np.arange(40.0, 1500.0, 0.1)

#: TEOBResumS mode indices k = l(l-1)/2 + m - 2.
K = {(2, 2): 1, (2, 1): 0, (3, 3): 4, (4, 4): 8}
ALL = [(2, 2), (2, 1), (3, 3), (4, 4)]

Q, M, IOTA = 1.6, 2.8, 1.0
MSUN_SECONDS = 4.925490947e-6


def sYlm(l: int, m: int, theta: float) -> float:
    """{}_{-2}Y_{lm}(theta, 0) -- real for the phi = 0 used here."""
    c, si = math.cos(theta / 2), math.sin(theta / 2)
    s = 2
    norm = math.sqrt(
        math.factorial(l + m) * math.factorial(l - m)
        * math.factorial(l + s) * math.factorial(l - s)
    )
    d = sum(
        (-1) ** k * c ** (2 * l + m - s - 2 * k) * si ** (2 * k + s - m)
        / (math.factorial(k) * math.factorial(l + m - k)
           * math.factorial(l - s - k) * math.factorial(s - m + k))
        for k in range(max(0, m - s), min(l + m, l - s) + 1)
    )
    return math.sqrt((2 * l + 1) / (4 * math.pi)) * norm * d


def teob_modes(
    initial_frequency: float,
    requested: list[tuple[int, int]],
) -> dict[tuple[int, int], np.ndarray]:
    r"""Per-mode ``h_lm(f) {}_{-2}Y_{lm}`` on ``GRID`` from one EOBRunPy call,
    each phase re-referenced to the merger (``- 2 pi t_c f``)."""
    par = dict(
        q=Q, LambdaAl2=400.0, LambdaBl2=400.0, chi1=0.05, chi2=-0.05, M=M,
        distance=100.0, inclination=IOTA, initial_frequency=initial_frequency,
        domain=1, use_geometric_units="no", use_spins=1, interp_freqs="yes",
        freqs=list(GRID), output_hpc="no", arg_out="yes",
        use_mode_lm=sorted(K[mode] for mode in requested),
    )
    f_spa, *_, hflm, htlm, dyn = EOBRunPy(par)
    f_spa = np.asarray(f_spa)
    tc = float(dyn["tc"]) if "tc" in dyn else float(np.asarray(htlm["t"])[-1])
    tc_s = tc * M * MSUN_SECONDS
    out = {}
    for mode in requested:
        amp = np.asarray(hflm[str(K[mode])][0])
        phase = np.asarray(hflm[str(K[mode])][1]) - 2 * np.pi * tc_s * f_spa
        out[mode] = amp * np.exp(1j * phase) * sYlm(*mode, IOTA)
    return out


def mismatch(a: np.ndarray, b: np.ndarray) -> float:
    """Flat-PSD mismatch, maximised over an overall phase and a time shift."""
    norm = np.sqrt(np.abs(np.vdot(a, a)) * np.abs(np.vdot(b, b)))
    if norm == 0:
        return np.nan
    shifts = np.linspace(-0.02, 0.02, 8001)
    overlaps = np.exp(2j * np.pi * np.outer(shifts, GRID)) @ (np.conj(a) * b)
    return 1.0 - np.max(np.abs(overlaps)) / norm


def mismatch_coalescence(ref_modes, trial_modes) -> float:
    r"""Summed-waveform mismatch, maximised over a time shift and a genuine
    coalescence phase (mode m rotated by :math:`e^{i m \phi_c}`)."""
    keys = list(ref_modes)
    m_values = np.array([m for _, m in keys])[:, None]
    ref = sum(ref_modes.values())
    stack = np.stack([trial_modes[k] for k in keys])
    ref_norm_sq = np.abs(np.vdot(ref, ref))
    time_kernels = np.exp(
        2j * np.pi * np.outer(np.linspace(-0.02, 0.02, 4001), GRID)
    )
    best = 0.0
    for phi_c in np.linspace(-np.pi, np.pi, 721):
        trial = (stack * np.exp(1j * m_values * phi_c)).sum(axis=0)
        norm = np.sqrt(ref_norm_sq * np.abs(np.vdot(trial, trial)))
        best = max(best, np.max(np.abs(time_kernels @ (np.conj(ref) * trial)) / norm))
    return 1.0 - best


def main() -> None:
    print(f"q = {Q}, M = {M} Msun, iota = {IOTA}, band [40, 1500] Hz")

    ref22 = teob_modes(20.0, [(2, 2)])
    refall = teob_modes(20.0, ALL)
    print("\n1/2. summed waveform vs its own 20 Hz-start version:")
    for start in (15.0, 10.0, 7.0, 5.0):
        a22 = teob_modes(start, [(2, 2)])
        aall = teob_modes(start, ALL)
        print(f"   f0 = {start:4.1f} Hz :  (2,2) {mismatch(ref22[(2,2)], a22[(2,2)]):.1e}"
              f"   full HOM, one phase {mismatch(sum(refall.values()), sum(aall.values())):.1e}"
              f"   full HOM, e^(i m phi_c) {mismatch_coalescence(refall, aall):.1e}")

    print("\n3. each mode: from a single-mode call vs from the joint 4-mode "
          "call (should be identical), and vs its own 20 Hz start:")
    for mode in ALL:
        solo = teob_modes(20.0, [mode])[mode]
        joint = teob_modes(20.0, ALL)[mode]
        solo5 = teob_modes(5.0, [mode])[mode]
        print(f"   {mode}: solo-vs-joint {mismatch(solo, joint):.1e}   "
              f"20Hz-vs-5Hz {mismatch(solo, solo5):.1e}")


if __name__ == "__main__":
    main()
