r"""Check: how a TEOBResumS multi-mode FD waveform depends on the ODE
integration start frequency, in a band far above every mode's onset.

Self-contained (only ``EOBRun_module``). Two frequency-domain calls,
*identical except for* ``initial_frequency``, both asked (via
``interp_freqs``/``freqs``) to return the waveform on the same explicit
40--1500 Hz grid -- no interpolation on our side, and every
:math:`(\ell, m)` up to :math:`(4,4)` has full support across the band.

Findings for a q = 1.6, M = 2.8 :math:`M_\odot` BNS:

* the :math:`(2,2)`-only waveform is start-frequency independent to a
  mismatch of ``~3e-8`` (numerical noise);
* with the higher modes on, the *summed* waveform depends on the start
  frequency at ``~1e-3`` if you only allow a single global phase and a
  time shift -- but this is **not** a genuine inconsistency. A
  coalescence-phase change rotates mode :math:`(\ell, m)` by
  :math:`e^{i m \phi_c}`, and once that per-mode rotation is allowed the
  mismatch drops back to ``~7e-8``. TEOBResumS simply assigns a
  different coalescence-phase zero depending on ``initial_frequency``
  (the extra :math:`(2,2)` phase accumulated down to the lower start),
  which is a convention every parameter-estimation pipeline marginalises
  away, not a bug.
* the only genuine residual is a sub-:math:`10^{-4}` phase-shape change
  in the :math:`(4,4)` alone (``mm ~ 2e-4`` per mode, 20 vs 12 Hz),
  negligible once summed.

Nothing here needs reporting upstream. The practical note for
``mlgw_bns`` still stands: to keep a single consistent inter-mode phase
convention across the training set it generates every waveform from one
low, fixed start frequency and carries an explicit per-mode phase
constant through the surrogate.

Run with: python visualization/teob_hom_start_frequency.py
"""

from __future__ import annotations

import numpy as np

try:
    from EOBRun_module import EOBRunPy
except ImportError as error:  # pragma: no cover
    raise SystemExit(f"needs EOBRun_module: {error}")

#: 40 Hz is above the (4,4) onset (= 2 x f_orb) for every start frequency
#: used below, so the comparison band contains all four modes throughout.
GRID = np.arange(40.0, 1500.0, 0.1)

#: TEOBResumS mode indices k = l(l-1)/2 + m - 2.
K = {(2, 2): 1, (2, 1): 0, (3, 3): 4, (4, 4): 8}


def teob(initial_frequency: float, modes: list[tuple[int, int]]) -> np.ndarray:
    r"""``h_+ - i h_x`` on ``GRID`` for the given modes and start frequency."""
    par = dict(
        q=1.6,
        LambdaAl2=400.0,
        LambdaBl2=400.0,
        chi1=0.05,
        chi2=-0.05,
        M=2.8,
        distance=100.0,
        inclination=1.0,
        initial_frequency=initial_frequency,
        domain=1,
        use_geometric_units="no",
        use_spins=1,
        interp_freqs="yes",
        freqs=list(GRID),
        output_hpc="no",
        arg_out="no",
        use_mode_lm=[K[m] for m in modes],
    )
    _f, real_hp, imag_hp, real_hc, imag_hc = EOBRunPy(par)
    h_plus = np.asarray(real_hp) + 1j * np.asarray(imag_hp)
    h_cross = np.asarray(real_hc) + 1j * np.asarray(imag_hc)
    return h_plus - 1j * h_cross


def mismatch(a: np.ndarray, b: np.ndarray) -> float:
    r"""Flat-PSD mismatch, maximised over an overall phase and a time shift."""
    norm = np.sqrt(np.abs(np.vdot(a, a)) * np.abs(np.vdot(b, b)))
    shifts = np.linspace(-0.02, 0.02, 8001)
    overlaps = np.exp(2j * np.pi * np.outer(shifts, GRID)) @ (np.conj(a) * b)
    return 1.0 - np.max(np.abs(overlaps)) / norm


def mismatch_coalescence(
    ref_modes: dict[tuple[int, int], np.ndarray],
    modes: dict[tuple[int, int], np.ndarray],
) -> float:
    r"""Mismatch of the summed waveform, maximised over a time shift *and*
    a genuine coalescence-phase shift --- mode :math:`(\ell, m)` rotated by
    :math:`e^{i m \phi_c}`, not a single global phase.

    The start-frequency dependence is only a different coalescence-phase
    convention, so this drives the mismatch from ``~1e-3`` back down to
    the ``(2,2)``-only floor (``~7e-8``).
    """
    m_values = np.array([m for (_, m) in modes])
    ref = sum(ref_modes.values())
    stack = np.stack([modes[k] for k in modes])  # (n_modes, n_freq)
    ref_norm_sq = np.abs(np.vdot(ref, ref))

    time_shifts = np.linspace(-0.02, 0.02, 4001)
    time_kernels = np.exp(2j * np.pi * np.outer(time_shifts, GRID))
    best = 0.0
    for phi_c in np.linspace(-np.pi, np.pi, 361):
        trial = (stack * np.exp(1j * m_values[:, None] * phi_c)).sum(axis=0)
        norm = np.sqrt(ref_norm_sq * np.abs(np.vdot(trial, trial)))
        overlaps = np.abs(time_kernels @ (np.conj(ref) * trial)) / norm
        best = max(best, np.max(overlaps))
    return 1.0 - best


def main() -> None:
    for label, modes in (
        ("(2,2) only", [(2, 2)]),
        ("(2,2) + (2,1) + (3,3) + (4,4)", [(2, 2), (2, 1), (3, 3), (4, 4)]),
    ):
        reference = teob(20.0, modes)
        print(f"\n{label}:  mismatch vs the 20 Hz-start waveform, in [40, 1500] Hz")
        for start in (18.0, 15.0, 12.0, 10.0):
            print(f"   initial_frequency = {start:4.1f} Hz :  "
                  f"{mismatch(reference, teob(start, modes)):.2e}")

    print("\neach higher mode on its own, 20 Hz vs 12 Hz start "
          "(time + phase optimised):")
    for mode in ((2, 1), (3, 3), (4, 4)):
        a, b = teob(20.0, [mode]), teob(12.0, [mode])
        print(f"   {mode} :  {mismatch(a, b):.2e}")

    all_modes = [(2, 2), (2, 1), (3, 3), (4, 4)]
    ref_modes = {m: teob(20.0, [m]) for m in all_modes}
    print("\nsummed HOM waveform, 20 Hz vs lower start, maximised over a time "
          "shift and a *coalescence phase* (per-mode e^{i m phi_c}):")
    for start in (18.0, 15.0, 12.0, 10.0):
        trial_modes = {m: teob(start, [m]) for m in all_modes}
        mm = mismatch_coalescence(ref_modes, trial_modes)
        print(f"   initial_frequency = {start:4.1f} Hz :  {mm:.2e}")


if __name__ == "__main__":
    main()
