r"""Minimal example: TEOBResumS higher-mode phases depend on the ODE
integration start frequency, in a band far above every mode's onset.

This is a self-contained reproducer for upstream (only ``EOBRun_module``).

Two frequency-domain TEOBResumS calls are made, *identical except for*
``initial_frequency``, and both are asked (via ``interp_freqs``/``freqs``)
to return the waveform on the same explicit 40--1500 Hz grid -- so there
is no interpolation on our side and every :math:`(\ell, m)` up to
:math:`(4,4)` has full support across the whole comparison band.

Findings for a q = 1.6, M = 2.8 :math:`M_\odot` BNS:

* the :math:`(2,2)`-only waveform is start-frequency independent to a
  mismatch of ``~3e-8`` (i.e. numerical noise);
* switching the higher modes on, the two waveforms differ by a mismatch
  of ``~1e-3`` -- which, for the *same* physical source evaluated in the
  *same* band, should be zero;
* taken one mode at a time, the effect is a start-frequency-dependent
  *per-mode phase offset*: each higher mode alone still matches itself
  under a time-and-phase shift (``mm ~ 1e-7``), but the offset differs
  between modes, so a single coalescence phase cannot absorb it once
  they are summed. The :math:`(4,4)` additionally shows a small genuine
  phase-shape change.

The practical consequence is that a multi-mode TEOBResumS FD waveform is
only self-consistent at a fixed ``initial_frequency``; ``mlgw_bns``
sidesteps this by generating all of its training data from one low,
fixed start frequency and by carrying an explicit per-mode phase
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


if __name__ == "__main__":
    main()
