r"""Is the ~5e-4 FD validation floor resampling / quadrature, or a real
waveform difference?

``validate_extrinsic_against_teob.py`` evaluates the surrogate-vs-TEOBResumS
mismatch on the model's own decimated multiband grid (~1500 points in
[20, 2048] Hz, df up to ~9 Hz at the top), and it linearly interpolates the
independent ``EOBRunPy`` output onto that grid. Both the overlap-integral
quadrature and that interpolation are coarse.

Here the *same* comparison is run on a sequence of uniform grids of
decreasing df, with TEOBResumS asked (``interp_freqs``/``freqs``) to return
directly on each grid so there is no interpolation on our side. Two
references per grid:

* ``on-grid``  -- surrogate vs ``get_teob_modes_dict`` (the
  ``all_modes_amplitude_phase`` generator the surrogate is trained on),
  per-mode ``exp(i m phi_c)`` + time-shift marginalised. Grid-independent
  control: the surrogate is good here at ~1e-7 on any grid.
* ``independent`` -- surrogate vs a plain ``EOBRunPy(initial_frequency=15)``
  summed strain, same marginalisation.

If the ~5e-4 is quadrature/interpolation it collapses toward the on-grid
number as df shrinks. If it plateaus it is a genuine configuration
difference between the two TEOBResumS call paths.

Result (4 fixed binaries, df 0.5 -> 0.03 Hz):

* both mismatches are flat to 3 significant figures across a 16x
  refinement of df, and identical to the decimated model grid --
  quadrature and interpolation cost nothing. NOT resampling.
* on-grid (surrogate vs its training generator) is 4e-8 -- 7e-6 on every
  grid: the surrogate is excellent and grid-independent.
* independent (surrogate vs plain EOBRunPy) is 1.7e-5 -- 8.7e-3,
  strongly binary-dependent (worst at high q, high total mass).

Follow-up isolation (not in this script):

* it is entirely the m > 2 modes -- (2,2) alone agrees between the two
  TEOBResumS paths to 1.4e-8; adding (2,1)+(3,3) breaks it to 8.7e-3;
  (4,4) then adds nothing.
* flat under a band cut from 2048 to 512 Hz -- not the SPA high-frequency
  tail, not the (l,m) merger region.
* NOT the early ODE start or the srate_interp x2: forcing
  ``get_teob_modes_dict`` to f0 = 15 Hz, srate x1 (byte-identical config
  to the plain call) leaves the gap at 8.7e-3.
* the (2,2) and (2,1) train/plain phase offsets are exactly m * phi_c (a
  harmless coalescence phase / azimuth convention); (3,3) and (4,4)
  carry a frequency-dependent residual of 0.1 -- 0.35 rad on top, which
  a coalescence phase cannot absorb.

Conclusion: the FD "floor" is a genuine (3,3)/(4,4) phase difference
between TEOBResumS's per-mode ``hflm`` arrays (what
``all_modes_amplitude_phase``, and hence the surrogate, is built on) and
its internally summed ``h+/hx``. Which of the two is physically correct
needs a third waveform model (SEOBNRv5HM / IMRPhenomXHM). The surrogate
is not the source of the gap.

Run with: python visualization/fd_grid_convergence.py
"""

from __future__ import annotations

import numpy as np

from mlgw_bns.higher_order_modes import Mode, mode_to_k
from mlgw_bns.model import Model
from mlgw_bns.model_validation import ValidateModel
from mlgw_bns.mode_model import ParametersWithExtrinsic

MODES = [Mode(2, 2), Mode(2, 1), Mode(3, 3), Mode(4, 4)]
M_VALUES = np.array([mode.m for mode in MODES])

BAND = (20.0, 2048.0)
F0_TEOB = 15.0
DF_LADDER = (0.5, 0.25, 0.125, 0.0625, 0.03125)

#: A handful of fixed binaries spanning the box (q, chi1, chi2, l1, l2, M, iota).
BINARIES = [
    dict(mass_ratio=1.1, chi_1=0.05, chi_2=-0.02, lambda_1=800.0,
         lambda_2=600.0, total_mass=2.8, inclination=1.0),
    dict(mass_ratio=1.8, chi_1=0.2, chi_2=0.1, lambda_1=400.0,
         lambda_2=300.0, total_mass=3.2, inclination=0.9),
    dict(mass_ratio=2.6, chi_1=0.3, chi_2=-0.3, lambda_1=200.0,
         lambda_2=1500.0, total_mass=2.4, inclination=1.3),
    dict(mass_ratio=2.2, chi_1=-0.1, chi_2=0.25, lambda_1=1200.0,
         lambda_2=900.0, total_mass=3.6, inclination=0.5),
]


def teob_independent(params: ParametersWithExtrinsic, grid: np.ndarray) -> np.ndarray:
    r"""Plain ``EOBRunPy`` summed ``h_+ - i h_x`` directly on ``grid``."""
    from EOBRun_module import EOBRunPy

    par = dict(
        q=params.mass_ratio, LambdaAl2=params.lambda_1, LambdaBl2=params.lambda_2,
        chi1=params.chi_1, chi2=params.chi_2, M=params.total_mass,
        distance=params.distance_mpc, inclination=params.inclination,
        initial_frequency=F0_TEOB, domain=1, use_geometric_units="no",
        use_spins=1, interp_freqs="yes", freqs=list(grid), output_hpc="no",
        arg_out="no", coalescence_angle=0.0,
        use_mode_lm=sorted({mode_to_k(mode) for mode in MODES}),
    )
    _f, re_hp, im_hp, re_hc, im_hc = EOBRunPy(par)
    hp = np.conj(np.asarray(re_hp) + 1j * np.asarray(im_hp))
    hc = np.conj(np.asarray(re_hc) + 1j * np.asarray(im_hc))
    return hp - 1j * hc


def fd_mismatch(reference, mode_arrays, grid, psd, max_dt=0.02) -> float:
    r"""1 - overlap of ``reference`` against ``sum_k S_k exp(i(2 pi f t_c +
    m_k phi_c))``, maximised over ``(t_c, phi_c)``."""
    weight = np.gradient(grid) / psd
    stack = np.asarray(mode_arrays)
    ref_w = np.conj(reference) * weight
    norm_ref = np.sqrt(np.abs(np.sum(np.conj(reference) * reference * weight)))

    t_grid = np.linspace(-max_dt, max_dt, 4001)
    kernels = np.exp(2j * np.pi * np.outer(t_grid, grid))
    best = 0.0
    for phi_c in np.linspace(-np.pi, np.pi, 241):
        h = (stack * np.exp(1j * M_VALUES[:, None] * phi_c)).sum(axis=0)
        norm_h = np.sqrt(np.abs(np.sum(np.abs(h) ** 2 * weight)))
        overlaps = np.abs(kernels @ (ref_w * h)) / (norm_ref * norm_h)
        best = max(best, overlaps.max())
    return 1.0 - best


def main() -> None:
    model = Model(modes=MODES, filename="mlgw_bns/data/default_hom")
    model.load()
    validator = ValidateModel(model.mode_models[Mode(2, 2)])
    model_grid = validator.frequencies
    mg_inside = (model_grid >= BAND[0]) & (model_grid <= BAND[1])

    grids = [("model-decimated", model_grid[mg_inside])]
    for df in DF_LADDER:
        grids.append((f"uniform df={df:g}", np.arange(BAND[0], BAND[1], df)))

    for b in BINARIES:
        params = ParametersWithExtrinsic(distance_mpc=100.0, reference_phase=0.0, **b)
        print(f"\nq={b['mass_ratio']}, M={b['total_mass']}, chi=({b['chi_1']},"
              f"{b['chi_2']}), iota={b['inclination']}")
        print(f"  {'grid':>20}  {'n':>7}  {'on-grid':>10}  {'independent':>12}")
        for name, grid in grids:
            psd = validator.psd_at_frequencies(grid)
            surro = model.predict_modes_dict(grid, params)
            surro_arrays = [surro[(m.l, m.m)] for m in MODES]

            teob_og = model.get_teob_modes_dict(grid, params)
            ref_og = sum(teob_og.values())
            mm_og = fd_mismatch(ref_og, surro_arrays, grid, psd)

            ref_indep = teob_independent(params, grid)
            mm_indep = fd_mismatch(ref_indep, surro_arrays, grid, psd)

            print(f"  {name:>20}  {len(grid):>7}  {mm_og:>10.2e}  {mm_indep:>12.2e}")


if __name__ == "__main__":
    main()
