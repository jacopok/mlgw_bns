# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `mlgw_bns.neural_network.KernelRidgeNetwork`, a kernel-ridge alternative to
    the multi-layer perceptron, selected by passing it as `nn_kind` to `Model`
    or `ModeModel`. On the (2,2) mode with 8192 training waveforms it reaches a
    median mismatch of 3.4e-9 against the network's 7.2e-6 --- a factor of two
    thousand --- and fits in twenty seconds rather than ten minutes.

    The accuracy of the surrogate under a fixed training budget is limited by
    the map from parameters to principal-component coefficients, not by the
    basis: the truncation floor at thirty components sits at 1.8e-10, four
    orders of magnitude below what the network reaches, and the network stops
    improving altogether beyond about 2048 training waveforms while the kernel
    keeps improving as `n**-2`. Part of the difference is that the network
    minimizes an unweighted mean squared error over targets divided by
    `max|x_i|` per component, which weights component `i`'s contribution to the
    residual by `s_i**-2` --- some nine orders of magnitude in favour of the
    *least* important component. Kernel ridge solves `(K + alpha I)^-1 y`
    separately per output and is therefore equivariant under rescaling each
    output, so that weighting, and `pc_exponent` with it, cannot affect it.

    Which backend a saved model used is recorded in its metadata, so loading
    picks the right one without being told.
- `reference_amplitude`, an option on `Model`, `ModeModel` and `Dataset`, which
    divides the EOB amplitude by the Post-Newtonian amplitude of one fixed
    parameter set --- the centre of the parameter ranges --- rather than each
    waveform's own. The (2,1) and (3,3) PN amplitudes have a deep minimum at a
    parameter-dependent frequency, and dividing by it there sends the ratio to
    twenty or sixty while the waveform does nothing remarkable, so a handful of
    training waveforms end up setting the normalization for all of them. On the
    (3,3) mode this is worth a factor of eighteen in mismatch and on the (2,1)
    a factor of two and a half; on the (2,2), whose PN amplitude has no such
    minimum, it is a 3% improvement, and on (4,4) an 8% degradation, since it
    is applied per-dataset rather than per-mode. Off by default.
- Per-mode defaults for `KernelRidgeNetwork`, in
    `mlgw_bns/data/kernel_ridge_defaults.json`, read by
    `Hyperparameters.default_kernel_ridge` and written by
    `HyperparameterOptimization.save_best_as_default`. `hyperparameter_optimization.py`
    and `optimize_n_hours.py` are now specific to `KernelRidgeNetwork`: the
    search is over its two hyperparameters (`kernel_gamma`, `kernel_alpha`)
    only, at a fixed 8192 training waveforms, and single-objective ---
    reconstruction accuracy in residual space, not the PSD-weighted mismatch
    used for the MLP search, since kernel ridge has no architecture to trade
    against training time.
- `visualization/train_comparison_models.py`, which trains matched models for
    the legacy and improved pipelines, sharing the waveform generation between
    the two so that the comparison isolates the regressor.
- `experiments/`, the study behind the two changes above: cached residuals, a
    surrogate whose every stage is a knob, and the sweeps that measured them.
    Nothing there is imported by the package.

- Higher-order-mode support: the model shipped with the package now reconstructs
    the (2,2), (2,1), (3,3) and (4,4) modes and sums them into the observer-frame
    polarizations.
- Progress bars for the (long) dataset generation and training stages, and
    logging of the memory footprint of the arrays being allocated.
- Scripts under `visualization/` to validate a trained model and the time-shift
    predictor, and to inspect the TEOBResumS modes, their PN residuals and the
    parameters discarded during training.
- Precessing waveforms: `mlgw_bns.precessing_model.PrecessingModel` twists the
    surrogate's co-precessing multipoles into the inertial frame, along the
    Euler angles obtained by integrating the PN spin-precession dynamics
    (`mlgw_bns.twist_waveform`). The twist is done in the frequency domain, with
    each multipole's angles looked up at its own stationary-phase orbital
    frequency. `check_aligned_spin_limit` asserts that it reduces to
    `Model.predict` when the in-plane spins vanish.
- `Model.coprecessing_modes_dict`, returning the bare multipoles
    `A exp(i phi) / eta` without any sky projection, which is what the twist
    needs. The surrogate, post-Newtonian and EOB sources of amplitude and phase
    now all go through one internal helper rather than three copies of the same
    loop.
- `visualization/plot_twisted_waveforms.py`, which checks the twist against its
    analytic limits and plots the PN angles and the resulting polarizations, and
    `visualization/precessing_mismatches.py`, which computes precessing-waveform
    mismatches over random source positions and inclinations.
- `visualization/validate_twist_against_teob.py`, which validates the twist
    against TEOBResumS itself: it takes the co-precessing multipoles from an
    aligned-spin run, twists them here, and compares against the inertial-frame
    multipoles of a generic-spin run of the same binary. The dominant multipoles
    agree to a few times 1e-5, the polarizations to 6e-5 on average, and the
    residual is shown to vanish linearly with the opening angle, and to be
    entirely in the Euler angles: fitting the three angles at each time
    reproduces TEOBResumS' multipoles to 2e-14, so the rotation itself is
    exact, and the recovered angles agree with the ones integrated here to
    8e-06 rad in the combination the multipoles constrain sharply and to 3e-04
    rad in the opening angle.
- `visualization/validate_precessing_against_teob.py`, which closes the loop by
    comparing the full frequency-domain polarizations of
    `PrecessingModel.predict` against the `h+`, `hx` that TEOBResumS returns for
    the same precessing binary, over random orientations. Feeding the same
    pipeline TEOBResumS' own co-precessing multipoles instead of the surrogate's
    isolates the cost of *modelling* the precession (the PN angles, the
    stationary-phase multipoles, the resampling) from the network reconstruction
    error, which comes out at ~1e-6: the networks are not what limits a
    precessing waveform. The precession model is, and it climbs with the opening
    angle -- driven by the PN spin-precession angles.
- `PrecessingModel.EulerAngles.reanchored`, and `reanchor=True` (the default) on
    `PrecessingModel.predict` / `predict_modes_dict`. The Euler angles are
    integrated against `M Omega_orb` as advanced by a 3.5PN energy-balance
    `dOmega/dt`, which runs fast through the late inspiral, so the frequency the
    angles are labelled with drifts from the true one. `reanchored` re-tabulates
    them against the time-frequency relation carried by the surrogate's own
    (2,2) phase -- an EOB-accurate map -- which halves the mismatch against
    TEOBResumS at moderate-to-large opening angles (`beta ~ 0.2` rad: `1e-1` to
    `5e-2`), with no effect on the aligned-spin limit and no retrain. The
    residual still scales with `beta`.
- `twist_waveform.integrate_pn_spin_precession(independent_variable=
    "orbital_frequency", omega_dot=...)` and
    `precessing_model.eob_orbital_frequency_rate` /
    `PrecessingModel.euler_angles(anchor_to_reference_phase=True)`: march the PN
    spin precession *against* orbital frequency, along a `dOmega/dt` taken from
    an accurate `(2,2)` phase rather than the PN flux -- the surrogate analogue
    of the `SPIN_FLX_EOB` hand-off TEOBResumS does once its spin dynamics
    reaches the EOB band, and a way to fix the angle *trajectory* rather than
    just re-label it. Investigated because the `reanchored` residual scales with
    `beta`; over 10 binaries it came out a wash with `reanchored` (median
    mismatch `4.2e-3` vs `3.6e-3`), so the orbital-frequency evolution is not
    what limits a precessing waveform. Kept as an opt-in; all defaults
    unchanged. The shared PN precession r.h.s. was factored into
    `_pn_precession_derivatives`, with `_pn_spin_precession_rhs` now a thin
    time-domain wrapper (no behaviour change).
- Odd-m mode regressors are now weighted by each training waveform's integrated
    mode power, following arXiv:2609.03025: near equal mass the (2,1) and (3,3)
    amplitude residuals grow linearly out of the odd-m zero, a boundary layer
    the global-RBF kernel cannot resolve and rings across. Weighting by
    `(int A^2 df / max)^beta` for odd m only (`power_weighting`,
    `power_weight_exponent` on `ModeModel`, on by default, recorded in the
    metadata) improves the (2,1) optimised per-mode mismatch by 23x and (3,3) by
    11x on `default_hom`, with (2,2) and (4,4) bit-identical.

### Changed

- `twist_waveform.compute_hpc` returned `h+ + i hx` rather than `h+ - i hx`;
    with the sign corrected it reproduces TEOBResumS' polarizations to machine
    precision, given that the C code's `coalescence_angle` is the azimuth
    measured as `pi/2 - phi`. `PrecessingModel` was unaffected: it combines the
    multipoles through `polarizations_from_inertial_modes` instead.
- `twist_waveform.integrate_pn_spin_precession` now refines its output with the
    integrator's dense output. DOP853 covers a whole inspiral in a couple of
    hundred steps, which is accurate at those points but far too coarse to
    interpolate the precession between them.
- The Wigner d-function and the spin-weighted spherical harmonics now live only
    in `mlgw_bns.special_func`, vectorized over the angle;
    `higher_order_modes.wigner_d_function_spin_2`, the named harmonics in
    `mlgw_bns.spherical_harmonics` and `mlgw_bns.twist_waveform` all delegate to
    it instead of carrying their own copy.
- **Breaking**: `Model` is now the multi-mode surrogate, holding one `ModeModel`
    per spherical-harmonic mode. What used to be called `Model` --- the single-mode
    workhorse --- is now `ModeModel`, and lives in `mlgw_bns.mode_model`.
    The class previously called `ModesModel` is now `Model`, in `mlgw_bns.model`.
    Its `models` mapping is now called `mode_models`.
- **Breaking**: `Model.predict` returns the two polarizations `(hp, hc)`,
    like `ModeModel.predict`, instead of `(h, hp, hc)` where the first element
    was the redundant combination `hp - 1j * hc`.
- The `time_shifts` argument of `Model.predict` and `Model.predict_modes_dict`
    is now optional: if it is not given, the shifts aligning the mode mergers
    are predicted from the source parameters with `Model.time_shifts_predictor`.
- `Model.default_for_testing()` loads the higher-order-mode model
    (`mlgw_bns/data/default_hom`) rather than the old single-mode checkpoints.
- TEOBResumS is now taken from PyPI rather than from a checkout expected to sit
    next to this repository.
- The project is built and developed with [uv](https://docs.astral.sh/uv/)
    instead of poetry.
- **Breaking**: amplitude residuals are now stored and learned as
    `A / A_PN` rather than `log(A / A_PN)`; datasets and models saved with the
    previous convention cannot be reused.
- The multibanded frequency grid now accounts for the mode being represented:
    the seglen is scaled by `(m / 2) ** (8 / 3)`, and the safety margin on it
    went from 5% to 15%. All modes are trained on the same, finest, (4,4) grid,
    so that no cross-grid interpolation is needed when combining them.
- CI and tox run on Python 3.12 and 3.13 (previously 3.8 to 3.10), through uv.

### Fixed

- `Model.predict` did not rescale the mode time shifts, which are stored in
    units of the reference total mass of the dataset, to the total mass being
    requested, while `Model.predict_modes_dict` did: the two therefore
    disagreed for any total mass other than the reference one.
- The cubic spline used to go from the downsampled nodes back to the full
    frequency grid no longer extrapolates: points outside the node range are
    held at the nearest endpoint value. The outermost interval of the greedy
    downsampling can be orders of magnitude narrower than the extrapolation
    distance, in which case the extrapolated values diverged wildly.
- Mismatch computations no longer fall back to the value 1 whenever the
    L-BFGS-B refinement reports an abnormal termination, which happens
    routinely when the optimum sits at a bound of the periodic `phi_c`:
    the better of the refined and grid-search estimates is used instead.
- `SklearnNetwork.fit` clipped the mini-batch size to `x_data.shape[1]` --- the
    number of *features*, which is five --- rather than `shape[0]`, the number
    of samples, so the configured `batch_size` never survived at any
    training-set size and every packaged model was trained with a batch of five.
    The constructor already clipped correctly, to the sample count, so the
    second clip was pure slip. Set `Hyperparameters.legacy_batch_size_clip` to
    reproduce the old behaviour exactly.

    Repairing it does not by itself improve accuracy --- at 8192 waveforms the
    network scores 9.7e-6 against the slip's 7.2e-6, which is inside its own
    run-to-run scatter, and it now needs more iterations to converge because
    larger batches mean fewer gradient steps per iteration. It is fixed because
    until it was, no tuning of `batch_size` meant anything.
- `tests/test_downsampling_interpolation.py` asserted a generator expression,
    `assert (err < 1e-5 for err in errs_amp)`, which is a truthy object whatever
    it would yield --- so the reconstruction error was never checked. The errors
    are in fact of order 5e-4 and would have failed that bound; the test now
    compares them against the downsampling tolerances they are actually set by.

### Removed

- **Breaking**: the `default` and `fast` single-mode pretrained checkpoints, and
    with them `ModeModel.default_for_testing`.


## [0.12.1] - 2022-11-01

### Fixed

- Fixed [#46](https://github.com/jacopok/mlgw_bns/issues/46), an issue with the wrong version of joblib leading to models not being able to be loaded.

## [0.12.0] - 2022-10-15

### Added

- New functionality for [multiple default models](https://github.com/jacopok/mlgw_bns/pull/45)
    - two models available: the `default` one and a `fast` one, trained from 5 and 15Hz respectively.
- `extend_with_post_newtonian` and `extend_with_zeros_at_high_frequency` flags for the `Model` class,
    which determine whether to raise an exception or not when extending the model beyond its
    training frequency range.

### Changed

- The `flatten_phase` method of the `Residuals` dataclass now returns the timeshifts 
    which the waveforms were shifted by, instead of `None`
- Call signature for the `Model.default` classmethod: now, the first available argument 
    is `model_name`, which determines which of the default provided models to use;
    the keyword argument to use to choose the name to give to the current model is `filename`.

### Fixed

- Amplitude connection at low frequency: there is typically a (<1%) discrepancy in the EOB vs. 
    Post-Newtonian amplitude at the low frequency bound. Now, at frequencies lower than the minimum one,
    the amplitude varies continuously, and reaches its PN value at half of the minimum frequency.

## [0.11.0] - 2022-09-19

### Added

- Possibility to extend waveform evaluation to arbitrarily low frequencies, using the 
    post-Newtonian expressions. 
- Mention of this changelog in the README
- Reference documentation about the mathematical details of higher order modes
- Removed dependence on `pycbc` for PSD computations (see [this PR](https://github.com/jacopok/mlgw_bns/pull/38)): 
    this significantly decreases the dependency load of the package
- Also saving metadata with each saved model - this means the model does not rely on the settings
    used being the same as when the model was generated. 
    Metadata is saved as a human-readable yaml file.
- New convenience classmethod, `ParametersWithExtrinsic.gw170817()`, to get some quick parameters

### Removed

- Python 3.7 support

### Changed

- Standard model is now trained with `sklearn` version 1.1.2.

## [0.10.2] - 2022-07-01

### Fixed

- Improve evaluation speed, by reducing downsampled array size (set tolerance to 1e-5)
    - now the speeds, going down to 5Hz, are the same as those we had for 20Hz
- Improve test execution speed (in `tests/test_model.py`)

### Added

- Test profiling availability

## [0.10.1] - 2022-06-30

### Added

- Changelog!
- Some badges in the README:
    - coverage report with [coveralls](https://coveralls.io/)
    - downloads per month

### Changed

- Default model given now starts from 5Hz

### Fixed

- PCA now uses SVD
- Fix TEOB call error, which occurred when the integration time exceeded 1e9M
- Fix `ValidateModel` frequency arrays
- Various fixes to tests

[Unreleased]: https://github.com/jacopok/mlgw_bns/compare/v0.12.1...HEAD
[0.12.1]: https://github.com/jacopok/mlgw_bns/compare/v0.12.0...v0.12.1
[0.12.0]: https://github.com/jacopok/mlgw_bns/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/jacopok/mlgw_bns/compare/v0.10.2...v0.11.0
[0.10.2]: https://github.com/jacopok/mlgw_bns/compare/v0.10.1...v0.10.2
[0.10.1]: https://github.com/jacopok/mlgw_bns/compare/v0.10.0...v0.10.1
