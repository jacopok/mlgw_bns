# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
