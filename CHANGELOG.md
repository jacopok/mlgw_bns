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

### Changed

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
