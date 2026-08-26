# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Batched, per-mode evaluation, on numpy or JAX from a single implementation
    (`mlgw_bns.batched`). `Model.predict_modes_amp_phase(intrinsic,
    total_mass, frequencies, modes=..., distance_mpc=..., return_tf=...)`
    evaluates `N` binaries in one call --- `intrinsic` of shape `(N, 5)`,
    `total_mass` of shape `(N,)`, frequencies shared or one grid per row ---
    and returns the amplitude and phase of each requested mode, shape
    `(N, n_modes, k)`, with no angular factor applied. Only the requested
    modes' regressors are evaluated. `Model.jax_modes_amp_phase(modes,
    return_tf)` returns the same computation as a pure JAX function (a single
    waveform is the batch `N = 1`). Both include the post-Newtonian
    continuation below the trained band and the zero padding above it.
    About 0.7 ms per waveform for the four modes (2,2), (2,1), (3,3), (4,4)
    on numpy (batches of 1000 on 4 cores), against ~14 ms for
    `predict_modes_dict`; the JAX version compiles in under 10 s.
- `return_tf=True` also returns each mode's time--frequency map
    `t_lm(f) = -(1/2 pi) d phi_lm / d f`, from the derivative of the phase
    spline in the band and of the post-Newtonian phase below it.
- `mlgw_bns.batched.mode_polarizations`, the per-mode projection on
    `h_+, h_x` with the spin-weighted spherical harmonics, batched and
    numpy/JAX-generic; summed over the modes it reproduces `Model.predict`.
- Out-of-range rows come back as NaN instead of raising, so that one bad row
    does not abort a batch; `BatchedSurrogate.valid` gives the mask.
- `Model.parameter_ranges`, whose setter applies new ranges to every mode.

### Changed

- `mlgw_bns.jax_predict.model_to_jax_waveform` is now a thin wrapper around
    the batched pipeline instead of a separate port; its internal helpers
    (`mode_model_to_jax_residuals`, `make_not_a_knot_spline_jax`, ...) are
    gone. It compiles in seconds rather than ~40 s.
- `pn_modes.reference_phase_backbone` is vectorised over the parameter rows
    (shared with the batched path); its output changes at the ~1e-7 rad level.

### Fixed

- A `Model` loaded with a subset of the trained modes, e.g.
    `default_for_testing(modes=[(2,2), (2,1), (3,3), (4,4)])`, read the
    mode-phases predictor's columns by position in its own mode list rather
    than the predictor's, mis-phasing (3,3) and (4,4) by O(1) rad.

### Known issues

- The kernel-ridge regressors of the shipped model have dual coefficients up
    to ~1e13, and their prediction is a sum that cancels by some fourteen
    orders of magnitude. Its floating-point rounding error is therefore
    visible: evaluating the same parameters in a batch or one at a time, on
    numpy or on JAX, or at parameters differing by one part in 1e12, changes
    the (2,2) and (4,4) phases by up to ~1e-2 rad near the merger. In a
    likelihood this is a noise floor (O(0.1--1) in ln L at SNR ~100, growing
    as SNR^2). Evaluating the sum in 80-bit precision removes it (6e-6 rad)
    but costs ~20x; the lasting fix is a better-conditioned regressor, which
    needs retraining.

## [1.0.1] - 2026-09-21

### Fixed

- `.gitignore` patterns meant to exclude stale root-level copies of the
    default model's (2,1), (2,2), (3,3) and (4,4) mode configs were unanchored,
    so they also matched (and were excluded by `hatchling`'s VCS-based sdist
    build) the real, tracked copies under `mlgw_bns/data/`. The 1.0.0 release
    tarball was missing those four modes' `.yaml` files; the patterns are now
    anchored to the repository root.

## [1.0.0] - 2026-09-21

### Added

- Higher-order-mode support: the model shipped with the package now
    reconstructs seven modes --- (2,2), (2,1), (3,1), (3,2), (3,3), (4,3) and
    (4,4) --- and sums them into the observer-frame polarizations, weighted by
    the spin-weighted spherical harmonics.
- `Model.default_for_testing` takes an optional `modes` argument, to load only
    a subset of the shipped modes instead of all seven.
- `mlgw_bns.neural_network.KernelRidgeNetwork`, a kernel-ridge alternative to
    the multi-layer perceptron, selected by passing it as `nn_kind` to `Model`
    or `ModeModel`. On the (2,2) mode with 8192 training waveforms it reaches a
    median mismatch of 3.4e-9 against the network's 7.2e-6 --- a factor of two
    thousand --- and fits in twenty seconds rather than ten minutes. Kernel
    ridge solves `(K + alpha I)^-1 y` separately per output and is therefore
    equivariant under rescaling each output, unlike the network's mean squared
    error, which implicitly weights components by their PCA scale. Which
    backend a saved model used is recorded in its metadata, so loading picks
    the right one without being told. Per-mode hyperparameter defaults live in
    `mlgw_bns/data/kernel_ridge_defaults.json`; `HyperparameterOptimization`
    tunes the two of them (`kernel_gamma`, `kernel_alpha`) directly against
    reconstruction accuracy in residual space.
- `reference_amplitude`, an option on `Model`, `ModeModel` and `Dataset`, which
    divides the EOB amplitude by the Post-Newtonian amplitude of one fixed
    parameter set --- the centre of the parameter ranges --- rather than each
    waveform's own. The (2,1) and (3,3) PN amplitudes have a deep minimum at a
    parameter-dependent frequency, which was letting a handful of training
    waveforms set the normalization for all of them; worth a factor of
    eighteen in mismatch on (3,3) and two and a half on (2,1). Off by default.
- Odd-`m` mode regressors are weighted by each training waveform's integrated
    mode power (`power_weighting`, `power_weight_exponent` on `ModeModel`, on
    by default), following arXiv:2609.03025 (DANSur_HM): at exactly `q = 1` the
    odd-`m` amplitude vanishes identically, and the boundary layer around it
    was otherwise too steep for the global-RBF `KernelRidgeNetwork` to fit.
    Median mismatch improves 22x on (2,1) and 8.7x on (3,3).
- `mlgw_bns.jax_predict`, an experimental JAX port of the prediction pipeline
    (`jax` optional-dependency extra), adapted from Saulo Albuquerque's
    `mlgw-bns-jax`. `model_to_jax_waveform(model)` returns a pure,
    `jax.jit` / `jax.vmap`-able function reproducing the full `Model.predict`,
    including the low-frequency PN splice and high-frequency zero-padding.
    Agrees with the numpy pipeline to ~1e-4 relative; roughly 1.5x faster than
    numpy for a single waveform and 5--15x faster batched with `jax.vmap`.
- Progress bars for the (long) dataset generation and training stages, and
    logging of the memory footprint of the arrays being allocated.
- Various new scripts under `visualization/`, checking the surrogate's
    extrinsic-parameter handling, its low- and high-frequency extensions, and
    its mismatch against total mass and frequency-grid choices, against
    independent TEOBResumS calls.

### Changed

- `Model.predict` is about 2.5x faster at the fixed (grid-size independent)
    cost, which dominates for the frequency grids used in parameter
    estimation (~20 ms down to ~8 ms). Four changes, all bit-for-bit identical
    in output:
    - the shared per-mode reference-phase predictor
        (`ModeModel._predicted_mode_phase0`) was evaluated once per mode even
        though one call returns every mode's phase --- it is now cached on
        the shared predictor and evaluated once per waveform;
    - `ModeModel.predict_amplitude_phase{,_optimized}` run their
        scikit-learn `predict` calls under `assume_finite` /
        `skip_parameter_validation` (scoped to the call), since the input is
        a single already-clean parameter row and the default per-call
        validation cost more than the regression it guards;
    - `KernelRidgeNetwork.predict` (the parameters-to-coefficients map, run
        once per mode) evaluates the fitted RBF kernel and the two
        standardizations directly instead of through
        `sklearn.kernel_ridge.KernelRidge.predict`, which re-validates the
        whole training matrix on every call; the floating-point operation
        order is matched to scikit-learn's so the result is unchanged;
    - `Dataset._frequencies` / `_frequencies_hz` are cached with
        `maxsize=None` rather than `maxsize=1`: a multi-mode `Model` holds
        one `Dataset` per mode and they were evicting each other, so the
        ~519k-point grid was reconverted to natural units once per mode on
        every call.
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
- Multi-mode training-data generation issues one TEOBResumS call per parameter
    point for every requested mode, instead of one call per (mode, point) pair.
- TEOBResumS is now taken from PyPI rather than from a checkout expected to sit
    next to this repository. Its packaged root-finder is less robust than a
    local, unreleased fix at high tidal deformability (see *Fixed*), so
    dataset generation and validation degrade gracefully by skipping a failed
    draw instead of requiring it.
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

- `ParametersWithExtrinsic.reference_phase` is the coalescence phase, so
    shifting it by `phi_c` must rotate the `(l, m)` mode by `exp(i m phi_c)`;
    `ModeModel.predict_amplitude_phase{,_optimized}` added the *same*
    `reference_phase` to every mode instead. For a single-mode `(2,2)` model
    this was a harmless convention (a factor of two on a rarely-used
    parameter); for a multi-mode waveform a flat phase is not a physical
    rotation, so `reference_phase` did not do what it says. Now applied as
    `m * reference_phase` (a mode-less model is the `(2,2)`, so `m = 2`).
    `time_shift`, a genuine time-domain shift, stays the same for every mode.
    **Breaking** for anyone who passed a non-zero `reference_phase` to a
    `(2,2)` `ModeModel` and expected it verbatim --- multiply by two.
- The post-Newtonian low-frequency extension in
    `ModeModel.predict_amplitude_phase{,_optimized}` glued the model band onto
    the PN segment by shifting the *band* to match the PN phase at the
    connection, which overwrote each mode's per-mode phase constant
    (carrying the inter-mode alignment) with the PN one. For a
    higher-order-mode waveform this mis-phased the modes relative to each
    other by a constant the coalescence-phase-marginalised full-waveform
    mismatch could not remove: a ~1000x degradation for every `total_mass`
    below the dataset reference (2.8 Msun), the only regime in which the
    extension fires. The fix shifts the PN *segment* to match the band
    instead, leaving the band's phase --- and its per-mode constant ---
    untouched.
- `Model.predict`/`predict_modes_dict` anchored each mode's time-shift-induced
    linear phase to the first element of whatever frequency array the caller
    passed in, rather than a fixed physical reference; as long as every caller
    queried starting exactly at the trained band's edge this was invisible,
    but querying into the post-Newtonian low-frequency extension exposed it as
    a query-grid-dependent phase shared coherently by every mode. Now anchored
    to `dataset.effective_initial_frequency_hz`, matching the convention
    `ModeModel.predict_amplitude_phase_optimized` already used internally.
- Dataset generation and `ValidateModel` no longer abort an entire batch when
    one EOB draw fails root-finding, which the packaged TEOBResumS does
    routinely at high tidal deformability (up to ~22% of draws at
    `lambda_max=12000`); the failed draw is now skipped and logged instead.
- A multi-mode frequency-domain TEOBResumS call's low-frequency support
    depends on `initial_frequency` (the (2,2) GW frequency at the ODE start):
    each `(l, m)` multipole is identically zero below `(m/2) * f0`, so training
    data generated from an ODE start that isn't lowered per mode is missing
    higher-mode content just above the nominal band edge. Dataset generation
    now lowers the ODE start by `initial_frequency_scaling(modes)`.
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
    training-set size and every packaged model was trained with a batch of
    five. The constructor already clipped correctly, to the sample count, so
    the second clip was pure slip. Set `Hyperparameters.legacy_batch_size_clip`
    to reproduce the old behaviour exactly. Repairing it does not by itself
    improve accuracy at a fixed training-set size, but until it was fixed no
    tuning of `batch_size` meant anything.
- `tests/test_downsampling_interpolation.py` asserted a generator expression,
    `assert (err < 1e-5 for err in errs_amp)`, which is a truthy object whatever
    it would yield --- so the reconstruction error was never checked. The test
    now compares against the downsampling tolerances it is actually set by.

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
