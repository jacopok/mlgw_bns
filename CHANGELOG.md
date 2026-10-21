# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `Model.predict` and the other single-binary predictions keep BLAS on one
    thread: the regressors' products are too small for a multithreaded BLAS,
    which spent far longer synchronizing its threads than computing them,
    and slowed the rest of the call down with them. A prediction of the
    shipped model at 4000 frequencies takes ~13 ms instead of ~100; the
    waveforms are bit-for-bit unchanged.

### Documentation

- The overview lists the shipped models and how to load and download them,
    and shows how to generate precessing waveforms; the installation guide
    points to the fork of TEOBResumS the models are trained with and to the
    `jax` extra. The profiling guide, the TEOBResumS comparison in the
    precession page and the README's mismatch plot are updated to the
    2.0 models.

## [2.0.1] - 2026-10-21

### Changed

- `pytest-cases` is no longer a dependency: it breaks on pytest 9, and the
  test fixtures it parametrized are now plain pytest ones. It was also listed
  as a runtime dependency, which it never was.
- The development dependencies require pytest 9.0.3 or newer and lock uv
  0.13.4, following security advisories.
- Dropped the unused dependencies: `toml`, `setuptools` and the type stubs
  from the runtime ones (`types-PyYAML` is now a development dependency),
  and tox (with `tox.ini`), `pytest-cov`, `pylint`, `mutmut`, `scalene`,
  `colorama`, `click` and `pytest-profiling` from the development ones.

## [2.0.0] - 2026-10-20

This release breaks compatibility with 1.0: models saved by 1.0 cannot be
loaded, and the waveforms are referenced differently in time and phase.

### Changed

- **The time-shift and mode-phase regressors are gone.** The phase residuals
    of every training waveform are now referenced, all modes together, to its
    (2,2) mode at the lowest frequency of the band, `f0`: every mode is
    shifted in time and rotated in orbital phase so that the (2,2) residual
    and its slope vanish there (`data_management.reference_gauge` and
    `re_reference`). Previously each waveform kept TEOBResumS's reference ---
    time zero at the merger, orbital phase zero where the integration starts
    --- so each mode's phase at `f0` carried 2 pi f0 times the time to the
    merger, 1e4--1e6 rad and strongly parameter dependent, which the shared
    `TimeshiftsNN` and `ModePhasesNN` regressors were there to predict. In
    the new reference the stationary-phase relation between the modes,
    which the post-Newtonian phases the residuals are taken against already
    contain, leaves each mode's residual at `f0` a constant up to the
    difference between the EOB and post-Newtonian phases at orbital
    frequencies `f0/m` and `f0/2`. Over 96 binaries (public TEOBResumS,
    tidal deformabilities up to 5000) the standard deviation of the
    residual at `f0` goes from 0.9--3.6e5 rad with the merger at `t = 0`
    (0.4--1.3e5 rad in TEOBResumS's own reference) to

    | mode | std at `f0` |
    |------|-------------|
    | (2,2) | 7e-7 rad |
    | (3,2) | 3e-6 rad |
    | (3,1), (3,3), (4,3) | 1--2e-3 rad |
    | (4,4) | 3e-3 rad |
    | (2,1) | 2e-3 rad on 95 of 96; one point, at q = 1.04, off by pi, where the EOB amplitude has the opposite sign to the PN one |

    (`visualization/phase_reference_study.py anchors`). These are learned
    with the rest of the residual by each mode's PCA and regressor. No
    change to the TEOBResumS call is needed, and the reference does not
    depend on how a given TEOBResumS version aligns its merger.
- **Merger time and coalescence phase.** The surrogate is referenced to the
    merger, read off its own (2,2) mode: after the merger the
    frequency-domain (2,2) phase is linear in `f` (the stationary-phase
    time stops at the merger), from Mf ~ 0.016 for Lambda ~ 5000 to ~ 0.036
    for Lambda ~ 5, within the band (which ends at Mf = 0.0403). The tangent
    to the (2,2) phase at the top of the band, `s f + b`
    (`ModeModel.merger_reference`, `Model.merger_reference`), gives the
    merger time `-s / 2 pi` and the (2,2) phase `b` there; every mode is
    shifted by `-s f - (m / 2) b`, so that the merger is at `t = 0` with
    coalescence phase zero, and `t_lm(f)` (`return_tf`) vanishes at the top
    of the band. This merger is within ~5 M of TEOBResumS's amplitude peak.
    The batched path evaluates the (2,2) for this even when it is not
    requested.
- `ParametersWithExtrinsic.reference_phase` and `.time_shift` are now
    `coalescence_phase` (the orbital phase at the merger: mode `(l, m)` is
    rotated by `exp(i m phi_c)`) and `merger_time` (in seconds; the phase
    gets `- 2 pi f t_c`, the opposite sign to the old `time_shift`).
    `jax_predict.model_to_jax_waveform` takes `coalescence_phase` and
    `merger_time`.
- A `Model` must include the (2,2) mode.
- `Model.get_teob_modes_dict` is referenced to the merger in the same way,
    so it can be compared with `predict_modes_dict` directly, and is in the
    same (physical) units; the new `Model.teob_modes_amp_phase` gives the
    underlying amplitudes and phases. `ValidateModel.merger_referenced`
    does the same for waveforms at the downsampling nodes, and the
    single-mode mismatches use it.
- `Model.generate` has no reference pre-pass (and no `reference_*`
    arguments): one EOB sweep fewer.
- `Model.generate` evaluates the post-Newtonian baselines only at the
    downsampling nodes that are kept (and the (2,2) phase in the reference
    window), rather than on the whole grid for every mode: 1.06 to 0.40 s
    a training waveform, with residuals identical to the bit.
- TEOBResumS (a development dependency) is installed from the
    `initial-spa-fix` branch of `github.com/jacopok/teobresums`: the public
    release aborts on some of the training waveforms. Optuna 5 is required.
- `Dataset.generate_residuals` has no `flatten_phase` argument: the
    residuals of a single mode are referenced to their own value and slope
    at `f0`. `PrincipalComponentTraining` takes no predictors.
- **Two pretrained models are shipped** (`MODELS_AVAILABLE`), both
    retrained in the new format with `KernelRidgeNetwork` regressors on
    16384 waveforms from the TEOBResumS fork, with the same ranges of
    mass, mass ratio and spins as before:

    | model | tidal deformabilities | full-waveform mismatch, median / 90% | merger time error, median / 90% |
    |-------|-----------------------|--------------------------------------|---------------------------------|
    | `default_hom` (the default) | 5--2000 | 2.5e-8 / 4.6e-7 | 1.4 / 8.7 us |
    | `default_hom_lambda12000` | 5--12000 | 9.1e-7 / 9.8e-6 | 36 / 132 us |

    (mismatches optimised over time and phase on 16384 held-out waveforms,
    `visualization/modes_scale.py`; merger-time errors against TEOBResumS
    on 96 binaries below 2000 and 5000 respectively,
    `visualization/merger_reference_by_size.py`). At the same number of
    waveforms, a narrower range of tidal deformabilities is sampled more
    densely, and the merger time and coalescence phase the model predicts
    are what this limits (`visualization/merger_reference_range.py`).
    Kernel ridge regression at 16384 waveforms is fast to evaluate, which
    is why a narrower range rather than a larger training set is shipped.
    Load the wide one with
    `Model.default_for_testing("default_hom_lambda12000")`. The regressor
    files of a model are those named `<model>_l<l>_m<m>_nn.pkl`, so that
    one model's name can prefix another's. These are not comparable with
    the 1.0 model's test values, which were measured against a different
    TEOBResumS than it was trained with.

- `KernelRidgeNetwork` chooses its regularization per principal component
    (`Hyperparameters.kernel_alpha_selection = "loo"`, the new default): each
    output gets the ridge penalty minimizing its exact leave-one-out error
    plus the rounding error its dual coefficients cause when a prediction
    is evaluated, estimated as `0.1 * eps * sum_j |K(x, x_j) a_j|`
    (`kernel_rounding_factor`, calibrated on the packaged regressors). One
    eigendecomposition of the kernel gives both for a whole grid of
    penalties (`neural_network.kernel_ridge_leave_one_out`); at 24576
    training points it takes ~10x the time and ~2x the memory of the
    Cholesky solve it replaces. The fit is stored as an ordinary
    `KernelRidge` with a vector `alpha`, so prediction, the batched/JAX
    path and saved files are unchanged. `"fixed"` keeps the single
    `kernel_alpha` of the packaged models.

    Measured on 8192 freshly generated waveforms (public TEOBResumS,
    tidal deformabilities up to 5000; 7168 to train, 1024 held out;
    `visualization/kernel_regularization_study.py`), against the packaged
    per-mode penalties, held-out RMS errors in residual space:

    | mode  | phase, fixed | phase, per-component | amplitude, fixed | amplitude, per-component |
    |-------|--------------|----------------------|------------------|--------------------------|
    | (2,2) | 0.152 rad    | 0.093 rad            | 9.9e-3           | 6.2e-3                   |
    | (4,4) | 0.921 rad    | 0.593 rad            | 3.2e-2           | 6.4e-3                   |
    | (2,1) | 0.239 rad    | 0.165 rad            | 1.37e-2          | 1.42e-2                  |
    | (3,3) | 0.379 rad    | 0.264 rad            | 1.48e-2          | 1.54e-2                  |

    (the odd-`m` rows weighted by mode power, as those fits are; unweighted,
    their amplitude error grows, at the vanishing-amplitude waveforms near
    `q = 1`). The largest dual coefficients fall from 5e6--7e11 to
    1e4--2e6, and the spread of the predicted phase between a batch and
    single-row evaluations from up to 9e-5 rad to below 3e-9 rad. At this
    training size the leave-one-out optimum is already at penalties of
    1e-7 or more, so the rounding term does not bind; it is there for the
    full-size (24576-point) fits, whose errors are smaller. The absolute
    errors here are well above the packaged model's --- a third of its
    training data, and the public TEOBResumS --- only the comparison is
    meaningful.
- `HyperparameterOptimization` (and `optimize_n_hours.py`) then only search
    `kernel_gamma`, in a separate `<filename>_loo_study.pkl`;
    `kernel_alpha_selection="fixed"` (`--fixed-alpha`) restores the search
    over one shared `kernel_alpha`, which is how the packaged (2,2) penalty
    ended up at 3.9e-14 --- numerically no regularization at all, and the
    source of the rounding noise described under Known issues.
- `mlgw_bns.jax_predict.model_to_jax_waveform` is now a thin wrapper around
    the batched pipeline instead of a separate port; its internal helpers
    (`mode_model_to_jax_residuals`, `make_not_a_knot_spline_jax`, ...) are
    gone. It compiles in seconds rather than ~40 s.

### Removed

- `neural_network.TimeshiftsNN`, `TimeshiftsGPR`, `ModePhasesNN` and their
    loaders; `pn_modes.reference_phase_backbone`;
    `principal_component_analysis.remove_linear_trend`;
    `Residuals.flatten_phase` and `phase_timeshifts`; the `time_shifts`
    argument of `Model.predict` and `predict_modes_dict`;
    `BatchedSurrogate.time_shifts` and `mode_reference_phases`;
    `ModeModel.predict_amplitude_phase_optimized` (now
    `predict_amplitude_phase`, which takes the (2,2) `merger_reference`);
    the `include_time_shifts` arguments of `ValidateModel`; the
    visualization scripts that only studied the removed regressors.

### Fixed

- **The sign of `hx`.** `Model.predict`, `batched.mode_polarizations` (and
    so `jax_predict.model_to_jax_waveform` and the batched surrogate) and
    `PrecessingModel` returned `hx` with the opposite sign to TEOBResumS'
    and LALSuite's convention relative to `h+`: `hx/h+ = +0.836i` for the
    (2,2) at inclination 1 against `-0.836i` (the single-mode
    `ModeModel.predict` was right). That is the waveform seen from
    `pi - iota`, which mirrors the inclination posterior in multi-detector
    inference. The frequency-domain projection had been derived keeping
    the `m > 0` multipoles at positive frequencies, which holds in the
    `exp(+2 pi i f t)` Fourier convention of TEOBResumS' frequency-domain
    output, not in the `exp(-2 pi i f t)` one of the model's polarizations
    (`docs/explanation/higher_order_modes.md`). No comparison could see it:
    the tests compared `h+` and `hx` with TEOBResumS' each on its own, by
    a mismatch maximised over a constant phase, which absorbs the sign;
    the full-waveform mismatch builds both sides' polarizations from
    multipoles in the same convention, and maximises over a polarization
    angle and the absolute value of the overlap, either of which absorbs
    it; and for aligned spins negating `hx` is the same as `iota -> pi -
    iota`, so no single-detector mismatch can see it at all. The
    precessing test against TEOBResumS' `hx` compared it with `-hc`. Now
    `h+ + i hx`, not `h+ - i hx`, is the sum of the multipoles weighted by
    the spherical harmonics, and `Model.predict_modes_dict` and
    `get_teob_modes_dict` return each mode's contribution to it (the same
    arrays as before). The
    trained models are unchanged.

- `Model.get_teob_modes_dict` returned amplitudes without the physical
    prefactor (off by ~1e-27 relative to `predict_modes_dict`).

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
- Training at scale, on a SLURM cluster (`slurm/modes`,
    `visualization/modes_scale.py`; see the "Training at scale" guide):
    training sets in shards on disk made by any number of jobs
    (`mlgw_bns.sharding`, `mlgw_bns.modes_dataset.ShardedModesDataset`),
    principal components accumulated a shard at a time
    (`principal_component_analysis.CovarianceAccumulator`),
    `ModeModel.fit_nn` and `Model.train_downsampling` split out of the
    training, and `model_validation.stored_waveform_mismatches`, which
    validates a model against stored waveforms without calling TEOBResumS.
    `slurm/modes/release.env.example` trains and validates the shipped
    model.
- The regressors of the pretrained model (`data/default_hom_*_nn.pkl`) are
    no longer in the package (1.5 MB instead of ~50 MB): `Model.default_for_testing`
    downloads them on first use, checks them against the SHA-256 of the
    shipped manifest `data/default_hom_files.json`, and caches them
    (`$MLGW_BNS_CACHE`, else `~/.cache/mlgw_bns`); in a checkout of the
    repository the files there are used. `python -m mlgw_bns.model_files
    download` fills the cache beforehand, for machines without internet
    access (`mlgw_bns.model_files`).
- `neural_network.JaxMLPNetwork`, a JAX perceptron (`mlgw_bns.jax_mlp`) as
    the regressor of a mode, also in the batched evaluation; and
    hyperparameter studies of the regressors shared by workers on many
    nodes (`hyperparameter_optimization.run_worker`, journal storage).

- **Precessing waveforms**: `mlgw_bns.precessing_model.PrecessingModel`
    twists the surrogate's co-precessing multipoles into the inertial frame
    along Euler angles integrated from the PN spin-precession equations
    (`mlgw_bns.twist_waveform`, 3.5PN + tidal `dOmega/dt`). The twist is done
    in the frequency domain, each multipole's angles looked up at its own
    stationary-phase orbital frequency; with no in-plane spins it reduces to
    `Model.predict`. `PrecessingParametersWithExtrinsic` gives the spin
    vectors at `reference_frequency_hz`, where the orbital phase is fixed to
    `reference_phase` (as `f_ref`, `phiRef` in LALSuite);
    `PrecessingModel.teob_reference_phase` converts TEOBResumS' phase origin.
    `PrecessingModel.predict` evaluates the JAX predictor below (the `jax`
    extra, and `reference_frequency_hz`, are required); the numpy
    implementation, `PrecessingModel.predict_numpy`, is kept for testing.
    Against TEOBResumS' precessing waveforms the mismatch is the same as the
    aligned-spin one of the same binaries (7.0e-8 median, 48 binaries x 4
    lines of sight, outside TEOBResumS' `alpha` steps, see Known issues).
    See `docs/explanation/precession.md` and
    `docs/examples/precessing_vs_teobresums.py`.
- **Batched precessing waveforms in JAX**: `PrecessingModel.jax_predict()`
    (`mlgw_bns.batched_precession`) is a pure JAX function of `N` binaries,
    which can be jitted, vmapped and differentiated. It integrates the
    precession against the orbital frequency with fixed DOP853 steps and
    matches `PrecessingModel.predict_numpy` to mismatches of 1e-11--3e-10; ~20 ms
    a waveform, alone or in a batch of 64 (CPU), against ~25 s for the
    numpy path and ~0.13 s for TEOBResumS. `PrecessingModel.jax_predict_modes`
    (`batched_precession.precessing_mode_components`) returns the
    co-precessing multipoles and their twist coefficients separately, for
    mode-by-mode relative binning, and `batched_precession.precession_angles`
    integrates the angles once for a frequency array evaluated in pieces.
    `special_func.wigner_d_function`/`spinsphericalharm` take `xp=` (numpy
    or `jax.numpy`).
- `Model.coprecessing_modes_dict` and
    `Model.coprecessing_amplitudes_and_phases`: the multipoles without the
    sky projection, which is what the twist needs.
- The LVK spin angles: `mlgw_bns.spin_conversion.lvk_to_precessing` (and
    `PrecessingParametersWithExtrinsic.from_lvk`) maps `theta_jn, phi_jl,
    tilt_1, tilt_2, phi_12, a_1, a_2, phase` at `f_ref` to the precessing
    parameters; `lal_precessing_spins`, a numpy/JAX port of LALSuite's
    `SimInspiralTransformPrecessingNewInitialConditions`, agrees with it to
    1e-11.
- `scipy` is now a declared dependency (it was used through scikit-learn's).
- `scikit-learn` is pinned to exactly 1.9.1, the version the packaged
  regressors are pickled with: any other version warns on every load.

### Known issues

- The kernel-ridge regressors of the 1.0 model have dual coefficients up
    to ~1e13, and their prediction is a sum that cancels by some fourteen
    orders of magnitude. Its floating-point rounding error is therefore
    visible: evaluating the same parameters in a batch or one at a time, on
    numpy or on JAX, or at parameters differing by one part in 1e12, changes
    the (2,2) and (4,4) phases by up to ~1e-2 rad near the merger. In a
    likelihood this is a noise floor (O(0.1--1) in ln L at SNR ~100, growing
    as SNR^2). Evaluating the sum in 80-bit precision removes it (6e-6 rad)
    but costs ~20x; the lasting fix is a better-conditioned regressor, which
    needs retraining.
- TEOBResumS' precession angle `alpha` has spurious ~2 pi steps, which its
    frequency-domain twist splines through, so its precessing `h+`, `hx`
    are wrong on the few frequency bins around each (up to ~4e-3 of
    mismatch): `prolong_euler_angles_FD` unwraps `alpha` with `unwrap_HM`
    rather than `unwrap_euler`. A defect of the reference, not of
    `PrecessingModel` (reported upstream), and fixed in the fork of
    TEOBResumS the development dependencies install, which the validation
    now needs: it no longer masks those bins.
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

[Unreleased]: https://github.com/jacopok/mlgw_bns/compare/v2.0.1...HEAD
[2.0.1]: https://github.com/jacopok/mlgw_bns/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/jacopok/mlgw_bns/compare/v1.0.1...v2.0.0
[1.0.1]: https://github.com/jacopok/mlgw_bns/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/jacopok/mlgw_bns/compare/v0.12.1...v1.0.0
[0.12.1]: https://github.com/jacopok/mlgw_bns/compare/v0.12.0...v0.12.1
[0.12.0]: https://github.com/jacopok/mlgw_bns/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/jacopok/mlgw_bns/compare/v0.10.2...v0.11.0
[0.10.2]: https://github.com/jacopok/mlgw_bns/compare/v0.10.1...v0.10.2
[0.10.1]: https://github.com/jacopok/mlgw_bns/compare/v0.10.0...v0.10.1
