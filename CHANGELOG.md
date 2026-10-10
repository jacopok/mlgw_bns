# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- `Dataset.generate_residuals` has no `flatten_phase` argument: the
    residuals of a single mode are referenced to their own value and slope
    at `f0`. `PrincipalComponentTraining` takes no predictors.
- The packaged `default_hom` model is an **interim** one, trained with the
    public TEOBResumS (tidal deformabilities up to 5000) on 8192 waveforms,
    in the new format; it is to be replaced by a retrain before release.
    Against the same TEOBResumS (16 binaries, total mass 2.8, inclination
    1), the full-waveform mismatch maximised over time and phase is 6.8e-8
    median and 4.0e-6 worst (1.5--3.7e-7 median at total masses 2.2--3.6).
    With nothing maximised --- the model's own merger time and coalescence
    phase --- it is 1.1e-3 median and 3.7e-2 worst: over 64 held-out
    binaries the merger time agrees with TEOBResumS's to 2.4e-5 s and the
    coalescence phase to 0.13 rad (90th percentiles;
    `visualization/phase_reference_study.py merger`). These are not
    comparable with the 1.0 model's test values, which were measured
    against a different TEOBResumS than it was trained with.

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
- `Model.generate`'s multi-mode TEOBResumS sweep is 2.6 times faster (1.06
    to 0.40 s a waveform for seven modes from 5 Hz): the post-Newtonian
    amplitude and phase the residuals are taken against are evaluated only
    at the downsampling nodes of each mode (and for the (2,2) phase in the
    window of the reference fit), not on all of the 5e5-point grid, which
    took two thirds of the time. The residuals are the same to the bit.

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
- The re-anchoring of the precession angles (`EulerAngles.reanchored`, the
    `reanchor=` argument of `PrecessingModel.predict` and
    `predict_modes_dict`, `EulerAngles.time`), never better than the plain
    lookup against TEOBResumS, and the scripts that studied it
    (`ab_precession_floor.py`, `recover_frequency_remapping.py`).
- The numpy precessing model from the timing benchmarks (~20 s a
    waveform, against ~70 ms for its JAX port).

### Fixed

- `PrecessingModel.jax_predict(modes=...)` read the orbital phase at the
    reference frequency from the twisted multipoles only, so the odd-m branch
    could differ from the numpy path's, which reads it from all of the
    model's.
- The docs' Markdown pages rendered `$...$` math as text (`myst_parser`'s
    `dollarmath` was not enabled).
- `Model.get_teob_modes_dict` returned amplitudes without the physical
    prefactor (off by ~1e-27 relative to `predict_modes_dict`).
- Below the reference frequency, every odd-m co-precessing multipole of a
    precessing waveform had its sign flipped. The PN integration runs
    backwards from `Lhat = z` there, where `alpha = atan2(L_y, L_x)` jumps by
    pi and the integrated `gamma` did not follow, a rotation by pi about
    `Lhat` (`precessing_model.backward_gamma`). TEOBResumS' angles do the
    same, but its multipoles start at the reference frequency, so nothing
    reads them there.
- `stationary_phase_transform` fitted a quadratic to the phase in its
    +-2 mHz window, biased by `f Psi''' w^2 / 10`: -3.4e-3 rad on the (2,2)'s
    `X` at 9.5 Hz. A cubic agrees with the analytic `X = phi + 2 pi f t(f)`
    to ~1e-6. The bias happened to cancel part of a ~-7e-3 rad offset
    between the surrogate's (2,2) phase at 9.5 Hz and TEOBResumS', so the
    mismatches against it below went from 4.8e-8 to 7.0e-8.
- `twist_modes_frequency_domain` rotated the surrogate's multipoles by
    `D(alpha, beta, gamma)`; being the complex conjugates of the multipoles'
    transforms, they need `D* = D(-alpha, beta, -gamma)`. The precession came
    out mirrored -- invisible in the aligned-spin limit, ~9e-2 against
    TEOBResumS at `beta ~ 0.27`, 5e-4 once fixed.
- The PN spin-precession `dOmega/dt` lacked the leading-order tidal term
    TEOBResumS adds for a BNS (`twist_waveform.tidal_flux_coefficient`,
    `lambdas=`): the angles drifted by ~0.1 rad by 400 Hz, ~1 rad by 1.5 kHz.
- The comparisons with TEOBResumS' precessing waveforms started it at 15 Hz,
    leaving it without (4,4) below 28.5 Hz and (3,3) below 21.4 Hz; mapped the
    azimuth to its `coalescence_angle` as `pi/2 - phi` (`PrecessingModel` at
    azimuth `phi` is TEOBResumS at `pi/2 + phi` with `hx` negated, a
    consequence of `Model.predict`'s `hx` sign, below); summed only the
    inertial multipoles listed in `use_mode_lm`, dropping the (3,1), (3,2),
    (4,1)--(4,3) that precession mixes out of the co-precessing (3,3), (4,4)
    (5e-6 to 7e-5 of mismatch, growing with the opening angle; now
    `use_mode_lm_inertial`); and did not give `PrecessingModel` TEOBResumS'
    own reference point: its spins at 0.95 `initial_frequency`, and its
    orbital phase at the first sample of its integration, which its EOB
    dynamics puts at ~9.506 Hz rather than 9.5
    (`PrecessingModel.teob_reference_phase`). See
    `docs/explanation/precession.md` and `visualization/teob_precessing.py`.

### Added

- **Regressed precession angles (prototype)**, `mlgw_bns.precession_regression`:
    `PrecessingModel.jax_predict(precession_regressor=...)` (and
    `jax_predict_modes`, `batched_precession.precessing_mode_components(...,
    precession=...)`) take the Euler angles from a regressor instead of
    integrating the PN precession equations over the whole band, which was
    ~95% of the cost of a JAX precessing waveform. The Euler angles
    themselves cannot be regressed: in the reference frame `alpha` swings by
    ~pi each time `Lhat` passes near `z`, and in the frame of `J` its winding
    number jumps where the two in-plane spins are comparable. Instead, in the
    frame of `J`, the in-plane part of `Lhat` is written as smooth envelopes
    times two carriers `exp(i Phi_k)`, whose derivatives --- the normal-mode
    frequencies of the linearized two-spin precession, with `|J| / L`
    evolving and each mode carried round by the other --- are known in
    closed form and integrated by a quadrature; `alpha_J - gamma_J`, from
    the minimal-rotation condition, the same way, with an analytic baseline
    for its secular growth. The envelopes (cubic B-splines, fitted by
    penalized least squares to the integrated angles) are compressed by PCA
    and regressed by `KernelRidgeNetwork`, up to ~165 Hz (for a total mass of
    2.8); above, where the carriers turn too slowly to tell the envelopes
    apart, the last few precession cycles are integrated (48 DOP853 steps)
    from the state the regressor gives there, the spins included. Trained
    on 16384 binaries, against the integration
    (`visualization/precession_regression_study.py`, 256 held out): waveform
    mismatches of 5.8e-5 median, 1.8e-3 at the 90th percentile, 6e-2 at
    worst, falling as ~N^-0.6 with the training set (1.1e-4 with 4096) for
    `q > 1.5` but not below, where the carriers get the two spins' beat
    wrong by up to ~20%. The angles take 10 ms for one binary and 0.8 ms
    each in a batch of 128, against 155 ms and 29 ms, and the whole waveform
    25 ms and 7.5 ms against 155 ms and 34 ms (four CPU cores).
    `twist_waveform`'s PN coefficients, 2PN orbital angular momentum and
    `dOmega/dt` are now helpers (`_spin_orbit_coefficients`,
    `orbital_angular_momentum`, `_orbital_frequency_rate`), unchanged to the
    last bit; `batched_precession.integrate_angles(final_frequency_22=...,
    full_state=...)` integrates up to a given frequency and returns the
    spins too.
- **Training at scale, on SLURM** (`docs/usage_guides/cluster.md`). Training
    sets too large for one process are kept on disk in shards
    (`mlgw_bns.sharding.ShardedStore`): each a function of the dataset's seed
    and its index alone, claimed through lock files by any number of
    processes on any number of machines, written atomically, made again
    identically if lost; the first `N` items are a uniform sample for every
    `N`, so the training sets of a learning curve are nested. Jobs stop
    cleanly on `SIGUSR1`/`SIGTERM` (what SLURM sends before the walltime)
    and requeue themselves. Two pipelines use it:
    `mlgw_bns.modes_dataset.ShardedModesDataset`,
    `visualization/modes_scale.py` and `slurm/modes/` for the co-precessing
    modes `Model` (the seven modes of `hom7_big`: a learning curve of kernel
    ridge and perceptrons up to 2^18 waveforms, each model validated on 2^14,
    timed and sized); `mlgw_bns.precession_dataset.ShardedDataset`,
    `visualization/precession_scale.py` and `slurm/precession/` for the
    regressed precession angles, up to 2^20 binaries.
- `mlgw_bns.neural_network.JaxMLPNetwork`, a perceptron trained with JAX
    (`mlgw_bns.jax_mlp.JaxMLP`: Adam with a cosine schedule, the best epoch
    on a held-out tenth kept, single precision, evaluated in double) as the
    regressor of a `ModeModel` (`nn_kind=JaxMLPNetwork`, also in the batched
    evaluation). Its stored size and evaluation time do not grow with the
    training set, unlike kernel ridge's. Its training checkpoints its whole
    optimizer state, every few minutes and on `SIGTERM`/`SIGUSR1`, and
    resumes from it to the bit (`TrainingInterrupted`); its loss weighs the
    principal components as they contribute to the residuals, and the
    training waveforms by their power for odd-`m` modes, as kernel ridge
    does. `ModeModel.fit_nn` fits a regressor on principal components
    reduced elsewhere (a shard at a time), `Model.train_downsampling` is the
    first step of `Model.generate` on its own.
- `mlgw_bns.model_validation.stored_waveform_mismatches`: the mismatches of
    every mode and of the full waveform of a `Model` against waveforms
    stored as their residuals, without TEOBResumS, ~0.4 s a waveform: for
    validation sets of thousands of waveforms.
- The regressed precession angles: training targets the regressor can
    learn, `AngleGrid.smoothing` and `refine_envelopes` (fold priors), worth
    about four times the data; the perceptron as the regressor
    (`PrecessionRegressor.train(mlp=...)`), 5.7e-5 median mismatch on 16384
    binaries against 1.5e-4 for kernel ridge
    (`docs/explanation/precession_regression.md`); training a shard at a time
    (`PrecessionRegressor.train_on_chunks`, with
    `principal_component_analysis.CovarianceAccumulator`); and the training
    sets ~16 times faster to make, ~25 ms of a core a binary (banded
    Cholesky fits, carrier tables batched in JAX).
- A docs page on precession (`docs/explanation/precession.md`): the twist,
    the conventions of the Euler angles and of the orbital phase at the
    reference frequency, and how to reproduce TEOBResumS' precessing waveforms
    with the JAX predictor (`docs/examples/precessing_vs_teobresums.py`, to
    ~8e-8). `PrecessingModel.teob_reference_phase` converts TEOBResumS'
    orbital phase origin.
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
    analytic limits and plots the PN angles and the resulting polarizations.
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
- `visualization/teob_precessing.py`: TEOBResumS' precessing waveforms in
    `mlgw_bns`' conventions, for the comparisons in `validate_model.py`.
    `teob_run(frequencies=)` evaluates on given frequencies (TEOBResumS'
    `interp_freqs`), ~500 times faster than its uniform grid (0.3 s against
    ~2 min a binary from 10 Hz) and the same there to ~1e-10: TEOBResumS
    twists every frequency it outputs. Its start frequency leads the list,
    since the twist takes the lowest one as the start of the (2,2).
- `twist_waveform.integrate_pn_spin_precession(independent_variable=
    "orbital_frequency", omega_dot=...)` and
    `precessing_model.eob_orbital_frequency_rate` /
    `PrecessingModel.euler_angles(anchor_to_reference_phase=True)`: march the PN
    spin precession *against* orbital frequency, along a `dOmega/dt` taken from
    an accurate `(2,2)` phase rather than the PN flux -- the surrogate analogue
    of the `SPIN_FLX_EOB` hand-off TEOBResumS does once its spin dynamics
    reaches the EOB band, and a way to fix the angle *trajectory* rather than
    just re-label it (a re-labelling, "re-anchoring" the angles to the (2,2)
    phase, was tried and dropped). Over 10 binaries the two came out a wash
    (median mismatch `4.2e-3` vs `3.6e-3`), so the orbital-frequency evolution is not
    what limits a precessing waveform (measured before the fixes below). Kept as
    an opt-in; all defaults
    unchanged. The shared PN precession r.h.s. was factored into
    `_pn_precession_derivatives`, with `_pn_spin_precession_rhs` now a thin
    time-domain wrapper (no behaviour change).
- `PrecessingParametersWithExtrinsic.reference_frequency_hz`: the frequency at
    which the spin vectors (and `Lhat = z`) are given. The PN spin precession
    is integrated both ways from it (`integrate_pn_spin_precession(f_start=)`).
    `None` keeps the previous behaviour, the start of the integration.
    TEOBResumS' frequency-domain path imposes its spins at 0.95 times its
    `initial_frequency` (its time-domain path at `initial_frequency` itself);
    matching that cuts the mismatch against it ~5x.
- `Model.coprecessing_amplitudes_and_phases`, the multipoles of
    `coprecessing_modes_dict` as amplitude and continuous phase.
- **The orbital phase of a precessing waveform is fixed at the reference
    frequency.** With precession it is no longer a choice of observer: it is
    the angle between the orbital separation and the in-plane spins. As in
    LALSuite (`f_ref`, `phiRef`), NRSur7dq4, SEOBNR and TEOBResumS, it is now
    set where the spins are given: with `reference_frequency_hz`,
    `PrecessingModel` rotates the co-precessing multipoles so that their
    orbital phase there is `PrecessingParametersWithExtrinsic.reference_phase`
    (which replaces `coalescence_phase`; the merger is still the reference
    without a `reference_frequency_hz`). It reads the orbital phase off the
    surrogate's own (2,2) at that frequency
    (`PrecessingModel.reference_orbital_phase`), through the time-shift
    invariant `X = Psi - f Psi'` (`precessing_model.stationary_phase_transform`)
    and the leading-order multipole phases (`LEADING_ORDER_MODE_PHASES`),
    with the branch modulo pi from the (2,1) or (3,3). Before, it was
    zero at the merger, which against TEOBResumS is a binary-dependent
    rotation of the co-precessing multipoles worth a 2.4e-3 median
    mismatch (90th percentile 8.2e-3), independent of the opening
    angle. With TEOBResumS' reference point given
    (`PrecessingModel.teob_reference_phase`), the precessing
    mismatch against it is 9.8e-8 (90th percentile 1.9e-3), and
    7.0e-8 (90th percentile 2.4e-7) without the frequencies where TEOBResumS' own `alpha` is wrong
    (below), the same as the aligned-spin mismatch of the same binaries with
    the in-plane spins zeroed, 6.8e-8 (2.3e-7); line of sight by line of sight the two
    agree to a median ratio of 1.00 (48 binaries x 4 lines of sight, total
    mass 2.8).
- **Batched precessing waveforms in JAX**: `PrecessingModel.jax_predict()`
    (`mlgw_bns.batched_precession`) returns a pure JAX function of `N`
    binaries at once, which can be jitted, vmapped and differentiated. It
    integrates the PN precession equations against the orbital frequency
    with fixed DOP853 steps (2048 each way from the reference frequency, in
    `x = ln Omega - 16 Omega_lo / Omega`), integrating `alpha - gamma`,
    which is regular where `Lhat` passes through `z`. It matches
    `PrecessingModel.predict` to mismatches of 1e-11--3e-10
    (`tests/test_batched_precession.py`). On a CPU a waveform takes ~75 ms
    alone and ~20 ms in a batch of 64, against ~20 s for the numpy path
    from the bottom of the band (SciPy's adaptive integration) and ~0.13 s
    for TEOBResumS. `special_func.wigner_d_function`/`spinsphericalharm`,
    `twist_modes_frequency_domain`, `polarizations_from_inertial_modes` and
    the PN precession right-hand side now take `xp=` (numpy or
    `jax.numpy`); `batched.py`'s duplicate Wigner d function is gone.
- `validate_model.py` validates the precessing model through its JAX
    predictor: the full precessing waveform against TEOBResumS' own
    `h+`, `hx` on a detector, in the mismatch distributions (TEOBResumS'
    alpha-step frequencies excluded,
    `precessing_full_waveform_mismatches`). The mismatch-distribution figure
    is now the optimised mismatches alone, with the mean and spread of the
    optimal time shifts in the legend (`full_waveform_mismatches` and
    `precessing_full_waveform_mismatches` return them); the not-optimised
    panel is its own figure, `_mismatches_not_optimised.png`. Also: surrogate and TEOBResumS
    co-precessing multipole by multipole after the twist and the projection,
    against their power fraction, for 400 binaries
    (`twisted_mismatch_vs_power_by_mode`); and its evaluation time, single
    and batched, against TEOBResumS precessing, in the second panel of the
    fit-line timing plot, which carries every approximant's fit in its
    legend entry and replaces the separate numpy-against-JAX one.
    `benchmark_evaluation_time.py` has the same approximants and figure
    (`--no-precessing` to skip them).
- `visualization/probe_hcross_sign.py`, which shows `Model.predict`'s `hx`
    sign against TEOBResumS' (Known issues).
- **Twisted co-precessing multipoles, for mode-by-mode relative binning**
    (Leslie, Dai & Pratten 2021): `precessing_waveform` is now the sum over
    co-precessing `(l, m)` of `c_+ h_lm` and `c_x h_lm`.
    `batched_precession.precessing_mode_components` (and
    `PrecessingModel.jax_predict_modes(modes, n_steps)`) returns the three
    factors, each of shape `(N, n_modes, k)`. `h_lm` is the co-precessing
    multipole with the reference-phase rotation and the merger-time shift
    applied, so a change of `reference_phase` by `delta` multiplies it by
    `exp(i m delta)` and leaves `c_+`, `c_x` alone.
    `c_+` and `c_x` hold the twist and the projection on the line of sight,
    and come from `precessing_model.twist_coefficients`, which is the twist
    of a unit multipole. `PrecessingModel.mode_components` is the numpy
    equivalent. `batched_precession.precession_angles` integrates the
    precession angles alone, and `precessing_mode_components` takes them as
    `angles=`, so a long frequency array can be evaluated in pieces with a
    single integration.
- **The LVK spin angles**: `mlgw_bns.spin_conversion.lvk_to_precessing`
    maps `theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2` and `phase` at
    `f_ref` to the spins, line of sight and `reference_phase` of
    `PrecessingParametersWithExtrinsic` (and `.from_lvk`).
    `lal_precessing_spins` is a vectorised numpy/JAX port of LALSuite's
    `SimInspiralTransformPrecessingNewInitialConditions` and agrees with it
    to 1e-11. `lvk_to_precessing` takes the spins at `phiRef = 0` and passes
    `phase` as `reference_phase`, so the precession angles do not depend on
    it. Against IMRPhenomXPHM, co-precessing mode by co-precessing mode,
    the overlap phases do not depend on `phase`, `phi_jl` or `theta_jn`:
    the geometry and the sense of `phase` are LALSuite's, apart from the
    sign of `hx` (Known issues), with matches of ~0.997
    (`visualization/validate_lvk_conventions.py`).

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
- TEOBResumS' precession angle `alpha` has spurious ~2 pi steps (in band on
    22 of 48 binaries), which its frequency-domain twist cubic-splines
    through, so its `h+`, `hx` are wrong on the few frequency bins around
    each (up to 4% of them; up to ~4e-3 of mismatch). This is a
    defect of the reference, not of `PrecessingModel`: its
    `prolong_euler_angles_FD` unwraps `alpha` with `unwrap_HM`, which misses
    the wraps of `alpha` running backwards through +-pi; the time-domain path
    uses `unwrap_euler` (reported upstream to the TEOBResumS developers).
- `Model.predict` (and so `PrecessingModel.predict`) returns `hx` with the
    opposite sign to TEOBResumS' (and LAL's) convention relative to `h+`:
    `hx/h+ = +0.836i` at inclination 1 against TEOBResumS' `-0.836i`
    (`visualization/probe_hcross_sign.py`), i.e. the polarizations of inclination
    `pi - iota`. For an aligned-spin binary this is an exact symmetry, so no
    mismatch can see it; in multi-detector inference it mirrors the inclination
    posterior. Not changed here, since it changes the shipped model's output.

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
- Odd-m mode regressors are now weighted by each training waveform's integrated
    mode power, following arXiv:2609.03025: near equal mass the (2,1) and (3,3)
    amplitude residuals grow linearly out of the odd-m zero, a boundary layer
    the global-RBF kernel cannot resolve and rings across. Weighting by
    `(int A^2 df / max)^beta` for odd m only (`power_weighting`,
    `power_weight_exponent` on `ModeModel`, on by default, recorded in the
    metadata) improves the (2,1) optimised per-mode mismatch by 23x and (3,3) by
    11x on `default_hom`, with (2,2) and (4,4) bit-identical.
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
