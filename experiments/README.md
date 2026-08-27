# Accuracy experiments

Scratch work on where the aligned-spin `mlgw_bns` surrogate loses accuracy
under a budget of fewer than ten thousand training waveforms, and which
changes to the pipeline recover it.

Nothing here is imported by the package. The modules re-implement the
stages between the residuals and the mismatch so that each design choice
becomes a knob, and score every setting of those knobs with the mismatch
`mlgw_bns.model_validation.ValidateModel` already computes.

## Layout

| file | what it does |
| --- | --- |
| `cache.py` | Generates the expensive part once --- downsampling nodes, training and validation residuals, PN baselines --- and stores it outside the repository. |
| `sampling.py` | A scrambled-Sobol parameter generator, as an alternative to the i.i.d. uniform draw used for training parameters. |
| `pipeline.py` | The reduction and regression stages, with the design choices exposed as a `Config`. |
| `evaluate.py` | Trains one `Config` and scores it by mismatch against the cached EOB validation waveforms. |
| `run_sweep.py` | Runs a stage of the study, one variant per worker process. |
| `diagnose.py` | Three figures: what the network is asked to fit, where in the band the mismatch lives, and the truncation floor. |
| `downsampling_floor.py` | The error contributed by the downsampling alone, which every mismatch in the sweep cancels out. |
| `amplitude_zeros.py` | Characterises the sign change in the (2,1) and (3,3) amplitude residuals. |
| `mode_power.py` | Each mode's share of the noise-weighted power, which converts a per-mode mismatch into what it costs the summed waveform. |
| `timing.py` | What the accuracy gain costs at evaluation time. |
| `plot_results.py` | The summary figure: accuracy against training budget, and the error budget by mode. |

## Running

```bash
python experiments/cache.py --modes 22,21,33 --n-train 8192
python -m experiments.run_sweep --stage floors           # truncation floors, fast
python -m experiments.run_sweep --stage one_at_a_time    # one change at a time
python -m experiments.run_sweep --stage regressor_tuning
python -m experiments.run_sweep --stage sizes             # accuracy against training budget
python -m experiments.run_sweep --stage per_mode --mode 21
python -m experiments.downsampling_floor --mode 22
python -m experiments.mode_power
python -m experiments.diagnose
python -m experiments.plot_results
```

Caches go to `$MLGW_EXPERIMENT_CACHE` (a scratch directory by default);
results land in `experiments/results/` as JSON plus PNG.

## Reading the numbers

Every mismatch here compares two waveforms reconstructed from the *same*
downsampling nodes, so the spline interpolation between nodes cancels.
That is deliberate --- it isolates the reduction and regression stages ---
but it means the numbers are not directly comparable to a finished
model's. `downsampling_floor.py` measures the part that was cancelled.

## What the sweeps found

Measured on 8192 training waveforms at 5 Hz, against 256 validation
binaries with the ET PSD. Full write-up, with the tables and figures, is
the "The Surrogate's Real Bottleneck" artifact.

* **The basis is not the bottleneck.** The 30-component truncation floor
  for (2,2) is 1.76e-10, against the 7.2e-6 the shipped model reaches.
  The whole gap is in the parameters-to-coefficients regression.
* **The training set is not the bottleneck either.** The shipped
  pipeline stops improving at 2048 waveforms; the tuned kernel keeps
  improving roughly as `n**-2` all the way to 8192.
* **The network's loss is weighted backwards.** Dividing each component
  coefficient by `max|x_i|` before an unweighted MSE weights component
  `i`'s residual-space error by `s_i**-2`, which runs a factor 1e9 in
  favour of the *least* important component. A single shared scale fixes
  it; a kernel ridge regression is immune to it by construction, since
  `(K + alpha I)^-1 y` is equivariant under per-output rescaling.
* **`SklearnNetwork.fit` clips the mini-batch to the feature count.**
  `x_data.shape[1]` (5) rather than `shape[0]`; see
  `mlgw_bns/neural_network.py:393`. Every packaged model was trained with
  a batch size of 5. Repairing it does not measurably help, but it makes
  tuning `batch_size` meaningful.
* **The (2,1)/(3,3) amplitude sign changes are real and cost little.**
  3.1% of (2,1) training waveforms cross zero, giving component
  coefficients tens of times the typical spread; modelling the amplitude
  against a fixed reference rather than each waveform's own PN baseline
  is worth 18x on (3,3). But the (2,1) mode carries 1e-4 of the
  noise-weighted power, and 2e-6 of it near equal masses --- which is
  exactly where its mismatch is worst.
* **(2,2) now sits on the resampling floor.** The tuned kernel scores
  3.39e-9 against a downsampling-only floor of 2.39e-9. Further accuracy
  on that mode has to come from the node placement, not from waveforms.

* **Flooring the PN amplitude works, but buys nothing over a fixed
  reference.** `Config(amplitude="softened_pn")` keeps each waveform's own
  PN baseline but divides by `sqrt(A_pn**2 + (delta * A_ref)**2)`, which
  equals `|A_pn|` where that is large and levels off at `delta * A_ref`
  near its zero. It does what it is meant to: at `delta = 0.1` the largest
  (3,3) amplitude ratio in the training set drops from 66.7 to 1.06 while
  the median moves only from 0.98 to 0.90, so the outliers are removed
  without distorting the bulk. But the mismatch converges monotonically to
  the `reference_ratio` value and never beats it --- 4.68e-6 against
  4.68e-6 on (3,3), 6.44e-4 against 6.47e-4 on (2,1).

  The conclusion is that the *only* thing wrong with `A_eob / A_pn` was
  the near-singularity. Keeping the per-waveform PN baseline, which is what
  the softening was for, gains nothing: its parameter dependence is
  smooth, so the regression learns it just as easily either way. Note
  also that a continuous divisor which changes sign must pass through
  zero, so the floor can only be applied to the magnitude --- the sign is
  carried by `A_eob`, which crosses zero smoothly on its own.
* **What the reference amplitude is barely matters; that it is *fixed*
  is the whole point.** On (3,3) with 8192 training waveforms, five
  different divisors --- the training-set mean PN amplitude, the PN
  amplitude of the box centre (which is what the library ships), the two
  opposite corners of the parameter box, and a completely flat divisor of
  ones --- all score within 0.1% of each other, 4.678e-6 to 4.684e-6,
  against 8.69e-5 for the per-waveform PN baseline. The factor of
  eighteen therefore comes from every waveform sharing one divisor, not
  from that divisor being a good approximation to anything: what it buys
  is that the handful of waveforms whose own PN amplitude has a deep
  minimum stop setting the normalisation for the whole training set.

  The reference's overall *normalisation* is a mild knob, because with
  `weighting="none"` the amplitude and phase blocks enter one joint PCA
  unscaled, so the amplitude block's magnitude sets the balance between
  them. Multiplying the box-centre reference by a constant gives

  | factor | median | worst |
  |--------|--------|-------|
  | 1e-6 | 2.173e-4 | 8.65e-3 |
  | 1e-3 | 4.596e-6 | 6.26e-2 |
  | 1    | 4.679e-6 | 6.19e-2 |
  | 1e3  | 5.301e-6 | 7.25e-2 |
  | 1e6  | 1.159e-5 | 8.06e-2 |

  --- a broad plateau from 1e-3 to 1 with the physical normalisation
  inside it, falling off by 2.5x at 1e6 and by 46x at 1e-6. The
  asymmetry is informative: shrinking the divisor inflates the amplitude
  block until it crowds the phase out of the leading components, which
  costs the median an order of magnitude while *improving* the worst
  case sevenfold. There is no free accuracy here, but there is a
  median-against-tail trade that the column weighting could exploit
  deliberately rather than by accident.
* **On the even-m modes the fixed reference is a wash.** (2,2) scores
  3.521e-9 with the per-waveform PN baseline against 3.416e-9 with the
  fixed reference, a 3% improvement with both at that mode's
  downsampling floor of 2.39e-9; (4,4) scores 4.306e-8 against 4.642e-8,
  an 8% *degradation*. Neither PN amplitude has a deep minimum --- (2,2)
  min/max across the band is 0.44, against 5e-6 for (3,3) --- so there
  is nothing to repair and the differences are second-order. Applying
  the option per-dataset rather than per-mode therefore costs about 8%
  on the mode that contributes the least power, in exchange for factors
  of eighteen and two and a half on the odd-m ones. A per-mode flag
  would recover that 8% if it ever mattered.
* **A Sobol training draw buys the tail, not the boundary.** Scrambled
  low-discrepancy training parameters cut the single worst validation
  binary from 7.11e-5 to 1.83e-5 at an unchanged median, but leave the
  systematic edge effect untouched: the correlation between mismatch and
  distance to the faces of the parameter box is +0.43 either way.

Summed over the three modes and weighted by power, the full-waveform
median mismatch goes from 7.50e-6 to 5.39e-8, with no extra training
data.
