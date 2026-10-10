(cluster-training)=
# Training at scale on a SLURM cluster

Two models are trained at scale by the pipelines in `slurm/`, with the same
machinery:

- `slurm/modes/`: the co-precessing modes model, {class}`~mlgw_bns.model.Model`
  with the seven modes of `hom7_big` (`make_hom7_dataset.py`), trained on
  $2^{18}$ waveforms. It traces the learning curve of kernel ridge and of
  perceptrons ({class}`~mlgw_bns.neural_network.JaxMLPNetwork`) from $2^{12}$ to
  $2^{18}$ waveforms, and measures for each model its accuracy (on
  $2^{14}$ held-out waveforms), its size on disk and its evaluation time.
- `slurm/precession/`: the regressed precession angles of
  {mod}`mlgw_bns.precession_regression` (see [](precession-regression-error)),
  trained on up to $2^{20}$ binaries and validated on $2^{14}$.

Each pipeline is a driver script, `visualization/modes_scale.py` or
`visualization/precession_scale.py`, whose commands also run on their own
(see their `--help`), and two shell scripts: `submit.sh` submits the jobs, each
waiting for those it needs; `job.sh` is what each job runs.

## Setting up

On the cluster, in a clone of the repository:

1. `uv sync` --- the modes pipeline also needs TEOBResumS's Python module
   (`teobresums`, a development dependency): build it on the cluster, and
   point `[tool.uv.sources]` of `pyproject.toml` at its checkout (e.g.
   `teobresums = { path = "../teobresums/Python" }`) before syncing.
2. `cp slurm/modes/cluster.env.example slurm/modes/cluster.env` (or
   `slurm/precession/...`), and edit it: the account and partition, and
   `DATA`, where everything goes. It must be on a filesystem that every node
   sees, with room for the training set: scratch, not home. Module loads and
   exports go at its end; the jobs source it too.
3. `slurm/modes/submit.sh` (or `slurm/precession/submit.sh`) submits all
   the stages; `submit.sh STAGE...` only some of them.

Both scripts can be run where there is no SLURM to try them out: `job.sh STEP`
runs a step in the foreground.

## What runs

The modes pipeline (`slurm/modes/submit.sh`, defaults of
`cluster.env.example`):

| stage | jobs | what | cost |
|---|---|---|---|
| `downsampling` | 1 | the downsampling indices of each mode, on 64 waveforms each ({meth}`Model.train_downsampling <mlgw_bns.model.Model.train_downsampling>`), shared by the validation set | minutes |
| `generate` | an array of 8, 32 CPUs each | shards of 1024 waveforms of the training and validation sets, until none is left | ~0.4 s of a core a waveform: ~30 core-hours, 10 GB |
| `pca` | 1 | the principal components of each mode, on the first 32768 waveforms | minutes |
| `train` | per series and size: an array of 7 (one per mode), then one validating | the regressors, then the assembled model validated on the whole validation set, timed and sized | kernel ridge: hours and ~70 GB at $2^{16}$ (cubic, quadratic in $N$); perceptrons: 200000 steps, ~1 h a mode on 16 CPUs |

The datasets are {class}`~mlgw_bns.modes_dataset.ShardedModesDataset`
directories: the parameters and the residuals of every mode at their
downsampling indices, from one TEOBResumS call a waveform
({meth}`Model._multimode_mode_residuals <mlgw_bns.model.Model._multimode_mode_residuals>`).
The validation mismatches
({func}`~mlgw_bns.model_validation.stored_waveform_mismatches`) compare the
model with these stored waveforms, mode by mode (time and phase optimised,
as {class}`~mlgw_bns.model_validation.ValidateModel` does) and for the sum
of the modes at an isotropic inclination (time and azimuth optimised), with
the ET noise: they measure the regression and the truncation of the
principal components, the part of the error that the size of the training
set and the regressor control.

The precession pipeline (`slurm/precession/submit.sh`):

| stage | jobs | what | cost |
|---|---|---|---|
| `generate` | an array of 8 | shards of 4096 binaries of both sets: the integrated angles and their envelopes | ~25 ms of a core a binary: ~10 core-hours, 68 GB for $2^{20}$ |
| `exact` | 1 | the waveforms of the validation set with the integrated angles | ~30 ms a binary |
| `refine` | per iteration, an array of 4 (the priors of each fold), then an array of 8 (the refits) | {func}`~mlgw_bns.precession_regression.refine_envelopes`, twice | ~20 minutes a prior, ~5 ms a refit |
| `train` | per series and size, one | kernel ridge (up to $2^{15}$) or a perceptron on the first $N$, then validated | as above |

## Following it

`DATA` and the other settings exist only in the scripts: `submit.sh` prints
where the logs go, and sourcing `cluster.env` from the repository sets them
in a shell too.

```bash
source slurm/modes/cluster.env    # DATA
squeue --me
tail -f $DATA/logs/*.log
.venv/bin/python visualization/modes_scale.py status $DATA/train $DATA/validation
```

`status` says how many shards are made, by whom the others are being made,
and how long a shard takes. As the validations end:

```bash
.venv/bin/python visualization/modes_scale.py summary $DATA/runs
```

prints (and writes to `$DATA/runs/summary.md`) a table of the median, 90th
percentile and largest full-waveform mismatch, the median mismatch of each
mode, the size on disk and the evaluation time of every model, and plots the
three against the size of the training set (`$DATA/runs/summary.png`).

## Interruptions

Every step can be interrupted and run again, and redoes nothing it had
finished:

- Shards are claimed through lock files, refreshed while they are being
  made; a lock left by a job that died is taken over after 10 minutes. Each
  shard is a function of the dataset's seed and of its index alone, so a
  shard made again is the same.
- SLURM sends `USR1` five minutes (`SIGNAL_LEAD`) before the walltime: a job
  making shards abandons the one it is on, a perceptron saves its
  checkpoint (its whole optimizer state: the training resumes exactly where
  it was), and the job exits with status 75, which makes `job.sh` requeue it
  (`scontrol requeue`, at most `MAX_REQUEUES` times). If the cluster does not
  allow a job to requeue itself, submit the stage again by hand.
- A kernel ridge fit cannot be checkpointed while it solves; the modes
  pipeline fits one mode per job, and keeps each mode once it is trained.
- `scancel` stops a job without requeueing it.
- `submit.sh STAGE...` submits stages again: after a failure, or after
  adding sizes or series to `cluster.env`. A dataset whose last shard is full
  can be grown by raising its size.

## When a job fails

Every job logs to `$DATA/logs/<pipeline>-<step>-<id>.log` (array tasks
`-<id>_<task>.log`), appending: the output of every attempt of a requeued
job is there. If a job ends without a log, SLURM could not write it (the
filesystem of `DATA` must be visible from the compute nodes) or did not
start it; its accounting says why:

```bash
sacct -j <id> --format=JobID%20,JobName%30,State,ExitCode,Reason,Elapsed,NodeList,Restarts
scontrol show job <id>    # while it is still known
```

A step stopping by itself as it starts (exit status 75 within
`MIN_RUN_SECONDS`, 10 minutes, without the walltime signal) is not requeued,
so that a broken job fails instead of looping.

## Copying the results back

What is worth copying is small; the training sets are not, and stay on the
cluster (delete them when the runs are done, or keep them to train more).

From the **modes** pipeline (`$DATA` of `slurm/modes/cluster.env`):

| path | what | size | copy |
|---|---|---|---|
| `runs/` | each model (`<series>_<N>/model_*`, without training data, loadable with `Model(modes, filename=".../model").load()`), `validate_<series>_<N>.npz` (the mismatches of every validation waveform, per mode and in full, and the power of each mode) and `.json` (their statistics, the model's size and evaluation time) | ~1 GB (kernel ridge at $2^{16}$: ~130 MB a model; perceptrons ~6 MB) | yes |
| `logs/` | the output of every job | small | yes |
| `train/config.json`, `train/downsampling.npz`, `train/pca/`, `validation/config.json`, `validation/downsampling.npz` | what the models were trained and validated on | ~10 MB | yes |
| `train/shards/`, `validation/shards/` | the datasets | 9.4 GB + 0.6 GB | no |
| `*/locks/`, `runs/*/checkpoint_*` | in-progress state | | no |

```bash
rsync -av --exclude 'shards/' --exclude 'locks/' --exclude '*checkpoint*' \
    cluster:$DATA/ visualization/modes_scale/run/
```

From the **precession** pipeline (`$DATA` of `slurm/precession/cluster.env`):

| path | what | size | copy |
|---|---|---|---|
| `runs/` | each regressor (`<series>_<N>.joblib`, with its `.json`) and `validate_<series>_<N>.npz` | ~0.5 GB | yes, without the `.checkpoint` files |
| `logs/` | the output of every job | small | yes |
| `train/config.json`, `validation/config.json`, `train/refine/*/config.json`, `train/refine/*/fold*.joblib` | what was trained on, and the priors of the refinements | ~30 MB | yes |
| `validation/exact.npz` | the validation waveforms with integrated angles, to validate other regressors without integrating again | ~0.8 GB | if you will validate more |
| `train/shards/`, `train/refine/*/shards/`, `validation/shards/` | the datasets | ~70 GB | no |

```bash
rsync -av --exclude 'shards/' --exclude 'locks/' --exclude '*.checkpoint' \
    cluster:$DATA/ visualization/precession_scale/run/
```

Then, here:

```bash
# the evaluation times of all the models on one machine, comparable with each other
python visualization/modes_scale.py timing visualization/modes_scale/run/runs/*/model
python visualization/modes_scale.py summary visualization/modes_scale/run/runs
# learning curves: of the full waveforms, or of a mode
python visualization/precession_learning_curve.py --directory visualization/modes_scale/run/runs --threshold 1e-4
python visualization/precession_learning_curve.py --directory visualization/modes_scale/run/runs --threshold 1e-4 --key "mode l2_m1"
python visualization/precession_learning_curve.py --directory visualization/precession_scale/run/runs
```

`visualization/modes_scale/` and `visualization/precession_scale/` are
ignored by git.
