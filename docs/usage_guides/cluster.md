(cluster-training)=
# Training at scale on a SLURM cluster

The pipeline in `slurm/modes/` trains {class}`~mlgw_bns.model.Model`, with
the seven modes of `default_hom`, on up to $2^{18}$ waveforms. It traces the
learning curve of kernel ridge and of perceptrons
({class}`~mlgw_bns.neural_network.JaxMLPNetwork`), and measures for each
model its accuracy (on $2^{14}$ held-out waveforms), its size on disk and its
evaluation time; it also makes the model shipped with a release (see
[](#release-model)).

It is a driver script, `visualization/modes_scale.py`, whose commands also
run on their own (see its `--help`), and two shell scripts: `submit.sh`
submits the jobs, each waiting for those it needs; `job.sh` is what each job
runs.

## Setting up

On the cluster, in a clone of the repository:

1. `uv sync --extra jax` (JAX for the perceptrons). This also builds
   TEOBResumS (`teobresums`, a development dependency) from the
   `initial-spa-fix` branch of `github.com/jacopok/teobresums`, at the commit
   in `uv.lock` (`[tool.uv.sources]` of `pyproject.toml`): the public release
   aborts on some of the waveforms of the training sets. Building it needs a C
   compiler and GSL (`module load gsl` or similar first); after new commits
   on the branch, `uv lock --upgrade-package teobresums`.
2. `cp slurm/modes/cluster.env.example slurm/modes/cluster.env` (or
   `release.env.example`, see [](#release-model)), and edit it: the account and partition, and
   `DATA`, where everything goes. It must be on a filesystem that every node
   sees, with room for the training set: scratch, not home. Module loads and
   exports go at its end; the jobs source it too.
3. `slurm/modes/submit.sh` submits all the stages; `submit.sh STAGE...` only some of them.

The scripts can be run where there is no SLURM to try them out: `job.sh STEP`
runs a step in the foreground.

## What runs

The stages (defaults of `cluster.env.example`):

| stage | jobs | what | cost |
|---|---|---|---|
| `downsampling` | 1 | the downsampling indices of each mode, on 64 waveforms each ({meth}`Model.train_downsampling <mlgw_bns.model.Model.train_downsampling>`), shared by the validation set | minutes |
| `generate` | an array of 8, 32 CPUs each | shards of 1024 waveforms of the training and validation sets, until none is left | ~0.4 s of a core a waveform: ~30 core-hours, 10 GB |
| `pca` | 1 | the principal components of each mode, on the first 32768 waveforms | minutes |
| `train` | per series and size: an array of 7 (one per mode), then one validating | the regressors, then the assembled model validated on the whole validation set, timed and sized | kernel ridge: hours and ~70 GB at $2^{16}$ (cubic, quadratic in $N$); perceptrons: 200000 steps, ~1 h a mode on 16 CPUs |
| `teob` | per model, one (only where `TEOB_N` is set) | `visualization/validate_model.py` on it, against TEOBResumS: the end-to-end check, merger reference and post-Newtonian extension included | a few TEOBResumS calls a waveform |

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

## Hyperparameter studies

The `optimize` stage (not run by default) is the cluster's
`optimize_n_hours.py`:

```bash
slurm/modes/submit.sh optimize 24          # the studies, for 24 hours (their walltime)
```

runs an [optuna](https://optuna.org) study per entry of `OPTIMIZE_STUDIES`
(per mode: an array of 7 jobs), kept in a journal
file under `$DATA/optimize/<study>/`
({func}`~mlgw_bns.hyperparameter_optimization.open_study`, no database
needed), each in a job on a node of its own (`OPTIMIZE_CPUS`) for the hours given. In it, `OPTIMIZE_WORKERS` processes share the cores
({func}`~mlgw_bns.hyperparameter_optimization.run_in_parallel`: each
pinned to its share, with as many threads) and the study, each reading the
training and validation sets once, then running trials until the walltime
({func}`~mlgw_bns.hyperparameter_optimization.run_worker`; `--hours`
instead stops them when the study's trials have taken that many hours,
summed). The job is not requeued: submitting the stage again continues
the studies where they were. In a study:

- a trial trains a perceptron (the search space is
  {data}`~mlgw_bns.hyperparameter_optimization.MLP_SEARCH_SPACE`: depth,
  width, activation, learning rate and its final fraction, warmup, clipping
  of the gradients, batch, weight decay) on the first `--n` samples for a
  fixed `MLP_STEPS` steps, or kernel ridge (`--kind krr`: its width only);
- its value is measured on the validation set: the mean square error of
  the regression of the mode's residuals (without the truncation of the
  principal components, the same in every trial);
- the perceptrons report their best held-out loss as they train, and
  successive halving (`--pruner`) stops the worse ones at a tenth of their
  steps, then at three tenths;
- a worker warned of the walltime stops its trial (the perceptron saving
  its checkpoint) and enqueues it again; the retry, by the next job,
  resumes from the checkpoint. A trial whose worker died without warning
  (its heartbeat stopped for `--stale-minutes`) is retried by the next
  worker to start.

A study searches at one training-set size `--n`, the one the regressor
will be trained on. A perceptron's trial costs the same whatever the size
(its steps are fixed), so a search on fewer samples saves nothing, and
what is best depends on the size: in the learning curves, batches of 4096
are worse than 1024 at $2^{14}$ binaries and better at $2^{20}$, and deeper
or wider networks keep up to $2^{18}$ but not at $2^{20}$. (Kernel ridge,
whose cost grows as $N^3$, has one parameter to search: cheap at any size
it can be trained at.) `--enqueue-from` seeds a study with the best trials
of another as its workers start; `--fix NAME=VALUE` holds parameters of
the search space. Kernel ridge is best run with `--workers 1`, its linear
algebra on all the cores.

How well a perceptron trains depends on its seed about as much as on its
hyperparameters: the same 4 x 256 network on the same $2^{13}$ waveforms of
the (2,2) mode reaches a held-out loss of 5e-6, 2e-5 or 2e-4 with three
seeds, as it learns or not the corner of the highest mass ratios and
$\Lambda_1$ (on $2^{14}$, 7e-5 to 4e-4). Each trial's seed is therefore a
function of its parameters (another for every trial, the same for a retry),
so that the study does not favour the configurations lucky with one seed;
the best trials of a study are still the luckiest of their neighbours, and
are worth training again with a few other seeds (`--mlp-seed`, in a
`SERIES`) before one is chosen.

```bash
python visualization/modes_scale.py studies $DATA/optimize/*        # the best trials, and the train options of each
```

The study files are small: copy `optimize/` back without its
`checkpoints/` and lock files, at any time, while the studies run as well.
A journal is only appended to, line by line, so a copy is the study as it
was some moment before (a last line not copied in full is left out), and
`studies` reads it without writing to it or waiting for its lock
({func}`~mlgw_bns.hyperparameter_optimization.read_study`); running trials
are counted as running.

```bash
rsync -av --exclude 'checkpoints/' --exclude '*.lock' cluster:$DATA/optimize/ visualization/modes_scale/run/optimize/
python visualization/modes_scale.py studies visualization/modes_scale/run/optimize/*
```

To follow them, `visualization/studies_dashboard.py` does this every 10
minutes and shows all the studies in an
[optuna dashboard](https://optuna-dashboard.readthedocs.io) (the history
of each, the importance of its parameters, the learning curves of its
perceptrons as they were pruned or not, the value of its trials against
their number of weights; and a note with its best trial's `train`
options):

```bash
python visualization/studies_dashboard.py --modes cluster:/path/of/the/DATA   # then open http://localhost:8080
```

The `studies` report gives the importance of the parameters too
(PED-ANOVA, from 10 completed trials). `--sampler gp` samples with a
Gaussian process instead of TPE: better on few trials, but it learns only
from the completed ones (not the pruned), so it is meant with
`--pruner none`, and it needs torch (`uv sync --extra torch`).

The trials keep no regressor (their checkpoints are deleted as they end).
The `best` stage trains, at any point of the studies, the regressor of the
best trial of each one so far, and validates it as the `train` stage does:

```bash
slurm/modes/submit.sh best        # runs/<study>_best<trial of each mode>_<N>/model, and its validate_*.npz
```

The perceptron is built from the trial's parameters
({func}`~mlgw_bns.hyperparameter_optimization.study_mlp_config`, with
`train --from-study`), not from its `command_line`, which rounds them:
trained again, it is the trial's own, seed included (on the modes, the
same objective to all the digits). Each mode's regressor is
the perceptron only where it does better than kernel ridge on the first
`BEST_KRR_N` waveforms (cluster.env), which is fitted and scored on the
study's own objective first; the choice of each mode, with both values,
is in `runs/<run>/model_l2_m2.choice.json` and so on. The trial numbers
are in the run's name, so a later `best`, once the studies have found
better trials, makes another run.

## Copying the results back

What is worth copying is small; the training sets are not, and stay on the
cluster (delete them when the runs are done, or keep them to train more).

From `$DATA` (of `slurm/modes/cluster.env`):

| path | what | size | copy |
|---|---|---|---|
| `runs/` | each model (`<series>_<N>/model_*`, without training data, loadable with `Model(modes, filename=".../model").load()`), `validate_<series>_<N>.npz` (the mismatches of every validation waveform, per mode and in full, and the power of each mode) and `.json` (their statistics, the model's size and evaluation time) | ~1 GB (kernel ridge at $2^{16}$: ~130 MB a model; perceptrons ~6 MB) | yes |
| `logs/` | the output of every job | small | yes |
| `teob/` | the end-to-end validations against TEOBResumS: figures and `<run>_mismatches.npz` (their logs are in `logs/`) | small | yes |
| `train/config.json`, `train/downsampling.npz`, `train/pca/`, `validation/config.json`, `validation/downsampling.npz` | what the models were trained and validated on | ~10 MB | yes |
| `train/shards/`, `validation/shards/` | the datasets | 9.4 GB + 0.6 GB | no |
| `optimize/` | the hyperparameter studies (optuna journals), also while they run | small | yes, without `checkpoints/` and `*.lock` |
| `*/locks/`, `runs/*/checkpoint_*`, `optimize/*/checkpoints/` | in-progress state | | no |

```bash
rsync -av --exclude 'shards/' --exclude 'locks/' --exclude '*checkpoint*' --exclude '*.lock' \
    cluster:$DATA/ visualization/modes_scale/run/
```

Then, here:

```bash
# the evaluation times of all the models on one machine, comparable with each other
python visualization/modes_scale.py timing visualization/modes_scale/run/runs/*/model
python visualization/modes_scale.py summary visualization/modes_scale/run/runs
```

`visualization/modes_scale/` is ignored by git.


(release-model)=
## Training the model of a release

The model shipped in the package, `default_hom`, is trained by this
pipeline with the settings of `slurm/modes/release.env.example`: kernel
ridge on the first $2^{13}$ to $2^{16}$ of $2^{16}$ training waveforms
(same domain as `default_hom`: $q \in [1, 3]$, $|\chi| < 0.5$,
$\Lambda \in [5, 12000]$), each model validated on the $2^{14}$ stored
validation waveforms and end to end against TEOBResumS (`teob`, 1000
waveforms a distribution).

On the cluster:

```bash
cp slurm/modes/release.env.example slurm/modes/cluster.env   # and edit: account, partition, DATA
slurm/modes/submit.sh        # downsampling, generate, pca, train, teob
```

Here, once it is done:

```bash
rsync -av --exclude 'shards/' --exclude 'locks/' --exclude '*checkpoint*' --exclude '*.lock' \
    cluster:$DATA/ visualization/modes_scale/release/
python visualization/modes_scale.py summary visualization/modes_scale/release/runs
```

Choose the model from `runs/summary.md` (accuracy, size and evaluation
time against the size of the training set) and from the end-to-end
validations, `teob/krr_<N>_*` and the `logs/modes-teob_krr_<N>-*.log` (their per-mode and
full-waveform mismatches, with and without the time and phase optimised).
The size matters: the model is in the wheel, and PyPI takes 100 MB a file
unless asked for more; a kernel ridge model is ~130 MB at $2^{16}$, half
that at $2^{15}$. Then ship it, as `default_hom`:

```bash
RUN=visualization/modes_scale/release/runs/krr_32768
for file in $RUN/model_l*; do cp "$file" "mlgw_bns/data/default_hom_${file##*/model_}"; done
uv run pytest      # the accuracy thresholds of the tests may need recalibrating
```
