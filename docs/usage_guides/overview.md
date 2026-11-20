(overview_section)=
# Overview

The basic object provided by `mlgw_bns` is a `Model`:
it contains the functionality to generate new waveforms.

A `Model` is a multi-mode surrogate: it holds one `ModeModel` per
spherical-harmonic mode $(\ell, m)$, and sums their contributions,
weighted by the spin-weighted spherical harmonics, into the observer-frame
polarizations $h_+$ and $h_\times$.
The modes covered by the models shipped with the package are
$(2,2)$, $(2,1)$, $(3,1)$, $(3,2)$, $(3,3)$, $(4,3)$ and $(4,4)$
(`mlgw_bns.model.DEFAULT_MODES`).

(shipped_models)=
## The shipped models

Four pretrained models come with the package, listed in
`mlgw_bns.model.MODELS_AVAILABLE` and named by the largest tidal
polarizability they are trained on; they differ only in that (see
[below](model_parameter_ranges) for the full ranges). `default_hom`, the
default, is another name for `default_hom_lambda2000`
(`mlgw_bns.model_files.MODEL_ALIASES`).

| name | `lambda_1`, `lambda_2` | full-waveform mismatch, median / 90% | merger time error, median / 90% |
|---|---|---|---|
| `default_hom_lambda2000` (`default_hom`) | 0--2000 | 7.7e-9 / 1.2e-7 | 0.6 / 1.7 $\mu$s |
| `default_hom_lambda4000` | 0--4000 | 1.3e-8 / 1.2e-7 | 0.9 / 2.5 $\mu$s |
| `default_hom_lambda8000` | 0--8000 | 8.6e-8 / 1.4e-6 | 4.2 / 22 $\mu$s |
| `default_hom_lambda12000` | 0--12000 | 9.1e-7 / 1.0e-5 | 26 / 108 $\mu$s |

The mismatches are over 16384 TEOBResumS waveforms drawn as the training
set is (below), the merger-time errors over 400 uniform draws of each
model's range (`visualization/merger_time_release.py`). All are trained on
the same number of waveforms, $2^{15}$, so a narrower range is sampled more
densely and is more accurate: use the smallest one that covers the tidal
polarizabilities needed.

The training waveforms are drawn uniformly in the mass ratio, in the
effective spin $\chi_{\rm eff}$ and in the effective tidal deformability
$\tilde\Lambda$, then uniformly along the segment of constant
$\tilde\Lambda$ (or $\chi_{\rm eff}$) inside the range
(`mlgw_bns.dataset_generation.EffectiveParameterGenerator`). This fills
the corners of the range, such as both tidal deformabilities small, better
than a uniform draw of each. The kernel ridge regressors predict 60
principal components of each mode, and those of `default_hom_lambda2000`
and `default_hom_lambda4000` take $\log(1 + \Lambda_{1,2})$ as inputs
(`Hyperparameters.input_coordinates`): it makes their merger time several
times more accurate. It also costs accuracy where $\Lambda_1$ is large,
$\Lambda_2$ small and $q \sim 2.5$, a corner the draws sample sparsely, the
more so the larger the range: `default_hom_lambda8000` and
`default_hom_lambda12000` regress in $\Lambda_{1,2}$ (the latter 30
principal components). All of them are least accurate where one tidal
deformability is small, $\Lambda_1 \lesssim 5$, and the other large, with
$q < 2$: there the median mismatch is ~2e-7 for `default_hom_lambda2000`
and a few 1e-6 for `default_hom_lambda4000` and `default_hom_lambda8000`,
with 5--30% of the draws above 1e-5.

The models are trained from $\Lambda = 1$ and accept tidal
deformabilities down to 0. Below 1, TEOBResumS used to switch to its
black-hole settings (no tides, a ringdown): the version they are trained
with keeps its neutron-star settings for any $\Lambda > 0$, continuing its
quasi-universal relations (such as the spin-induced quadrupole) smoothly to
their black-hole values, and the models follow it there with the accuracy
they have at $\Lambda = 1$. At exactly $\Lambda_1 = \Lambda_2 = 0$ TEOBResumS
still gives its black-hole waveform, which differs from this limit by a
mismatch of about $10^{-4}$ (more with spins).

```{warning}
`default_hom_lambda12000` is not accurate enough in the late inspiral to
predict the merger reliably, and loading it raises a
`mlgw_bns.model.ModelAccuracyWarning`. Spread over tidal deformabilities up
to 12000, its training waveforms sample the parameter space too sparsely
for the last few milliseconds before merger: its time to merger above
about 1 kHz is wrong in parts of the space, worst at small tidal
deformabilities, large mass ratios and at the $\Lambda_1 \approx 12000$,
$q \approx 3$ edge. Compared with TEOBResumS, the merger comes 0.2--0.5 ms
early at $q \approx 2.5$, $\Lambda_1 \approx 10$, $\Lambda_2 \approx 1500$
(where a GW170817 posterior has a secondary mode), and more than 0.2 ms
early or late for about 5% of uniform draws over its range, against 1 of
400 (0.4 ms) for `default_hom_lambda8000`.

This is a modelling error, not a difference of convention. The merger
reference itself, the tangent to the $(2,2)$ phase at the top of the band,
is arbitrary: any consistent choice would do, and the models make the
same one. What is wrong is the phase of the surrogate between about 1 and
2 kHz. Below 1 kHz it shows as a constant shift of the merger time, which
a sampled or marginalised coalescence time absorbs; but the merger time,
and anything measured from the last milliseconds before merger, are not
reliable. The tests that fail on it are marked as expected failures in
`tests/conftest.py` (`KNOWN_FAILURES`).
```

The fastest way to access a functional instance of a `Model` is to load
one of them:
```python
from mlgw_bns import Model

model = Model.default_for_testing()  # default_hom
wide_model = Model.default_for_testing("default_hom_lambda12000")
```
By default this loads every shipped mode; a subset can be requested
instead with the `modes` keyword, which skips loading the files for the
modes left out:
```python
from mlgw_bns.higher_order_modes import Mode

model = Model.default_for_testing(modes=[Mode(2, 2), Mode(2, 1)])
```

The regressors of the models are too large to be included in the package:
they are downloaded the first time a model is loaded, and cached in
`~/.cache/mlgw_bns` (or in `$MLGW_BNS_CACHE`, if set; see
{mod}`mlgw_bns.model_files`). On machines without internet access, such as
the compute nodes of a cluster, download them beforehand into a cache they
can see:
```bash
MLGW_BNS_CACHE=/shared/mlgw_bns python -m mlgw_bns.model_files download default_hom
```

## Predicting waveforms

Now we can predict waveforms; in order to do so however we
pass the parameters through the `ParametersWithExtrinsic` class.
Further, we need to provide an array of frequencies in Hz at which to
compute the waveform:
```python
from mlgw_bns import ParametersWithExtrinsic
import numpy as np

frequencies = np.linspace(20., 2048., num=2000)
params = ParametersWithExtrinsic(
        mass_ratio=1.0,
        lambda_1=500.0,
        lambda_2=50.0,
        chi_1=0.1,
        chi_2=-0.1,
        distance_mpc=1.0,
        inclination=0.0,
        total_mass=2.8,
    )

hp, hc = model.predict(frequencies, params)
```

The waveform is referenced to the merger: it happens at time
`params.merger_time` (0 by default, in seconds), with orbital phase
`params.coalescence_phase` (0 by default), which rotates the $(\ell, m)$
mode by $e^{i m \phi_c}$.
The merger is where the frequency-domain phase of the $(2,2)$ mode becomes
linear in the frequency, at the top of the trained band; the model reads it
off its own $(2,2)$ mode, so there is nothing else to predict.

To get the individual mode contributions instead of the summed
polarizations, use `model.predict_modes_dict(frequencies, params)`.
For many binaries at once, and for JAX, see
{mod}`mlgw_bns.batched` (`model.predict_modes_amp_phase` and
`model.jax_modes_amp_phase`).

(precessing_waveforms)=
## Generating precessing waveforms

The surrogate is aligned-spin; precessing waveforms are obtained from it by
twisting its multipoles with the Euler angles of the orbital plane, as
TEOBResumS does (see [](precession) for the details and conventions).
This is done by a {class}`PrecessingModel <mlgw_bns.precessing_model.PrecessingModel>`
wrapping one of the models above, and requires the `jax` extra
(`pip install "mlgw-bns[jax]"`).

The spins are now three-vectors, given at the reference frequency
`reference_frequency_hz` (the $(2,2)$ gravitational-wave frequency, as
LALSuite's `f_ref`), in the frame where the orbital angular momentum is
along $z$ there; `reference_phase` is the orbital phase at that frequency,
and `azimuth` is the azimuthal angle of the line of sight:
```python
from mlgw_bns.precessing_model import PrecessingModel, PrecessingParametersWithExtrinsic

precessing = PrecessingModel(model)

params = PrecessingParametersWithExtrinsic(
    mass_ratio=1.3,
    lambda_1=400.0,
    lambda_2=600.0,
    chi_1=(0.3, 0.0, 0.05),
    chi_2=(0.0, -0.3, 0.02),
    distance_mpc=100.0,
    inclination=1.0,
    azimuth=0.3,
    total_mass=2.8,
    reference_phase=0.0,
    reference_frequency_hz=20.0,
)

hp, hc = precessing.predict(frequencies, params)
```
The first call compiles the waveform, which takes a few seconds; later ones
take ~20 ms. The binary can also be given by the LVK spin angles, as in
LALSuite and bilby:
```python
lvk_params = PrecessingParametersWithExtrinsic.from_lvk(
    mass_1=1.6,
    mass_2=1.2,
    theta_jn=0.5,
    phi_jl=1.0,
    tilt_1=0.8,
    tilt_2=2.0,
    phi_12=0.4,
    a_1=0.3,
    a_2=0.2,
    phase=0.0,
    reference_frequency_hz=20.0,
    lambda_1=300.0,
    lambda_2=700.0,
    distance_mpc=40.0,
)
```

For many binaries at once, `precessing.jax_predict()` returns a pure JAX
function of a batch, which can be jitted, vmapped and differentiated;
{func}`batch_arguments <mlgw_bns.batched_precession.batch_arguments>`
builds its arguments:
```python
import jax
from mlgw_bns.batched_precession import batch_arguments

predict = jax.jit(precessing.jax_predict())
hp, hc = predict(*batch_arguments([params, lvk_params], frequencies))
# hp, hc have shape (2, len(frequencies))
```

(new_model)=
## Making a new model

The way to create a new model from scratch is as follows:
```python
from mlgw_bns.higher_order_modes import Mode

m = Model(modes=[Mode(2, 2), Mode(2, 1)], filename="my_model")
m.generate()
m.set_hyper_and_train_nn()
```

For this to work `mlgw_bns` must be able to import
`EOBRun_module`, from the fork of TEOBResumS the shipped models were
trained with (see [](install_teobresums)).

We can then save this model to file with `m.save()`; afterwards we
will be able to recover it with
```python
m = Model(modes=[Mode(2, 2), Mode(2, 1)], filename="my_model")
m.load()
```
where it is crucial that the model name is the same --- the `load` method
only checks for files with the given name (in the current folder).
Each mode is stored in its own set of files, named
`{filename}_l{l}_m{m}`. The modes must include the $(2,2)$, which sets the
merger time and coalescence phase of all of them.

The hyperparameters used here are those provided with the package;
to perform an optimization see [](hyperparameter_optimization_section).

## Working with a single mode

A single `ModeModel` --- the object which actually does the machine learning
for one mode --- can be reached through the `mode_models` mapping:
```python
from mlgw_bns.higher_order_modes import Mode

mode_model = model.mode_models[Mode(2, 2)]
hp, hc = mode_model.predict(frequencies, params)
```
The signature is the same as `Model.predict`. On its own, a mode is
referenced to its own merger (the tangent to its own phase at the top of
the band), which is the right one only for the $(2,2)$.

(model_parameter_ranges)=
## The parameter ranges

These may change as the package is updated. For the pretrained models
shipped with the package, `default_hom_lambda<L>`, the current ranges are

- `mass_ratio`: between 1 and 3
- `lambda_1` (tidal polarizability of the larger star): between 0 and `L`
  (trained from 1, see [above](shipped_models))
- `lambda_2`: between 0 and `L`
- `chi_1` (aligned spin of the larger star): between -0.5 and 0.5
- `chi_2`: between -0.5 and 0.5
- frequencies: see below

Any `total_mass` is accepted. The model is a function of the
mass-rescaled frequency $Mf$, so changing the total mass rescales the
waveform exactly, $\tilde h(f; M) = (M/M_0)^2 \, \tilde h(f M / M_0; M_0)$:
what is fixed is the trained band, which in $Mf$ runs from about
$4.9 \times 10^{-5}$ to $0.040$. For a binary of $M$ solar masses this is
from $10\,\mathrm{Hz} \times M_\odot / M$ to
$8192\,\mathrm{Hz} \times M_\odot / M$: 5 to 2048 Hz or wider between 2
and 4 solar masses, 1.7 to 1365 Hz at 6. Below the band the waveform is
continued with its post-Newtonian expression; above it, after the merger,
it is zero.

A freshly-created `Model` uses `lambda_1` and `lambda_2` between 5 and
5000 instead, and otherwise the same ranges; they are set by
{class}`ParameterRanges <mlgw_bns.data_management.ParameterRanges>` and
customizable through the `parameter_ranges` argument of `ModeModel` (or of
`Model`, which forwards it to every mode) when building a new model. Its
band is set by the `initial_frequency_hz` and `srate_hz` arguments, which
are given at the reference mass of 2.8 solar masses.

The ranges actually in effect can be checked, once a `Model` object is
initialized as described before, by looking at the ranges of any of its modes:
```python
>>> print(model.mode_models[Mode(2, 2)].parameter_ranges)
ParameterRanges(q_range=(1.0, 3.0), lambda1_range=(5.0, 2000.0), lambda2_range=(5.0, 2000.0), chi1_range=(-0.5, 0.5), chi2_range=(-0.5, 0.5))
```
and the band, in Hz at the reference mass, with
```python
>>> print(model.dataset.total_mass)
2.8
>>> print(model.dataset.initial_frequency_hz, model.dataset.srate_hz / 2)
3.5714285714285716 2925.714285714286
```
The trained frequency array itself is `model.dataset.frequencies_hz`
(or `model.dataset.frequencies` in natural units), again at the reference
mass.
