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

Two pretrained models come with the package, listed in
`mlgw_bns.model.MODELS_AVAILABLE`; they differ only in the range of the
tidal polarizabilities they are trained on (see
[below](model_parameter_ranges) for the full ranges):

| name | `lambda_1`, `lambda_2` | full-waveform mismatch, median / 90% | merger time error, median / 90% |
|---|---|---|---|
| `default_hom` (the default) | 5--2000 | 2.5e-8 / 4.6e-7 | 1.4 / 8.7 $\mu$s |
| `default_hom_lambda12000` | 5--12000 | 9.1e-7 / 9.8e-6 | 36 / 132 $\mu$s |

Both are trained on the same number of waveforms, so the narrower range is
sampled more densely, and is more accurate; use `default_hom_lambda12000`
only if tidal polarizabilities above 2000 are needed.

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

These may change as the package is updated. For the default pretrained
model shipped with the package, `default_hom`, the current ranges are

- `total_mass`: between 2 and 4 solar masses
- `mass_ratio`: between 1 and 3
- `lambda_1` (tidal polarizability of the larger star): between 5 and 2000
- `lambda_2`: between 5 and 2000
- `chi_1` (aligned spin of the larger star): between -0.5 and 0.5
- `chi_2`: between -0.5 and 0.5
- frequencies: between 5 and 2048 Hz

The other one, `default_hom_lambda12000`, has the same ranges except for
`lambda_1` and `lambda_2`, between 5 and 12000.

A freshly-created `Model` uses `lambda_1` and `lambda_2` between 5 and
5000 instead, and otherwise the same ranges; they are set by
{class}`ParameterRanges <mlgw_bns.data_management.ParameterRanges>` and
customizable through the `parameter_ranges` argument of `ModeModel` (or of
`Model`, which forwards it to every mode) when building a new model.

The ranges actually in effect can be checked, once a `Model` object is
initialized as described before, by looking at the ranges of any of its modes:
```python
>>> print(model.mode_models[Mode(2, 2)].parameter_ranges)
ParameterRanges(mass_range=(2.0, 4.0), q_range=(1.0, 3.0), lambda1_range=(5.0, 2000.0), lambda2_range=(5.0, 2000.0), chi1_range=(-0.5, 0.5), chi2_range=(-0.5, 0.5))
```
for the first six,
```python
>>> print(model.dataset.initial_frequency_hz)
5.0
```
for the initial frequency and
```python
>>> print(model.dataset.srate_hz / 2)
2048.0
```
for the maximum (Nyquist) frequency.

The actual array of possible frequencies, `model.dataset.frequencies_hz`
(or `model.dataset.frequencies` in natural units)
is wider, to accomodate the possibility of changing the total mass.
