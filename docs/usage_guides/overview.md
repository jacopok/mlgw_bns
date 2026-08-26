(overview_section)=
# Overview

The basic object provided by `mlgw_bns` is a `Model`:
it contains the functionality to generate new waveforms.

A `Model` is a multi-mode surrogate: it holds one `ModeModel` per
spherical-harmonic mode $(\ell, m)$, and sums their contributions,
weighted by the spin-weighted spherical harmonics, into the observer-frame
polarizations $h_+$ and $h_\times$.
The modes covered by the model shipped with the package are
$(2,2)$, $(2,1)$, $(3,3)$ and $(4,4)$.

The fastest way to access a functional instance of this object is to use
the pretrained one:
```python
from mlgw_bns import Model
model = Model.default_for_testing()
```

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

The mode mergers do not all happen at the same time, so they have to be
aligned before they are summed.
The per-mode time offsets which do this are predicted by a small regressor
trained alongside the model, `model.time_shifts_predictor`, and by default
`predict` queries it for us.
They can also be provided explicitly, as
`model.predict(frequencies, params, time_shifts=time_shifts)`, where
`time_shifts` is either one value per mode or a scalar, which is broadcast
to every mode.

To get the individual mode contributions instead of the summed
polarizations, use `model.predict_modes_dict(frequencies, params)`,
which takes the same optional `time_shifts` argument.

(new_model)=
## Making a new model

The simplest thing to do is to use the pretrained model provided with the package:
```python
m = Model.default_for_testing()
```

The way to create a new model from scratch is as follows:
```python
from mlgw_bns.higher_order_modes import Mode

m = Model(modes=[Mode(2, 2), Mode(2, 1)], filename="my_model")
m.generate()
m.set_hyper_and_train_nn()
```

For this to work `mlgw_bns` must be able to import
`EOBRun_module`.

We can then save this model to file with `m.save()`; afterwards we
will be able to recover it with
```python
m = Model(modes=[Mode(2, 2), Mode(2, 1)], filename="my_model")
m.load()
```
where it is crucial that the model name is the same --- the `load` method
only checks for files with the given name (in the current folder).
Each mode is stored in its own set of files, named
`{filename}_l{l}_m{m}`, with the shared time-shift predictor in
`{filename}_timeshifts.pkl`.

The hyperparameters used here are those provided with the package;
to perform an optimization see [](hyperparameter_optimization).

## Working with a single mode

A single `ModeModel` --- the object which actually does the machine learning
for one mode --- can be reached through the `mode_models` mapping:
```python
from mlgw_bns.higher_order_modes import Mode

mode_model = model.mode_models[Mode(2, 2)]
hp, hc = mode_model.predict(frequencies, params)
```
The signature is the same as `Model.predict`, minus the time shifts:
there is nothing to align a single mode against.

### The parameter ranges for a new model

These may change as the package is updated: the current ranges should be

- `total_mass`: between 2 and 4 solar masses
- `mass_ratio`: between 1 and 3
- `lambda_1` (tidal polarizability of the larger star): between 5 and 5000
- `lambda_2`: between 5 and 5000
- `chi_1` (aligned spin of the larger star): between -0.5 and 0.5
- `chi_2`: between -0.5 and 0.5
- frequencies: between 5 and 2048 Hz

These can be checked, once a `Model` object is initialized as described before,
by looking at the ranges of any of its modes:
```python
>>> print(model.mode_models[Mode(2, 2)].parameter_ranges)
ParameterRanges(mass_range=(2.0, 4.0), q_range=(1.0, 3.0), lambda1_range=(5.0, 5000.0), lambda2_range=(5.0, 5000.0), chi1_range=(-0.5, 0.5), chi2_range=(-0.5, 0.5))
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
