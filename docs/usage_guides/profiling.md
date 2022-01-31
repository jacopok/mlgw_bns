# Profiling the model evaluation

Within the test suite for `mlgw_bns` the evaluation time for  
a prediction by a trained model is checked thanks to `pytest-benchmark`; 
however one might wish to get more granular information
about which functions are taking the longest to run. 

In order to achieve this, we can make use of the `cProfile` module, combined
with the `snakeviz` visualization tool. 
Both of these dependencies should be installed when running the command
```bash
poetry install
```
without the `--no-dev` option. 

We write a script with the following imports:
```python
from mlgw_bns import Model
from mlgw_bns.model import ParametersWithExtrinsic

import cProfile
import pstats
```

and define a utility function for the profilation of a function's execution:
```python
def profile(func, *args, **kwargs):
    
    with cProfile.Profile() as pr:
        func(*args, **kwargs)
    
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    return stats
```

Now, we are ready to start profiling: 
we need a trained model, so we can either do
```python 
m = Model('your_model_name')
m.load()
```
if the model was already generated previously, or 
```python
m = Model('your_model_name')
m.generate()
m.set_hyper_and_train_nn()
```
where the default values for the size of the generated dataset are used, or 
```python
m = Model.default('your_model_name')
```
to make use of the default model provided with the package.

We also need a set of parameters for the generated waveform: 
an example set is given here.
```python
params = ParametersWithExtrinsic(
    mass_ratio=1.0,
    lambda_1=500,
    lambda_2=50,
    chi_1=0.1,
    chi_2=-0.1,
    distance_mpc=1.0,
    inclination=0.0,
    total_mass=2.8,
)
```

We also need an array of frequencies at which the new waveform will need to be computed:
the standard, FFT-grid frequency set is provided by 
`m.dataset.frequencies_hz`, so we can use this as a baseline. 
Its only issue is that it is very dense, so we can downsample by a certain amount, 
for example defining
```python
frequencies = m.dataset.frequencies_hz[::1024]
```

We are almost ready to profile: 
one issue is the fact that several functions in the 
prediction pipeline are decorated to make use of `numba`'s just-in-time compilation,
or they employ caching, therefore the first run of the prediction function will 
be very slow (on the order of a few seconds) compared to the speed 
the pipeline can achieve. 

To account for this, we need to make a dry run of the prediction function before
profiling it. 
The profilation code will therefore look like:
```python
m.predict(frequencies, params)

stats = profile(m.predict, frequencies, params)
stats.dump_stats('prediction_profile.prof')
```

This will generate a profile file which is not human-readable, 
but it can be nicely visualized with the `snakeviz` utility: 
simply run 
```bash
poetry run snakeviz prediction_profile.prof 
```
from the command line, and a local webpage containing an interactive visualization 
of the execution times of the various internal functions.

The expected result, if the length of the required frequencies 
is of the order of a few thousands (as is needed, for example, with reduced order quadratures), 
is that: 
- most of the time will be taken by interpolation: the internal function 
    is called `model.cartesian_waveforms_at_frequencies`, which calls 
    `downsampling_interpolation.resample` twice (once for amplitude, once for phase),
    and which in turns calls the scipy low-level functions `splrep` and `splev`;
- some time is taken by the prediction of the waveforms (`model.Model.predict_waveforms_bulk`), of which
  - some time (about half) is taken by the prediction of the residuals with the neural network and PCA:
        `model.Model.predict_residuals_bulk`;
  - some time (about half) is taken by the evaluation of the post-Newtonian amplitude and phase:
        `dataset_generation.Dataset.recompose_residuals`;
- any remaining time required should be comparatively small.