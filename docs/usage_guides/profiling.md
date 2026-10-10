# Profiling the model evaluation

Within the test suite for `mlgw_bns` the evaluation time for  
a prediction by a trained model is checked thanks to `pytest-benchmark`; 
however one might wish to get more granular information
about which functions are taking the longest to run. 

In order to achieve this, we can make use of the `cProfile` module, combined
with the `snakeviz` visualization tool. 
Both of these dependencies are installed when running
```bash
uv sync
```

We write a script with the following imports:
```python
from mlgw_bns import Model
from mlgw_bns.mode_model import ParametersWithExtrinsic

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
we need a trained model, which we can obtain as outlined in [](new_model).
Let us call our model `m`.

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
uv run snakeviz prediction_profile.prof 
```
from the command line, and a local webpage containing an interactive visualization 
of the execution times of the various internal functions.

The expected result, for the packaged seven-mode model and a few hundred
frequencies, is that almost all of the time goes to
`mode_model.ModeModel.predict_amplitude_phase`, once for each mode (and once
more for the $(2,2)$, which also sets the merger reference,
`mode_model.ModeModel.merger_reference`). Of that:
- about 40% is the regressor of each mode (`neural_network.KernelRidgeNetwork.predict`,
  through `mode_model.ModeModel.predict_residuals_bulk`), which evaluates a
  kernel on every training waveform, so that it grows with the training set
  (see [](cluster-training) for a perceptron, whose cost does not);
- about 30% is the cubic-spline resampling of the amplitude and phase from
  the downsampling nodes to the requested frequencies
  (`downsampling_interpolation.resample`, through scipy's `CubicSpline`);
- about 15% is the post-Newtonian amplitude and phase which the residuals
  are taken against (`pn_modes`, `taylorf2`).

With thousands of frequencies the resampling and the post-Newtonian
expressions take a larger share.
- any remaining time required should be comparatively small.