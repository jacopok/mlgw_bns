# Overview

The basic object provided by `mlgw_bns` is a `Model`:
it contains the functionality to generate new waveforms. 

The fastest way to access a functional instance of this object is to use 
the default one: 
```python
from mlgw_bns import Model

model = Model.default(filename='my_model')
```

Now we can predict waveforms; in order to do so however we 
pass the parameters through the `ParametersWithExtrinsic` class
(simpler APIs are planned for the future). 
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
        reference_phase=0.0,
        time_shift=0.0,
        total_mass=2.8,
    )

hp, hc = model.predict(frequencies, params)
```

