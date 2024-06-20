from mlgw_bns.data_management import ParameterRanges
from mlgw_bns import Model
import numpy as np

ranges = ParameterRanges()

m = Model(parameter_ranges=ranges)

_, training_params, training_residuals = m.dataset.generate_residuals(5_000, flatten_phase=False)

training_timeshifts = training_residuals.flatten_phase(m.dataset.frequencies_hz)

np.save("time_shifts_experiments/data/timeshifts_5k.npy", training_timeshifts)
np.save("time_shifts_experiments/data/params_5k.npy", training_params.parameter_array)