"""
The purpose of the ``mlgw_bns`` package is to speed up the generation of gravitational
waveforms for binary neutron star mergers, by training a machine 
learning model on effective-one-body waveforms.

The code can be found on the `github page <https://github.com/jacopok/mlgw_bns>`_.
"""


from .hyperparameter_optimization import HyperparameterOptimization
from .model import Model, ParametersWithExtrinsic
from .model_validation import ValidateModel
