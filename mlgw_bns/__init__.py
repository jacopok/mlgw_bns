"""
The purpose of the ``mlgw_bns`` package is to speed up the generation of gravitational
waveforms for binary neutron star mergers, by training a machine 
learning model on effective-one-body waveforms.

The code can be found on the `github page <https://github.com/jacopok/mlgw_bns>`_.
"""
from importlib import metadata  # type: ignore

import toml  # type: ignore

from .model import Model, ParametersWithExtrinsic

__version__ = metadata.version(__package__)  # type: ignore
