"""
The purpose of the ``mlgw_bns`` package is to speed up the generation of gravitational
waveforms for binary neutron star mergers, by training a machine 
learning model on effective-one-body waveforms.

The code can be found on the `github page <https://github.com/jacopok/mlgw_bns>`_.
"""
try:
    from importlib import metadata
except ImportError:
    # python <3.8 compatibility
    import importlib_metadata as metadata  # type: ignore

import toml  # type: ignore

from .model import Model, ParametersWithExtrinsic

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = toml.load("pyproject.toml")["tool"]["poetry"]["version"] + "dev"
