[![CI Pipeline for mlgw_bns](https://github.com/jacopok/mlgw_bns/actions/workflows/ci.yaml/badge.svg)](https://github.com/jacopok/mlgw_bns/actions/workflows/ci.yaml)
[![Documentation Status](https://readthedocs.org/projects/mlgw-bns/badge/?version=latest)](https://mlgw-bns.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/mlgw-bns.svg)](https://badge.fury.io/py/mlgw-bns)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Machine Learning for Gravitational Waves from Binary Neutron Star mergers

This package's purpose is to speed up the generation of template gravitational waveforms for binary neutron star mergers by training a machine learning model on a dataset of waveforms generated with some physically-motivated surrogate.

It is able to reconstruct them with mismatches lower than 1/10000,
with as little as 1000 training waveforms; 
the accuracy then steadily improves as more training waveforms are used.

Currently, the only model used for training is [`TEOBResumS`](http://arxiv.org/abs/1806.01772),
but it is planned to introduce the possibility to use others.

The documentation can be found [here](https://mlgw-bns.readthedocs.io/en/latest).

<!-- ![dependencygraph](mlgw_bns.svg) -->

## Installation

To install the package, use
```bash
pip install mlgw-bns
```

For more details see [the documentation](https://mlgw-bns.readthedocs.io/en/latest/usage_guides/install.html).

## Inner workings

The main steps taken by `mlgw_bns` to train on a dataset are as follows:

- generate the dataset, consisting of EOB waveforms
- decompose the Fourier transforms of the waveforms into phase and amplitude
- downsample the dataset to a few thousand points
- compute the residuals of the EOB waveforms from PN ones
- apply a PCA to reduce the dimensionality to a few tens of real numbers
- train a neural network on the relation
    between the waveform parameters and the PCA components
    
After this, the model can reconstruct a waveform within its parameter space,
resampled at arbitrary points in frequency space.

In several of the training steps data-driven optimizations are performed:

- the points at which the waveforms are downsampled are not uniformly chosen:
    instead, a greedy downsampling algorithm determines them
- the hyperparameters for the neural network are optimized, according to both
    the time taken for the training and the estimated reconstruction error, 
    also varying the number of training waveforms available. 
    