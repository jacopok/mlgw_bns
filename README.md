[![CI Pipeline for mlgw_bns](https://github.com/jacopok/mlgw_bns/actions/workflows/ci.yaml/badge.svg)](https://github.com/jacopok/mlgw_bns/actions/workflows/ci.yaml)
[![Documentation Status](https://readthedocs.org/projects/mlgw-bns/badge/?version=latest)](https://mlgw-bns.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/mlgw-bns.svg)](https://badge.fury.io/py/mlgw-bns)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage Status](https://coveralls.io/repos/github/jacopok/mlgw_bns/badge.svg?branch=master)](https://coveralls.io/github/jacopok/mlgw_bns?branch=master)
[![Downloads](https://pepy.tech/badge/mlgw-bns/week)](https://pepy.tech/project/mlgw-bns)

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

## Changelog

Changes across versions are documented in the [CHANGELOG](https://github.com/jacopok/mlgw_bns/blob/master/CHANGELOG.md).

## Reference

The reference paper is [this one](https://arxiv.org/abs/2210.15684), currently 
only on arxiv.