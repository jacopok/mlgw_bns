[![CI Pipeline for mlgw_bns](https://github.com/jacopok/mlgw_bns/actions/workflows/ci.yaml/badge.svg)](https://github.com/jacopok/mlgw_bns/actions/workflows/ci.yaml)

# Machine Learning for Gravitational Waves from Binary Neutron Star mergers

This package's purpose is to speed up the generation of template gravitational waveforms for binary neutron star mergers by training a machine learning model on a dataset of waveforms generated with some physically-motivated surrogate.

It is able to reconstruct them with mismatches $\bar{\mathcal{F}} \lesssim 10^{-4}$
with as little as $\sim 1000$ training waveforms; the number then steadily drops as more training waveforms are used.

Currently, the only model used for training is [`TEOBResumS`](http://arxiv.org/abs/1806.01772),
but it is planned to introduce the possibility to use others.

## Installation

Hopefully,
```python
conda install mlgw_bns
```
TODO

## Usage

TODO

## Inner workings

The main steps taken by `mlgw_bns` to train on a dataset are as follows:

- generate the dataset
- decompose the Fourier transforms of the waveforms into phase and amplitude
- downsample the dataset to a few thousand points
- apply a PCA to reduce the dimensionality to a few tens of real numbers
- train a neural network on the relation
    between the waveform parameters and the PCA components

In several of these steps data-driven optimizations are performed:

- the points at which the waveforms are downsampled are not uniformly chosen:
    instead, a greedy downsampling algorithm determines them
- the PCA is trained on a separate downsampled dataset, which is then thrown out
- the hyperparameters for the neural network are optimized according to both
    the time taken for the training and the estimated reconstruction error

