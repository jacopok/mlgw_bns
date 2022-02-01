[![CI Pipeline for mlgw_bns](https://github.com/jacopok/mlgw_bns/actions/workflows/ci.yaml/badge.svg)](https://github.com/jacopok/mlgw_bns/actions/workflows/ci.yaml)
[![Documentation Status](https://readthedocs.org/projects/mlgw-bns/badge/?version=latest)](https://mlgw-bns.readthedocs.io/en/latest/?badge=latest)

# Machine Learning for Gravitational Waves from Binary Neutron Star mergers

This package's purpose is to speed up the generation of template gravitational waveforms for binary neutron star mergers by training a machine learning model on a dataset of waveforms generated with some physically-motivated surrogate.

It is able to reconstruct them with mismatches lower than 1/10000,
with as little as 1000 training waveforms; 
the accuracy then steadily drops as more training waveforms are used.

Currently, the only model used for training is [`TEOBResumS`](http://arxiv.org/abs/1806.01772),
but it is planned to introduce the possibility to use others.

The documentation is currently hosted [here](https://jacopok.github.io/index.html); 
in the future it will be moved to a better place and inserted into the CI pipeline.

![dependencygraph](mlgw_bns.svg)

## Installation

When the package will be published hopefully it will look like
```bash
pip install mlgw_bns
```
but for now one should clone this repo, install poetry and run 
```bash
poetry install
```
in the project folder.

For this to work, the `TEOBResumS` repository must be installed in the same folder 
as `mlgw_bns`:

```
some_folder/
|--- mlgw_bns/
    |--- mlgw_bns/
    |--- docs/
    |--- tests/
    |--- ...
|--- teobresums/
    |--- Python/
    |--- C/ 
    |--- ...
```

## Usage

To make sure everything is working properly one can run a pipeline
```bash
poetry run tox
```
which will install all missing dependencies, 
run tests and also build the documentation locally, in the folder `docs/html/`;
one can access it starting from `index.html`.

To only run the tests, do 
```bash
poetry run pytest
```

To only build the documentation, do
```bash
poetry run sphinx-build docs docs/html
```

Make a pretty dependency graph with 
```bash
poetry run pydeps mlgw_bns/
```

To make an html page showing the test coverage of the code, do
```bash
poetry run coverage html
```

There are pre-commit hooks which will clean up the code, 
format everything with `black`, check that there are no large files,
check that the typing is correct with `mypy`. 

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

