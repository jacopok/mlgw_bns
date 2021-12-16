"""This module defines the `fixtures <https://docs.pytest.org/en/6.2.x/fixture.html>`__ 
which all other testing files can then use."""

import numpy as np
import pytest
from pytest_cases import fixture, fixture_union, parametrize  # type:ignore

from mlgw_bns.dataset_generation import Dataset, TEOBResumSGenerator, WaveformParameters
from mlgw_bns.downsampling_interpolation import DownsamplingTrainingDataset


@fixture(name="variable_dataset")
@parametrize(f_0=[10.0, 20.0, 30.0])
def fixture_variable_dataset(f_0):
    """Dataset object with variable initial frequency"""
    return Dataset(
        filename="data",
        initial_frequency_hz=f_0,
        delta_f_hz=1.0 / 256.0,
        srate_hz=4096.0,
    )


@pytest.fixture(name="dataset")
def fixture_dataset():
    """Dataset object with variable initial frequency (20 Hz)"""
    return Dataset(
        filename="data",
        initial_frequency_hz=20.0,
        delta_f_hz=1.0 / 256.0,
        srate_hz=4096.0,
    )


@fixture
@parametrize(
    wf_params=[
        (1, 300, 300, 0.3, 0.3),
        (1.9, 5000, 200, -0.5, 0.1),
        (1.1, 10, 20, 0.0, 0.4),
    ]
)
def variable_parameters(wf_params, variable_dataset):
    """Parametrized WaveformParameters object.
    `wf_params` are given in the same order as the
    inizialization of `WaveformParameters`.
    """
    return WaveformParameters(*wf_params, variable_dataset)


@pytest.fixture
def parameters(dataset):
    """Fixed WaveformParameters object."""
    return WaveformParameters(1, 300, 300, 0.3, 0.3, dataset)


@pytest.fixture
def frequencies(dataset):
    """Frequency array of the same frequencies as the dataset."""
    return np.arange(
        dataset.initial_frequency_hz, dataset.srate_hz / 2, dataset.delta_f_hz
    )


@pytest.fixture
def teob_generator():
    """Waveform generator based in TEOBResumS."""
    return TEOBResumSGenerator()


@pytest.fixture()
def downsampling_dataset():
    return DownsamplingTrainingDataset(
        degree=3,
        tol=1e-6,
        filename="data",
        initial_frequency_hz=20.0,
        delta_f_hz=1.0 / 256.0,
        srate_hz=4096.0,
    )


fixture_union("all", ["variable_dataset", "variable_parameters"])
