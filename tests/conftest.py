import numpy as np
import pytest

from mlgw_bns.dataset_generation import Dataset, WaveformParameters


@pytest.fixture
def dataset():
    return Dataset(
        filename="data",
        initial_frequency_hz=20.0,
        delta_f_hz=1.0 / 256.0,
        srate_hz=4096.0,
    )


@pytest.fixture
def parameters(dataset):
    return WaveformParameters(1, 300, 300, 0.3, 0.3, dataset)


@pytest.fixture
def frequencies(dataset):
    return np.arange(
        dataset.initial_frequency_hz, dataset.srate_hz / 2, dataset.delta_f_hz
    )
