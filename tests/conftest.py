"""This module defines the `fixtures <https://docs.pytest.org/en/6.2.x/fixture.html>`__ 
which all other testing files can then use."""

import os

import h5py
import numpy as np
import pytest
from EOBRun_module import EOBRunPy  # type: ignore
from pytest_cases import fixture, fixture_union, parametrize  # type:ignore

from mlgw_bns import Model
from mlgw_bns.data_management import ParameterRanges
from mlgw_bns.dataset_generation import Dataset, TEOBResumSGenerator, WaveformParameters
from mlgw_bns.downsampling_interpolation import GreedyDownsamplingTraining


def pytest_addoption(parser):
    parser.addoption(
        "--defaultunavailable",
        action="store_true",
        default=False,
        help="skip tests which require the default model to be up to date",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_default: need the default model to run"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--defaultunavailable"):
        # the user did not specify: run all tests
        return
    default_unavailable = pytest.mark.skipif(
        reason="Skip if the default dataset needs to be updated"
    )
    for item in items:
        if "requires_default" in item.keywords:
            item.add_marker(default_unavailable)


@fixture(name="variable_dataset")
@parametrize(f_0=[30.0, 40.0])
def fixture_variable_dataset(f_0):
    """Dataset object with variable initial frequency"""
    return Dataset(
        initial_frequency_hz=f_0,
        srate_hz=4096.0,
        waveform_generator=TEOBResumSGenerator(EOBRunPy),
        multibanding=True,
        parameter_ranges=ParameterRanges(mass_range=(2.8, 2.8)),
    )


@pytest.fixture(name="dataset")
def fixture_dataset():
    """Dataset object with variable initial frequency (20 Hz)"""
    return Dataset(
        initial_frequency_hz=20.0,
        srate_hz=4096.0,
        waveform_generator=TEOBResumSGenerator(EOBRunPy),
        parameter_ranges=ParameterRanges(mass_range=(2.8, 2.8)),
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
    return TEOBResumSGenerator(EOBRunPy)


@pytest.fixture()
def greedy_downsampling_training(dataset):
    return GreedyDownsamplingTraining(dataset=dataset)


@pytest.fixture(scope="session")
def model():
    name = "test_model"
    model = Model(name, pca_components_number=20)
    yield model
    os.remove(model.file_arrays.filename)


@pytest.fixture(scope="session")
def generated_model(model):
    model.generate(8, 100, 100)
    yield model


@pytest.fixture(scope="session")
def trained_model(generated_model):
    generated_model.set_hyper_and_train_nn()
    yield generated_model


@pytest.fixture(scope="session")
def default_model():
    yield Model.default()


@pytest.fixture
def file():
    fname = "test_file.h5"
    file = h5py.File(fname, mode="a")
    yield file
    file.close()
    os.remove(fname)


fixture_union("all", ["variable_dataset", "variable_parameters"])


@pytest.fixture
def random_array():
    rng = np.random.default_rng(seed=1)

    return rng.multivariate_normal(
        np.zeros(100), cov=np.diag(1 / np.arange(1, 101) ** 2), size=(100,)
    )
