"""This module defines the `fixtures <https://docs.pytest.org/en/6.2.x/fixture.html>`__ 
which all other testing files can then use."""

import glob
import os

import h5py
import numpy as np
import pytest
from EOBRun_module import EOBRunPy  # type: ignore
from pytest_cases import fixture, fixture_union, parametrize  # type:ignore

from mlgw_bns import Model, ModeModel, ParametersWithExtrinsic
from mlgw_bns.data_management import ParameterRanges
from mlgw_bns.dataset_generation import Dataset, TEOBResumSGenerator, WaveformParameters
from mlgw_bns.downsampling_interpolation import GreedyDownsamplingTraining
from mlgw_bns.higher_order_modes import Mode


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
def mode_model():
    """Untrained single-mode model, writing to files in the cwd."""
    name = "test_mode_model"
    mode_model = ModeModel(name, pca_components_number=20)
    yield mode_model

    for filename in [
        mode_model.filename_arrays,
        mode_model.filename_metadata,
        mode_model.filename_nn,
        mode_model.filename_timeshifts,
    ]:
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass


@pytest.fixture(scope="session")
def generated_mode_model(mode_model):
    mode_model.generate(8, 100, 100)
    yield mode_model


@pytest.fixture(scope="session")
def trained_mode_model(generated_mode_model):
    generated_mode_model.set_hyper_and_train_nn()
    yield generated_mode_model


@pytest.fixture(scope="session")
def default_model():
    """The pretrained multi-mode model shipped with the package."""
    yield Model.default_for_testing()


@pytest.fixture(scope="session")
def model():
    """Untrained multi-mode model, writing to files in the cwd."""
    name = "test_model"
    model = Model(
        modes=[Mode(2, 2), Mode(2, 1)], filename=name, pca_components_number=10
    )
    yield model

    for filename in glob.glob(f"{name}*"):
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass


@pytest.fixture(scope="session")
def generated_model(model):
    model.generate(6, 30, 30)
    yield model


@pytest.fixture(scope="session")
def trained_model(generated_model):
    generated_model.set_hyper_and_train_nn()
    yield generated_model


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


@pytest.fixture
def parameters_with_extrinsic():
    return ParametersWithExtrinsic(
        mass_ratio=1.2,
        lambda_1=500.0,
        lambda_2=50.0,
        chi_1=0.1,
        chi_2=-0.1,
        distance_mpc=1.0,
        inclination=0.0,
        reference_phase=0.0,
        time_shift=0.0,
        total_mass=2.8,
    )
