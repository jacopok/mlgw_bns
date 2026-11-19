"""This module defines the `fixtures <https://docs.pytest.org/en/6.2.x/fixture.html>`__ 
which all other testing files can then use."""

import glob
import os
import warnings

import h5py
import numpy as np
import pytest
from EOBRun_module import EOBRunPy  # type: ignore

from mlgw_bns import Model, ModeModel, ParametersWithExtrinsic
from mlgw_bns.model import MODELS_AVAILABLE, ModelAccuracyWarning
from mlgw_bns.dataset_generation import Dataset, TEOBResumSGenerator, WaveformParameters
from mlgw_bns.downsampling_interpolation import GreedyDownsamplingTraining
from mlgw_bns.higher_order_modes import Mode


@pytest.fixture(name="variable_dataset", params=[30.0, 40.0])
def fixture_variable_dataset(request):
    """Dataset object with variable initial frequency"""
    return Dataset(
        initial_frequency_hz=request.param,
        srate_hz=4096.0,
        waveform_generator=TEOBResumSGenerator(EOBRunPy),
        multibanding=True,
    )


@pytest.fixture(name="dataset")
def fixture_dataset():
    """Dataset object with variable initial frequency (20 Hz)"""
    return Dataset(
        initial_frequency_hz=20.0,
        srate_hz=4096.0,
        waveform_generator=TEOBResumSGenerator(EOBRunPy),
    )


@pytest.fixture(
    params=[
        (1, 300, 300, 0.3, 0.3),
        (1.9, 5000, 200, -0.5, 0.1),
        (1.1, 10, 20, 0.0, 0.4),
    ]
)
def variable_parameters(request, variable_dataset):
    """Parametrized WaveformParameters object.
    The parameters are given in the same order as the
    inizialization of `WaveformParameters`.
    """
    return WaveformParameters(*request.param, variable_dataset)


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


@pytest.fixture(scope="session", params=MODELS_AVAILABLE)
def default_model(request):
    """Each pretrained multi-mode model shipped with the package."""
    with warnings.catch_warnings():
        # its known caveat, tested in test_model.py
        warnings.simplefilter("ignore", ModelAccuracyWarning)
        model = Model.default_for_testing(request.param)
    yield model


#: Tests of ``default_model`` that a shipped model is known to fail, with
#: why. Strict: a model that starts passing one (after a retrain) fails the
#: suite until it is taken off this list.
KNOWN_FAILURES = {
    "default_hom_lambda12000": (
        {
            "test_default_model_merger_reference_matches_teobresums",
            "test_merger_reference_matches_teobresums_at_low_lambda_1",
            "test_full_waveform_mismatch_is_flat_in_total_mass",
            "test_matches_teobresums",
        },
        "trained on Lambda up to 12000 with as many waveforms as the others, "
        "it is not accurate enough in the late inspiral to predict the merger "
        "(0.2-0.5 ms early at small lambda_1 and large q, above 0.2 ms for 5% "
        "of its range) and is about ten times less accurate in mismatch "
        "(MODEL_CAVEATS)",
    ),
}


def pytest_collection_modifyitems(items):
    for item in items:
        callspec = getattr(item, "callspec", None)
        model_name = callspec.params.get("default_model") if callspec else None
        if model_name in KNOWN_FAILURES:
            tests, reason = KNOWN_FAILURES[model_name]
            if item.originalname in tests:
                item.add_marker(pytest.mark.xfail(reason=reason, strict=True))


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
        coalescence_phase=0.0,
        merger_time=0.0,
        total_mass=2.8,
    )
