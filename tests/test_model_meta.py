import numpy as np
import pytest
from EOBRun_module import EOBRunPy  # type: ignore

from mlgw_bns.dataset_generation import ParameterSet, TEOBResumSGenerator
from mlgw_bns.model import Model, ParametersWithExtrinsic
from mlgw_bns.model_validation import ValidateModel


def test_model_str(trained_model):
    assert str(trained_model) == (
        "Model(filename=test_model, "
        "auxiliary_data_available=True, nn_available=True, "
        "training_dataset_available=True, "
        "waveforms_available = 100, "
        "parameter_ranges=ParameterRanges(mass_range=(2.0, 4.0), "
        "q_range=(1.0, 3.0), lambda1_range=(5.0, 5000.0), "
        "lambda2_range=(5.0, 5000.0), chi1_range=(-0.5, 0.5), "
        "chi2_range=(-0.5, 0.5)))"
    )


def test_new_model_without_name():
    m = Model()

    with pytest.raises(ValueError):
        m.filename_arrays
    with pytest.raises(ValueError):
        m.filename_nn
    with pytest.raises(ValueError):
        m.filename_hyper

    assert not m.nn_available


def test_model_saving_downsampling_indices_and_pca(generated_model):

    generated_model.save()

    with generated_model.file_arrays as f:
        assert "downsampling/amplitude_indices" in f
        assert f["downsampling/amplitude_indices"][0] == 0
        assert "principal_component_analysis/eigenvalues" in f


def test_model_metadata_saving(generated_model):

    generated_model.save()
    saved_metadata_dict = generated_model.load_metadata()
    model_metadata_dict = generated_model.metadata_dict

    assert model_metadata_dict == saved_metadata_dict
    assert "initial_frequency_hz" in model_metadata_dict
    assert "srate_hz" in model_metadata_dict


def test_model_metadata_loading(generated_model):

    generated_model.save()
    generated_model.initial_frequency_hz = 1000.0
    generated_model.parameter_ranges.mass_range = (1.0, 5.0)
    generated_model.multibanding = False

    generated_model.load()

    assert generated_model.initial_frequency_hz < 20.0
    assert generated_model.parameter_ranges.mass_range == (2.0, 4.0)
    assert generated_model.multibanding == True


def test_downsampling_indices_characteristics(generated_model):

    generated_model.save()

    with generated_model.file_arrays as f:
        assert 50 < f["downsampling/amplitude_indices"][10] < 10_000

        # this holds when training on residuals
        # assert 20_000 < file["downsampling/amplitude_indices"][10] < 30_000
