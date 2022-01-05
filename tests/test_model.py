import pytest

from mlgw_bns.dataset_generation import TEOBResumSGenerator
from mlgw_bns.model import Model
from mlgw_bns.model_validation import ValidatingModel


def test_validating_model_creation():

    vm = ValidatingModel("test")

    assert isinstance(vm.waveform_generator, TEOBResumSGenerator)


def test_model_saving(generated_model):

    generated_model.save()

    with generated_model.file as file:
        assert "downsampling/amplitude_indices" in file
        assert file["downsampling/amplitude_indices"][0] == 0
        assert 25_000 < file["downsampling/amplitude_indices"][10] < 30_000

        assert "principal_component_analysis/eigenvalues" in file
