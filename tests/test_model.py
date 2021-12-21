import pytest

from mlgw_bns.dataset_generation import TEOBResumSGenerator
from mlgw_bns.model import Model
from mlgw_bns.model_validation import ValidatingModel


def test_validating_model_creation():

    vm = ValidatingModel("test")

    assert isinstance(vm.waveform_generator, TEOBResumSGenerator)


def test_model_generation(model):

    model.generate(4)

    with model.file as file:
        assert "downsampling_indices/amplitude_indices" in file
        assert file["downsampling_indices/amplitude_indices"][0] == 0
        assert 25_000 < file["downsampling_indices/amplitude_indices"][10] < 30_000
