import pytest

from mlgw_bns.dataset_generation import TEOBResumSGenerator
from mlgw_bns.model import Model
from mlgw_bns.model_validation import ValidatingModel


def test_validating_model_creation():

    vm = ValidatingModel("test")

    assert isinstance(vm.waveform_generator, TEOBResumSGenerator)
