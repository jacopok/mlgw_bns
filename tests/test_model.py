import pytest

from mlgw_bns import TEOBResumSModel
from mlgw_bns.dataset_generation import TEOBResumSGenerator
from mlgw_bns.model import Model
from mlgw_bns.model_validation import ValidatingModel


def test_validating_model_creation():

    vm = ValidatingModel()

    assert isinstance(vm.waveform_generator, TEOBResumSGenerator)


def test_teobresums_model_creation():

    tm = TEOBResumSModel()

    assert isinstance(tm.waveform_generator, TEOBResumSGenerator)
