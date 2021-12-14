import pytest

from mlgw_bns import TEOBResumSModel
from mlgw_bns.model import Model
from mlgw_bns.model_validation import ValidatingModel
from mlgw_bns.dataset_generation import TEOBResumSGenerator


def test_model_creation_fails():

    with pytest.raises(TypeError):
        m = Model()


def test_validating_model_creation():

    vm = ValidatingModel()

    assert isinstance(vm.waveform_generator, TEOBResumSGenerator)


def test_teobresums_model_creation():

    tm = TEOBResumSModel()

    assert isinstance(tm.waveform_generator, TEOBResumSGenerator)


def test_teob_model_generator(benchmark, parameters):

    tm = TEOBResumSModel()

    benchmark(tm.waveform_generator.effective_one_body_waveform, params=parameters)


def test_tf2_amp_model_generator(benchmark, parameters, frequencies):

    tm = TEOBResumSModel()

    benchmark(tm.waveform_generator.post_newtonian_amplitude, parameters, frequencies)


def test_tf2_phi_model_generator(benchmark, parameters, frequencies):

    tm = TEOBResumSModel()

    benchmark(tm.waveform_generator.post_newtonian_phase, parameters, frequencies)
