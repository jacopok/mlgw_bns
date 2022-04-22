import numpy as np
import pytest

from mlgw_bns import Model
from mlgw_bns.dataset_generation import ParameterSet, UniformParameterGenerator
from mlgw_bns.fixed_dataset_training import (
    FixedParameterGenerator,
    FixedWaveformGenerator,
    IndexedWaveformParameters,
    make_fixed_generation_pair,
)


@pytest.fixture
def fixed_generator_pair(dataset):

    gen = dataset.make_parameter_generator(seed=1)
    params = ParameterSet.from_parameter_generator(gen, 5)
    waveforms = dataset.generate_waveforms_from_params(params)
    return make_fixed_generation_pair(dataset.frequencies, params, waveforms)


def test_fixed_generator_parameters(fixed_generator_pair):
    fixed_parameter_generator, fixed_waveform_generator = fixed_generator_pair

    params = next(fixed_parameter_generator)
    assert isinstance(params, IndexedWaveformParameters)
    assert params.index == 0
    assert isinstance(params.dataset.waveform_generator, FixedWaveformGenerator)


def test_fixed_generator_residuals(fixed_generator_pair):
    fixed_parameter_generator, fixed_waveform_generator = fixed_generator_pair

    dataset = fixed_parameter_generator.dataset
    freqs, params, residuals = dataset.generate_residuals(5, flatten_phase=False)

    assert np.allclose(
        params.parameter_array, fixed_parameter_generator.parameter_set.parameter_array
    )

    waveforms = dataset.recompose_residuals(residuals, params)
    assert np.allclose(
        waveforms.amplitudes, fixed_waveform_generator.waveforms.amplitudes
    )
    assert np.allclose(waveforms.phases, fixed_waveform_generator.waveforms.phases)


def test_fixed_waveform_generation_limit(fixed_generator_pair):
    fixed_parameter_generator, fixed_waveform_generator = fixed_generator_pair

    dataset = fixed_parameter_generator.dataset
    freqs, params, residuals = dataset.generate_residuals(6, flatten_phase=False)
    fixed_parameter_generator.loop = False
    with pytest.raises(StopIteration):
        freqs, params, residuals = dataset.generate_residuals(6, flatten_phase=False)


def test_arbitrary_index_waveform_recovery(fixed_generator_pair):
    fixed_parameter_generator, fixed_waveform_generator = fixed_generator_pair

    params = list(fixed_parameter_generator.waveform_parameters)

    freqs = np.linspace(20, 2048, num=1000)

    freq1, amp1, phase1 = fixed_waveform_generator.effective_one_body_waveform(
        params[2], frequencies=freqs
    )
    freq2, amp2, phase2 = fixed_waveform_generator.effective_one_body_waveform(
        params[2], frequencies=freqs
    )

    assert np.allclose(freq1, freq2)
    assert np.allclose(freq1, freqs)
    assert np.allclose(amp1, amp2)
    assert np.allclose(phase1, phase2)


def test_training_model_on_fixed_data(fixed_generator_pair):
    fixed_parameter_generator, fixed_waveform_generator = fixed_generator_pair

    model = Model(
        waveform_generator=fixed_waveform_generator,
        parameter_generator=fixed_parameter_generator,
        pca_components_number=5,
    )
    model.generate(5, 5, 5)
