import numpy as np
import pytest

from mlgw_bns.dataset_generation import (
    Dataset,
    ParameterSet,
    TEOBResumSGenerator,
    UniformParameterGenerator,
    WaveformGenerator,
    WaveformParameters,
)


def test_teobresums_generator_creation(teob_generator):
    assert isinstance(teob_generator, TEOBResumSGenerator)
    assert isinstance(teob_generator, WaveformGenerator)


# def test_waveform_prefactors(dataset):
# TODO
# prefactor = 3.668693487138444e-19
# check with the astropy code:
# ((ac.G * u.Msun / ac.c ** 3)**(5 / 6) / (u.Hz)**(7 / 6) * ac.c / u.Mpc / u.s).decompose().value


def test_waveform_parameters_equality_fails(parameters):
    assert parameters.almost_equal_to(3) is NotImplemented


def test_waveform_parameters_teobresums_output(parameters):
    """Test the output of a dictionary compatible with teobresums."""
    for key, val in {
        "q": 1,
        "Lambda1": 300,
        "Lambda2": 300,
        "chi1": 0.3,
        "chi2": 0.3,
        "M": 2.8,
        "distance": 1.0,
        "initial_frequency": 0.00027582749319999996,
        "use_geometric_units": 1,
        "interp_uniform_grid": 0,
        "domain": 1,
        "srate_interp": 0.056489470607359996,
        "df": 5.3872557265624996e-08,
        "interp_FD_waveform": 1,
        "inclination": 0.0,
        "output_hpc": 0,
        "output_dynamics": 0,
        "time_shift_FD": 1,
    }.items():
        assert np.isclose(val, parameters.teobresums[key])


def test_random_parameter_generation(dataset):

    u = UniformParameterGenerator(dataset=dataset)

    assert next(u).almost_equal_to(
        WaveformParameters(
            mass_ratio=1.9636394957426813,
            lambda_1=3585.8072979230906,
            lambda_2=2486.6021441939356,
            chi_1=0.3841186422717332,
            chi_2=0.07320957984815568,
            dataset=dataset,
        )
    )


def test_generated_waveform_length(variable_parameters, teob_generator):

    freq, waveform = teob_generator.effective_one_body_waveform(variable_parameters)

    dset = variable_parameters.dataset

    assert len(freq) == len(waveform)

    assert (
        len(freq)
        == int((dset.srate_hz / 2 - dset.initial_frequency_hz) / dset.delta_f_hz) + 1
    )


@pytest.mark.benchmark(group="waveform-generation", min_rounds=3)
def test_teob_generation_time(benchmark, parameters, teob_generator):
    benchmark(teob_generator.effective_one_body_waveform, params=parameters)


@pytest.mark.benchmark(group="waveform-generation", min_rounds=3)
def test_tf2_amp_generation_time(benchmark, parameters, frequencies, teob_generator):
    benchmark(teob_generator.post_newtonian_amplitude, parameters, frequencies)


@pytest.mark.benchmark(group="waveform-generation", min_rounds=3)
def test_tf2_phi_generation_time(benchmark, parameters, frequencies, teob_generator):
    benchmark(teob_generator.post_newtonian_phase, parameters, frequencies)


@pytest.mark.benchmark(group="residual-generation", min_rounds=3)
def test_dataset_generation_size_1(benchmark, variable_dataset):
    frequencies, params, res = benchmark(variable_dataset.generate_residuals, size=1)
    amp, phi = res

    assert np.isclose(
        min(frequencies),
        variable_dataset.hz_to_natural_units(variable_dataset.initial_frequency_hz),
    )
    assert (
        len(frequencies)
        == amp.shape[-1]
        == phi.shape[-1]
        == variable_dataset.waveform_length
    )

    assert isinstance(params, ParameterSet)


def test_changing_parameter_generation_ranges(dataset):
    parameter_generator = UniformParameterGenerator(
        q_range=(0.0, 1.0),
        lambda1_range=(0.0, 1.0),
        lambda2_range=(0.0, 1.0),
        chi1_range=(0.0, 1.0),
        chi2_range=(0.0, 1.0),
        dataset=dataset,
    )

    params = next(parameter_generator)

    for attr in ["mass_ratio", "lambda_1", "lambda_2", "chi_1", "chi_2"]:
        assert 0 <= getattr(params, attr) <= 1


def test_frequencies_match_with_eob(variable_parameters, teob_generator):

    freq, waveform = teob_generator.effective_one_body_waveform(variable_parameters)

    assert np.allclose(freq, variable_parameters.dataset.frequencies, atol=0.0)


def test_residuals_are_not_too_large(variable_parameters, teob_generator):

    amp_residuals, phi_residuals = teob_generator.generate_residuals(
        variable_parameters
    )

    length = len(amp_residuals)

    # The residuals overall should be below a relatively loose bound
    assert np.all(abs(amp_residuals) < 10)
    assert np.all(abs(phi_residuals) < 300)

    # their low-frequency parts should satisfy a stricter one
    assert np.all(abs(amp_residuals[: length // 2]) < 2)
    assert np.all(abs(phi_residuals[: length // 2]) < 120)

    # TODO can these be lowered?
