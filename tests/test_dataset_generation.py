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
        "use_geometric_units": "yes",
        "interp_uniform_grid": "no",
        "domain": 1,
        "srate_interp": 0.056489470607359996,
        "df": 5.3872557265624996e-08,
        "interp_FD_waveform": 1,  # change this to "yes" as well? is it ignored?
        "inclination": 0.0,
        "output_hpc": "no",
        "time_shift_FD": "yes",
    }.items():
        if isinstance(val, float):
            assert np.isclose(val, parameters.teobresums[key])
        else:
            assert val == parameters.teobresums[key]


def test_random_parameter_generation(dataset):

    u = UniformParameterGenerator(dataset=dataset)

    new_parameter_set = next(u)
    assert new_parameter_set.almost_equal_to(
        WaveformParameters(
            mass_ratio=1.3060149999161021,
            lambda_1=3355.669164496995,
            lambda_2=1458.659021999292,
            chi_1=0.4387661392393323,
            chi_2=0.44998480116559636,
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


def test_parameter_generator_changes_seed(dataset):
    """Making two parameter generators should spawn
    different rngs with different seeds."""
    gen1 = dataset.make_parameter_generator()
    gen2 = dataset.make_parameter_generator()

    # a normal distribution, but it could be anything
    assert gen1.rng.normal() != gen2.rng.normal()
    # more importantly, the parameters generated should be the same
    assert not np.allclose(next(gen1).array, next(gen2).array)


def test_true_waveforms_in_validation(dataset, parameters):

    param_set = ParameterSet.from_list_of_waveform_parameters([parameters])
    waveforms = dataset.generate_waveforms_from_params(param_set)
    cartesian_wf_from_polar = waveforms.amplitudes[0] * np.exp(1j * waveforms.phases[0])

    f, cartesian_wf_direct = dataset.waveform_generator.effective_one_body_waveform(
        parameters
    )

    assert np.allclose(abs(cartesian_wf_direct), abs(cartesian_wf_from_polar))
    assert np.allclose(
        np.unwrap(np.angle(cartesian_wf_direct)) - np.angle(cartesian_wf_direct[0]),
        np.unwrap(np.angle(cartesian_wf_from_polar)),
    )
