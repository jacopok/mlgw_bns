import pytest
import numpy as np

from mlgw_bns.dataset_generation import (
    TEOBResumSGenerator,
    WaveformGenerator,
    Dataset,
    WaveformParameters,
    UniformParameterGenerator,
)


def test_teobresums_generator_creation():
    tg = TEOBResumSGenerator()
    assert isinstance(tg, TEOBResumSGenerator)
    assert isinstance(tg, WaveformGenerator)


def test_waveform_prefactors(dataset):

    # TODO

    prefactor = 3.668693487138444e-19
    # check with the astropy code:
    # ((ac.G * u.Msun / ac.c ** 3)**(5 / 6) / (u.Hz)**(7 / 6) * ac.c / u.Mpc / u.s).decompose().value


def test_waveform_parameters_teobresums_output(parameters):

    assert parameters.teobresums == {
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
    }


def test_random_parameter_generation(dataset):

    u = UniformParameterGenerator(dataset)

    next(u)
