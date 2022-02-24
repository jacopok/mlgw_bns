import numpy as np
import pytest
from EOBRun_module import EOBRunPy  # type: ignore

from mlgw_bns.dataset_generation import ParameterSet
from mlgw_bns.model import Model, ParametersWithExtrinsic


def test_teob_generator(dataset, teob_generator):
    params = ParametersWithExtrinsic(
        mass_ratio=1.0,
        lambda_1=500.0,
        lambda_2=50.0,
        chi_1=0.1,
        chi_2=-0.1,
        distance_mpc=1.0,
        inclination=0.0,
        reference_phase=0.0,
        time_shift=0.0,
        total_mass=2.8,
    )

    f_generator, waveform = teob_generator.effective_one_body_waveform(
        params.intrinsic(dataset)
    )

    teob_dict = params.teobresums_dict(dataset)
    assert teob_dict == params.intrinsic(dataset).teobresums()

    n_additional = 256
    f_0 = teob_dict["initial_frequency"]
    delta_f = teob_dict["df"]
    new_f0 = f_0 - delta_f * n_additional
    teob_dict["initial_frequency"] = new_f0

    f_eobrun, hpr, hpi, _, _ = EOBRunPy(teob_dict)
    f_eobrun, hpr2, hpi2, _, _ = EOBRunPy(teob_dict)

    assert np.allclose(f_generator, f_eobrun[n_additional:])
    assert np.allclose(hpr, hpr2)
    assert np.allclose(hpi, hpi2)

    assert np.allclose(waveform, (hpr - 1j * hpi)[n_additional:])


def test_geometric_units_normalization(dataset, teob_generator):

    params = ParametersWithExtrinsic(
        mass_ratio=1.0,
        lambda_1=500.0,
        lambda_2=50.0,
        chi_1=0.1,
        chi_2=-0.1,
        distance_mpc=1.0,
        inclination=0.0,
        reference_phase=0.0,
        time_shift=0.0,
        total_mass=2.8,
    )

    # f, waveform = teob_generator.effective_one_body_waveform(params)

    teob_dict = params.teobresums_dict(dataset)

    f, hpr, hpi, _, _ = EOBRunPy(teob_dict)
    wf = hpr - 1j * hpi

    teob_dict["use_geometric_units"] = "no"
    teob_dict["initial_frequency"] /= params.mass_sum_seconds
    teob_dict["srate_interp"] /= params.mass_sum_seconds
    teob_dict["df"] /= params.mass_sum_seconds

    f_phys, hpr_phys, hpi_phys, _, _ = EOBRunPy(teob_dict)
    wf_phys = hpr_phys - 1j * hpi_phys

    # constant can be calculated as
    # import astropy.units as u
    # import astropy.constants as ac
    # constant = ac.G**2 * u.Msun**2 / ac.c**5 / u.Mpc
    scaling = (
        params.total_mass ** 2
        / params.distance_mpc
        * params.intrinsic(dataset).eta
        * 2.35705224e-25
    )

    assert np.allclose(f_phys / f, 1 / params.mass_sum_seconds)

    assert np.allclose(np.unwrap(np.angle(wf)), np.unwrap(np.angle(wf_phys)), atol=1e-3)
    assert np.allclose(abs(wf_phys) / abs(wf), scaling)
