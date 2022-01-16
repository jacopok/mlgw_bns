import numpy as np
import pytest
from EOBRun_module import EOBRunPy  # type: ignore

from mlgw_bns.model import ExtendedWaveformParameters, Model


@pytest.mark.xfail
def test_teob_generator(dataset, teob_generator):
    params = ExtendedWaveformParameters(
        mass_ratio=1.0,
        lambda_1=500.0,
        lambda_2=50.0,
        chi_1=0.1,
        chi_2=-0.1,
        dataset=dataset,
        distance_mpc=1.0,
        inclination=0.0,
        reference_phase=0.0,
        time_shift=0.0,
        total_mass=2.8,
    )

    f_generator, waveform = teob_generator.effective_one_body_waveform(params)

    teob_dict = params.teobresums

    f_eobrun, hpr, hpi, _, _ = EOBRunPy(teob_dict)

    assert np.allclose(f_generator, f_eobrun)
    assert np.allclose(waveform[200:], (hpr - 1j * hpi)[200:])


def test_geometric_units_normalization(dataset, teob_generator):

    params = ExtendedWaveformParameters(
        mass_ratio=1.0,
        lambda_1=500.0,
        lambda_2=50.0,
        chi_1=0.1,
        chi_2=-0.1,
        dataset=dataset,
        distance_mpc=1.0,
        inclination=0.0,
        reference_phase=0.0,
        time_shift=0.0,
        total_mass=2.8,
    )

    f, waveform = teob_generator.effective_one_body_waveform(params)

    teob_dict = params.teobresums

    teob_dict["use_geometric_units"] = 0
    teob_dict["initial_frequency"] /= params.mass_sum_seconds
    teob_dict["srate_interp"] /= params.mass_sum_seconds
    teob_dict["df"] /= params.mass_sum_seconds
