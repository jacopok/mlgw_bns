import numpy as np
import pytest

from mlgw_bns.dataset_generation import ParameterSet, TEOBResumSGenerator
from mlgw_bns.model import (
    FrequencyTooHighError,
    FrequencyTooLowError,
    Model,
    ParametersWithExtrinsic,
)
from mlgw_bns.model_validation import ValidateModel

from .test_model import random_parameters

PHI_TOL = 1e-6
LOG_AMP_TOL = 1e-6


@pytest.mark.requires_default
@pytest.mark.parametrize("mass", [2.0, 2.8, 4.0])
@pytest.mark.parametrize("seed", list(range(2)))
def test_default_model_pn_connection(default_model, mass, seed):
    par = random_parameters(default_model, seed)
    par.total_mass = mass

    f_low = default_model.dataset.effective_initial_frequency_hz * (2.8 / mass)

    freqs = np.linspace(f_low - 1e-4, f_low + 1e-4, num=2000)

    hp, hc = default_model.predict(freqs, par)

    hp_gradient_aroud_f_low = np.ediff1d(np.log(abs(hp))[980:1005])
    avg_hp_gradient = np.average(hp_gradient_aroud_f_low[:18])

    assert np.allclose(
        hp_gradient_aroud_f_low, avg_hp_gradient, rtol=0, atol=LOG_AMP_TOL
    )

    unwrapped_phase_around_f_low = np.unwrap(np.angle(hp)[980:1005])
    delta_phi_between_points = np.ediff1d(unwrapped_phase_around_f_low)

    avg_delta_phi = np.average(delta_phi_between_points[:18])

    assert np.allclose(delta_phi_between_points, avg_delta_phi, rtol=0, atol=PHI_TOL)


@pytest.mark.requires_default
def test_default_model_extendibility():
    model = Model.default()
    params = ParametersWithExtrinsic.gw170817()

    assert model.extend_with_post_newtonian
    assert model.extend_with_zeros_at_high_frequency

    freqs = np.geomspace(1e-3, 1e4)
    hp, hc = model.predict(freqs, params)
    assert hp[-1] == 0.0j
    assert hc[-1] == 0.0j
    assert len(hp) == len(freqs)

    model.extend_with_post_newtonian = False

    with pytest.raises(FrequencyTooLowError):
        hp, hc = model.predict(freqs, params)

    model.extend_with_zeros_at_high_frequency = False

    with pytest.raises(FrequencyTooHighError):
        hp, hc = model.predict(np.linspace(20, 10000), params)


@pytest.mark.requires_default
def test_default_model_evaluation_does_not_depend_on_frequency_grid():
    model = Model.default()
    params = ParametersWithExtrinsic.gw170817()

    freqs = np.geomspace(1e-3, 1e3)

    hp, hc = model.predict(freqs, params)

    hps_single = []
    for freq in freqs:

        hp_one, hc_one = model.predict(np.array([freq]), params)
        hps_single.append(hp_one[0])

    for hp_c, hp_s in zip(hp, hps_single):
        assert np.isclose(abs(hp_c), abs(hp_s), atol=0.0, rtol=LOG_AMP_TOL)
