import numpy as np
import pytest

from mlgw_bns.dataset_generation import ParameterSet, TEOBResumSGenerator
from mlgw_bns.model import Model, ParametersWithExtrinsic
from mlgw_bns.model_validation import ValidateModel

from .test_model import random_parameters


@pytest.mark.requires_default
def test_default_model_pn_connection(default_model):
    par = random_parameters(default_model, 1)
    par.total_mass = default_model.dataset.total_mass

    f_low = default_model.dataset.effective_initial_frequency_hz

    assert f_low < 5.0

    freqs = np.linspace(f_low - 1e-8, f_low + 1e-8, num=30)

    with pytest.warns(UserWarning):
        hp, hc = default_model.predict(freqs, par)

    assert np.isclose(abs(hp)[0], abs(hp)[-1], atol=0, rtol=0.05)

    assert np.isclose(np.angle(hp)[0], np.angle(hp)[-1], atol=0.01, rtol=0)
