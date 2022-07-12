import numpy as np
import pytest

from mlgw_bns.dataset_generation import ParameterSet, TEOBResumSGenerator
from mlgw_bns.model import Model, ParametersWithExtrinsic
from mlgw_bns.model_validation import ValidateModel

from .test_model import random_parameters


@pytest.mark.xfail
@pytest.mark.requires_default
def test_default_model_pn_connection(default_model):
    par = random_parameters(default_model, 1)
    par.total_mass = 2.4

    f_low = default_model.dataset.effective_initial_frequency_hz * (2.8 / 2.4)

    freqs = np.linspace(f_low - 1e-4, f_low + 1e-4, num=2000)

    with pytest.warns(UserWarning):
        hp, hc = default_model.predict(freqs, par)

    assert np.isclose(abs(hp)[999], abs(hp)[1001], atol=0, rtol=0.05)

    assert np.isclose(np.angle(hp)[999], np.angle(hp)[1001], atol=0.01, rtol=0)
