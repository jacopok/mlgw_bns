import numpy as np
import pytest

from mlgw_bns.dataset_generation import ParameterSet
from mlgw_bns.model import ParametersWithExtrinsic


def test_wrong_number_of_parameters_fail():

    with pytest.raises(AssertionError):
        ParameterSet(np.array([[1, 2, 3]]))


def test_slicing_of_paramter_object():

    p = ParameterSet(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))

    p2 = p[:1]

    assert isinstance(p2, ParameterSet)


def test_170817_params():
    par = ParametersWithExtrinsic.gw170817()

    assert par.total_mass == 2.8
