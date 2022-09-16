from mlgw_bns.frequency_of_merger import frequency_of_merger


def test_frequency_of_merger_order_of_magnitude(parameters):

    f = frequency_of_merger(parameters)

    assert 0.02 < f
    assert 0.04 > f
