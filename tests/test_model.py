from mlgw_bns import Model


def test_model_creation():
    m = Model()

    assert isinstance(m, Model)
