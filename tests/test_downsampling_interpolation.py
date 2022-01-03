import pytest


def test_indices_calculation(greedy_downsampling_training):
    indices_amp, indices_phi = greedy_downsampling_training.train(4)

    assert len(indices_amp) > 100
    assert len(indices_phi) > 100


def test_indices_validation(greedy_downsampling_training):
    errs_amp, errs_phi = greedy_downsampling_training.validate_downsampling(8, 8)

    assert (err < 1e-5 for err in errs_amp)
    assert (err < 1e-5 for err in errs_phi)
