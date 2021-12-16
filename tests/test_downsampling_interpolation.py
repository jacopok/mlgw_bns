import pytest


def test_indices_calculation(downsampling_dataset):
    indices_amp, indices_phi = downsampling_dataset.calculate_downsampling(4)

    assert len(indices_amp) > 100
    assert len(indices_phi) > 100


def test_indices_validation(downsampling_dataset):
    errs_amp, errs_phi = downsampling_dataset.validate_downsampling(8, 8)

    assert (err < 1e-5 for err in errs_amp)
    assert (err < 1e-5 for err in errs_phi)
