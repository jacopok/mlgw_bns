import numpy as np
import pytest

from mlgw_bns.data_management import DownsamplingIndices


def test_downsampling_indices_saving(file):

    di = DownsamplingIndices([1, 2, 3], [4, 5, 6])

    di.save_to_file(file)

    di2 = DownsamplingIndices.from_file(file)

    assert np.array_equal(di.amplitude_indices, di2.amplitude_indices)
    assert np.array_equal(di.phase_indices, di2.phase_indices)


def test_downsampling_indices_saving_different_name(file):

    di = DownsamplingIndices([1, 2, 3], [4, 5, 6])
    di.group_name = "different_name"

    di.save_to_file(file)

    di2 = DownsamplingIndices.from_file(file, group_name="different_name")

    assert np.array_equal(di.amplitude_indices, di2.amplitude_indices)
    assert np.array_equal(di.phase_indices, di2.phase_indices)

    with pytest.raises(KeyError):
        di3 = DownsamplingIndices.from_file(file)


def test_saving_and_retrieval_of_data_inside_model(model):

    model.generate(2)

    di = model.downsampling_indices

    model.save()

    di2 = DownsamplingIndices.from_file(model.file)

    # Ugly workaround: the default __eq__ implemented by dataclasses
    # does not play well with arrays
    assert np.array_equal(di.amplitude_indices, di2.amplitude_indices)
    assert np.array_equal(di.phase_indices, di2.phase_indices)
