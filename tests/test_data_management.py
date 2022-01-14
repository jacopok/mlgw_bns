import numpy as np
import pytest

from mlgw_bns.data_management import DownsamplingIndices


@pytest.mark.parametrize("saving_count", [1, 2])
def test_downsampling_indices_saving(file, saving_count):

    di = DownsamplingIndices([1, 2, 3], [4, 5, 6])

    for _ in range(saving_count):
        # running this once or twice should make no difference
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

    assert DownsamplingIndices.from_file(file) is None


def test_saving_and_retrieval_of_data_inside_model(generated_model):

    di = generated_model.downsampling_indices

    generated_model.save_arrays()

    di2 = DownsamplingIndices.from_file(generated_model.file_arrays)

    # Ugly workaround: the default __eq__ implemented by dataclasses
    # does not play well with arrays
    assert np.array_equal(di.amplitude_indices, di2.amplitude_indices)
    assert np.array_equal(di.phase_indices, di2.phase_indices)
