import numpy as np
import pytest


def test_indices_calculation(greedy_downsampling_training):
    indices_amp, indices_phi = greedy_downsampling_training.train(4)

    assert len(indices_amp) > 100
    assert len(indices_phi) > 100


def test_indices_validation(greedy_downsampling_training):
    r"""The reconstruction error on fresh waveforms tracks the training tolerance.

    The greedy training adds sample points until the :math:`L_\infty`
    interpolation error on its *training* waveforms falls below
    ``tol_amp``/``tol_phi``. Waveforms it has not seen are not guaranteed
    to sit under the same bound, but they should not be far above it, so
    that is what is checked here, with a factor of two of headroom.

    This assertion used to read ``assert (err < 1e-5 for err in errs_amp)``,
    a generator expression --- which is a truthy object whatever it would
    yield, so nothing was tested. The errors are in fact of order
    ``5e-4``, and would have failed the ``1e-5`` bound.
    """
    training = greedy_downsampling_training
    errs_amp, errs_phi = training.validate_downsampling(8, 8)

    assert np.max(errs_amp) < 2 * training.tol_amp
    assert np.max(errs_phi) < 2 * training.tol_phi
