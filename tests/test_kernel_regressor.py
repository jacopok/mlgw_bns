r"""The kernel-ridge regressor, and the options that keep the old behaviour.

The surrogate's accuracy under a fixed training budget is limited by the
map from parameters to principal-component coefficients, not by the
basis: the truncation floor sits some four orders of magnitude below what
the network reaches. :class:`KernelRidgeNetwork` replaces that map.

The property these tests are really about is scale equivariance. Kernel
ridge solves :math:`(K + \alpha I)^{-1} y` separately per output, so
rescaling one output rescales its prediction by the same factor and
nothing else changes. That is what makes the regressor immune to
:attr:`Hyperparameters.pc_exponent`, and it is why component
coefficients spanning ten orders of magnitude are all reconstructed to
the same *relative* accuracy rather than the largest one swamping the
rest.
"""

import numpy as np
import pytest

from mlgw_bns.dataset_generation import Dataset
from mlgw_bns.mode_model import NN_KINDS, ModeModel
from mlgw_bns.neural_network import (
    Hyperparameters,
    KernelRidgeNetwork,
    SklearnNetwork,
)


@pytest.fixture(name="smooth_training_data")
def fixture_smooth_training_data():
    """Five inputs to three outputs whose scales differ by 15 decades."""
    rng = np.random.default_rng(seed=1)

    def targets(x):
        return np.stack(
            [
                np.sin(2.0 * x[:, 0]) * 1e3,
                np.cos(x[:, 1]) * 1e-7,
                (x[:, 2] ** 2) * 1e-12,
            ],
            axis=1,
        )

    x_train = rng.uniform(-1.0, 1.0, (400, 5))
    x_test = rng.uniform(-1.0, 1.0, (64, 5))
    return x_train, targets(x_train), x_test, targets(x_test)


def test_kernel_reconstructs_a_smooth_map(smooth_training_data):
    x_train, y_train, x_test, y_test = smooth_training_data

    network = KernelRidgeNetwork(Hyperparameters.default(len(x_train)))
    network.fit(x_train, y_train)
    prediction = network.predict(x_test)

    assert prediction.shape == y_test.shape

    relative = np.abs(prediction - y_test).max(axis=0) / np.abs(y_test).max(axis=0)
    assert np.all(relative < 0.05)


def test_kernel_is_equivariant_under_output_rescaling(smooth_training_data):
    """The whole point: an output's scale cannot change its relative error.

    This is what :attr:`Hyperparameters.pc_exponent` was trying, and
    failing, to control for the network, whose unweighted mean squared
    error weights each component by the inverse square of the scale it
    was divided by.
    """
    x_train, y_train, x_test, _ = smooth_training_data

    plain = KernelRidgeNetwork(Hyperparameters.default(len(x_train)))
    plain.fit(x_train, y_train)

    scales = np.array([1e-6, 1e4, 1e9])
    rescaled = KernelRidgeNetwork(Hyperparameters.default(len(x_train)))
    rescaled.fit(x_train, y_train * scales[np.newaxis, :])

    np.testing.assert_allclose(
        rescaled.predict(x_test) / scales[np.newaxis, :],
        plain.predict(x_test),
        rtol=1e-9,
    )


def test_kernel_hyperparameters_are_read(smooth_training_data):
    x_train, y_train, _, _ = smooth_training_data

    hyper = Hyperparameters.default(len(x_train))
    hyper.kernel_gamma = 0.37
    hyper.kernel_alpha = 1e-5

    network = KernelRidgeNetwork(hyper)
    assert network.regressor.gamma == 0.37
    assert network.regressor.alpha == 1e-5


def test_kernel_round_trips_through_a_file(smooth_training_data, tmp_path):
    x_train, y_train, x_test, _ = smooth_training_data

    network = KernelRidgeNetwork(Hyperparameters.default(len(x_train)))
    network.fit(x_train, y_train)

    path = str(tmp_path / "kernel.pkl")
    network.save(path)

    np.testing.assert_allclose(
        KernelRidgeNetwork.from_file(path).predict(x_test),
        network.predict(x_test),
        rtol=0.0,
        atol=0.0,
    )


def batch_size_used_during_fit(hyper, x_train, y_train, monkeypatch):
    """The mini-batch size scikit-learn actually saw.

    :meth:`SklearnNetwork.fit` restores the configured value once the fit
    is done, so the clipped one is only observable while the underlying
    regressor is running.
    """
    from sklearn.neural_network import MLPRegressor

    seen = {}
    original = MLPRegressor.fit

    def recording_fit(self, x, y):
        seen["batch_size"] = self.batch_size
        return original(self, x, y)

    monkeypatch.setattr(MLPRegressor, "fit", recording_fit)
    SklearnNetwork(hyper).fit(x_train, y_train)
    return seen["batch_size"]


def test_batch_size_clip_uses_the_sample_count(smooth_training_data, monkeypatch):
    """The configured mini-batch survives, rather than collapsing to five.

    Every model packaged before this was fixed was trained with a batch
    size of five, whatever was configured, because the clip used
    ``x_data.shape[1]`` --- the number of features.
    """
    x_train, y_train, _, _ = smooth_training_data

    hyper = Hyperparameters.default(len(x_train))
    hyper.max_iter = 1

    assert batch_size_used_during_fit(hyper, x_train, y_train, monkeypatch) == min(
        hyper.nn_params["batch_size"], len(x_train)
    )


def test_legacy_batch_size_clip_reproduces_the_packaged_behaviour(
    smooth_training_data, monkeypatch
):
    x_train, y_train, _, _ = smooth_training_data

    hyper = Hyperparameters.default(len(x_train))
    hyper.max_iter = 1
    hyper.legacy_batch_size_clip = True

    assert (
        batch_size_used_during_fit(hyper, x_train, y_train, monkeypatch)
        == x_train.shape[1]
    )


def test_hyperparameters_from_before_the_flag_get_the_repaired_clip(
    smooth_training_data, monkeypatch
):
    """Old checkpoints re-fit with the repaired mini-batch, not the legacy one.

    ``legacy_batch_size_clip`` is a dataclass field with a default, so it
    is a *class* attribute: an instance unpickled from before the field
    existed has nothing in its ``__dict__`` but still resolves the
    attribute, to the new default. Deleting the instance attribute here
    reproduces exactly that state.

    This costs the packaged models nothing --- the flag is read only when
    fitting, and loading one of them to predict never fits.
    """
    x_train, y_train, _, _ = smooth_training_data

    hyper = Hyperparameters.default(len(x_train))
    hyper.max_iter = 1
    del hyper.legacy_batch_size_clip
    assert "legacy_batch_size_clip" not in vars(hyper)

    assert batch_size_used_during_fit(hyper, x_train, y_train, monkeypatch) == min(
        hyper.nn_params["batch_size"], len(x_train)
    )


def test_mode_model_records_its_regressor_in_the_metadata():
    for name, kind in NN_KINDS.items():
        assert ModeModel(nn_kind=kind).metadata_dict["nn_kind"] == name


def test_mode_model_reads_its_regressor_back():
    model = ModeModel()
    assert model.nn_kind is SklearnNetwork

    model.set_metadata({"nn_kind": "KernelRidgeNetwork"})
    assert model.nn_kind is KernelRidgeNetwork


def test_metadata_without_a_regressor_keeps_the_network():
    """Models saved before the key existed still load, as networks."""
    model = ModeModel(nn_kind=SklearnNetwork)
    model.set_metadata({"srate_hz": 2048.0})
    assert model.nn_kind is SklearnNetwork


def test_reference_amplitude_is_off_by_default():
    assert Dataset(20.0, 4096.0).amplitude_reference_parameters is None


def test_reference_amplitude_sits_at_the_centre_of_the_ranges():
    dataset = Dataset(20.0, 4096.0, reference_amplitude=True)
    reference = dataset.amplitude_reference_parameters

    ranges = dataset.parameter_ranges
    assert reference.mass_ratio == pytest.approx(np.mean(ranges.q_range))
    assert reference.lambda_1 == pytest.approx(np.mean(ranges.lambda1_range))
    assert reference.chi_1 == 0.0
    assert reference.chi_2 == 0.0


def test_mode_model_propagates_the_reference_amplitude_to_its_dataset():
    assert ModeModel(reference_amplitude=True).dataset.reference_amplitude
    assert not ModeModel().dataset.reference_amplitude


def test_reference_amplitude_survives_a_metadata_round_trip():
    model = ModeModel(reference_amplitude=True)
    assert model.metadata_dict["reference_amplitude"]

    reloaded = ModeModel()
    reloaded.set_metadata(model.metadata_dict)
    assert reloaded.reference_amplitude
    # `load` rebuilds the dataset after applying the metadata, which is
    # what makes the flag take effect on the reconstruction path.
    reloaded.dataset = reloaded._make_dataset()
    assert reloaded.dataset.amplitude_reference_parameters is not None


@pytest.mark.parametrize("reference_amplitude", [False, True])
def test_recomposition_inverts_generation(reference_amplitude):
    """Whatever divides the EOB amplitude must multiply it back exactly.

    This is the invariant the reference amplitude could most easily
    break: `generate_residuals` and `recompose_residuals` pick the
    divisor independently, and if they ever disagree the error is a
    smooth multiplicative function of frequency, which a mismatch
    minimised over time and phase would partly hide.
    """
    from EOBRun_module import EOBRunPy  # type: ignore

    from mlgw_bns.data_management import ParameterRanges
    from mlgw_bns.dataset_generation import TEOBResumSGenerator

    dataset = Dataset(
        initial_frequency_hz=20.0,
        srate_hz=4096.0,
        waveform_generator=TEOBResumSGenerator(EOBRunPy),
        parameter_ranges=ParameterRanges(mass_range=(2.8, 2.8)),
        reference_amplitude=reference_amplitude,
    )

    _, parameters, residuals = dataset.generate_residuals(size=2, n_jobs=1)
    recomposed = dataset.recompose_residuals(residuals, parameters)

    true, _ = dataset.generate_waveforms_from_params(parameters, n_jobs=1)

    np.testing.assert_allclose(
        recomposed.amplitudes, true.amplitudes, rtol=1e-5, atol=0.0
    )
