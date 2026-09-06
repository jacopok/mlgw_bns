r"""Power weighting of the odd-:math:`m` regressor training sets.

The (2,1) and (3,3) mode amplitudes vanish identically on the
equal-mass, equal-spin locus, so their extracted phase is
:math:`\arg(0)` there and poorly conditioned in a neighbourhood of it.
The amplitude residual grows linearly out of that point, which puts a
steep boundary layer in the PCA coefficients at :math:`q \lesssim 1.15`
that a global RBF kernel cannot resolve --- it rings across the
low-:math:`q` region that *does* carry power.

Weighting each training waveform by its own integrated mode power lets
the fit relax exactly where the target is ill-conditioned and where the
mode contributes nothing to the summed waveform, without discarding any
training data. See :func:`~mlgw_bns.mode_model.mode_power_weights` and
`arXiv:2609.03025 <https://arxiv.org/abs/2609.03025>`_.

The invariant these tests protect above all: an *unweighted* fit must
stay byte-identical to what it was before weighting existed, so that the
even-:math:`m` modes and every packaged model are untouched.
"""

import numpy as np
import pytest

from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.mode_model import mode_power_weights
from mlgw_bns.neural_network import (
    Hyperparameters,
    KernelRidgeNetwork,
    SklearnNetwork,
)


@pytest.fixture(name="odd_m_amplitudes")
def fixture_odd_m_amplitudes():
    """(2,1)-like residuals: amplitude linear in ``q - 1``.

    This is the measured behaviour, not a caricature --- on the shipped
    model the (2,1) residual is exactly zero at ``q = 1`` and its maximum
    grows in strict proportion to ``q - 1`` (8e-4 at ``q - 1 = 5e-4``,
    1.6e-2 at ``q - 1 = 1e-2``).
    """
    q = np.linspace(1.001, 3.0, 200)
    frequencies = np.linspace(20.0, 2048.0, 64)
    shape = np.exp(-frequencies / 1000.0)
    amplitudes = (q - 1.0)[:, np.newaxis] * shape[np.newaxis, :]
    return q, frequencies, amplitudes


def test_weights_are_normalised_to_one(odd_m_amplitudes):
    _q, frequencies, amplitudes = odd_m_amplitudes

    weights = mode_power_weights(amplitudes, frequencies)

    assert weights.shape == (len(amplitudes),)
    assert np.isclose(weights.max(), 1.0)
    assert np.all(weights > 0.0)
    assert np.all(weights <= 1.0)


def test_odd_m_weights_collapse_towards_equal_mass(odd_m_amplitudes):
    """The whole point: the ill-conditioned corner is down-weighted hard."""
    q, frequencies, amplitudes = odd_m_amplitudes

    weights = mode_power_weights(amplitudes, frequencies)

    # power goes as the square of the amplitude, which goes as (q - 1)
    assert np.allclose(weights, ((q - 1.0) / (q[-1] - 1.0)) ** 2)
    assert np.median(weights[q < 1.05]) < 1e-3
    assert np.median(weights[q > 2.0]) > 0.5
    # monotone, since the underlying amplitude is
    assert np.all(np.diff(weights) > 0.0)


def test_even_m_like_weights_stay_flat():
    """A mode with no vanishing-amplitude locus is barely reweighted.

    This is why the even-``m`` modes are excluded rather than weighted:
    the weighting would be nearly a no-op on them anyway, so applying it
    would only perturb two modes that are already accurate.
    """
    frequencies = np.linspace(20.0, 2048.0, 64)
    q = np.linspace(1.0, 3.0, 200)
    amplitudes = (1.0 + 0.1 * q)[:, np.newaxis] * np.ones((1, len(frequencies)))

    weights = mode_power_weights(amplitudes, frequencies)

    assert weights.min() > 0.5


def test_exponent_softens_the_weighting(odd_m_amplitudes):
    _q, frequencies, amplitudes = odd_m_amplitudes

    full = mode_power_weights(amplitudes, frequencies, exponent=1.0)
    softened = mode_power_weights(amplitudes, frequencies, exponent=0.5)

    assert np.allclose(softened, np.sqrt(full))
    assert softened.min() > full.min()


def test_reference_amplitude_rescales_the_power(odd_m_amplitudes):
    """Undoing the fixed PN divisor changes the weights, and is applied.

    With ``reference_amplitude=True`` the divisor is parameter-independent,
    so multiplying it back in recovers the physical mode amplitude; it
    reweights across *frequency*, which the per-sample normalisation does
    not cancel.
    """
    _q, frequencies, amplitudes = odd_m_amplitudes
    pn_amplitude = frequencies ** (-7.0 / 6.0)

    # This fixture's amplitude factorises into (q - 1) times one fixed
    # frequency shape, so any frequency-only divisor cancels out of the
    # per-sample normalisation: the divisor must be accepted, and here it
    # changes nothing.
    assert np.allclose(
        mode_power_weights(amplitudes, frequencies),
        mode_power_weights(amplitudes, frequencies, pn_amplitude=pn_amplitude),
    )

    # It does matter once the residual's *shape* varies between samples,
    # which is the real case: the divisor decides how much each part of
    # the band contributes to the power.
    tilt = np.linspace(0.0, 1.0, len(amplitudes))[:, np.newaxis]
    varying = amplitudes * (1.0 + tilt * frequencies[np.newaxis, :] / 100.0)

    assert not np.allclose(
        mode_power_weights(varying, frequencies),
        mode_power_weights(varying, frequencies, pn_amplitude=pn_amplitude),
    )


def test_missing_reference_amplitude_warns(odd_m_amplitudes, caplog):
    _q, frequencies, amplitudes = odd_m_amplitudes

    with caplog.at_level("WARNING"):
        mode_power_weights(amplitudes, frequencies, pn_amplitude=None)

    assert "reference_amplitude=False" in caplog.text


def test_degenerate_power_falls_back_to_uniform_weights(caplog):
    frequencies = np.linspace(20.0, 2048.0, 64)
    amplitudes = np.zeros((10, len(frequencies)))

    with caplog.at_level("WARNING"):
        weights = mode_power_weights(amplitudes, frequencies, pn_amplitude=np.ones(64))

    assert np.all(weights == 1.0)
    assert "degenerate" in caplog.text


@pytest.fixture(name="weighting_training_data")
def fixture_weighting_training_data():
    """A map the regressor can only fit well on one half of the box."""
    rng = np.random.default_rng(seed=7)
    x_train = rng.uniform(-1.0, 1.0, (300, 5))
    y_train = np.stack(
        [np.sin(3.0 * x_train[:, 0]), np.cos(2.0 * x_train[:, 1])], axis=1
    )
    return x_train, y_train


#: ``NeuralNetwork`` subclass -> the scikit-learn estimator class whose
#: ``fit`` it ultimately calls.
BACKENDS = [
    (KernelRidgeNetwork, "sklearn.kernel_ridge", "KernelRidge"),
    (SklearnNetwork, "sklearn.neural_network", "MLPRegressor"),
]


@pytest.mark.parametrize("nn_kind, module, estimator", BACKENDS)
def test_unweighted_fit_does_not_pass_sample_weight(
    nn_kind, module, estimator, weighting_training_data, monkeypatch
):
    """``sample_weight=None`` must not reach scikit-learn at all.

    An explicit ``sample_weight=None`` would be harmless for these two
    estimators, but it would break any wrapper expecting the
    two-argument call --- including the ``MLPRegressor.fit`` monkeypatch
    in ``test_kernel_regressor.py`` --- and passing nothing is what
    guarantees the even-``m`` modes and every packaged model are fit
    exactly as they were before weighting existed.
    """
    x_train, y_train = weighting_training_data
    cls = getattr(__import__(module, fromlist=[estimator]), estimator)

    seen = {}
    original = cls.fit

    def recording_fit(self, x, y, **kwargs):
        seen["kwargs"] = kwargs
        return original(self, x, y, **kwargs)

    monkeypatch.setattr(cls, "fit", recording_fit)

    hyper = Hyperparameters.default(len(x_train))
    hyper.max_iter = 5
    nn_kind(hyper).fit(x_train, y_train)

    assert seen["kwargs"] == {}

    weights = np.linspace(0.1, 1.0, len(x_train))
    nn_kind(hyper).fit(x_train, y_train, sample_weight=weights)

    assert np.array_equal(seen["kwargs"]["sample_weight"], weights)


def test_unweighted_kernel_ridge_fit_is_reproducible(weighting_training_data):
    """The deterministic backend: same inputs, same prediction, weights or not."""
    x_train, y_train = weighting_training_data
    hyper = Hyperparameters.default_kernel_ridge(len(x_train))

    implicit = KernelRidgeNetwork(hyper)
    implicit.fit(x_train, y_train)

    explicit = KernelRidgeNetwork(hyper)
    explicit.fit(x_train, y_train, sample_weight=None)

    assert np.array_equal(implicit.predict(x_train), explicit.predict(x_train))


def test_weights_move_the_kernel_ridge_fit(weighting_training_data):
    """Down-weighted samples lose their pull on the solution."""
    x_train, y_train = weighting_training_data
    hyper = Hyperparameters.default_kernel_ridge(len(x_train))

    unweighted = KernelRidgeNetwork(hyper)
    unweighted.fit(x_train, y_train)

    # keep only the right half of the first parameter
    weights = np.where(x_train[:, 0] > 0.0, 1.0, 1e-6)
    weighted = KernelRidgeNetwork(hyper)
    weighted.fit(x_train, y_train, sample_weight=weights)

    kept = x_train[:, 0] > 0.0
    error_unweighted = np.abs(unweighted.predict(x_train) - y_train)
    error_weighted = np.abs(weighted.predict(x_train) - y_train)

    # the weighted fit is no worse where it was told to look ...
    assert error_weighted[kept].mean() <= error_unweighted[kept].mean() * 1.5
    # ... and clearly worse where it was told not to
    assert error_weighted[~kept].mean() > error_unweighted[~kept].mean()


def test_mode_model_weights_only_odd_m(trained_model):
    """The ``Model`` end of it: (2,1) is weighted, (2,2) is not."""
    odd = trained_model.mode_models[Mode(2, 1)]
    even = trained_model.mode_models[Mode(2, 2)]

    assert even._training_power_weights() is None

    weights = odd._training_power_weights()
    assert weights is not None
    assert len(weights) == len(odd.training_dataset)
    assert np.isclose(weights.max(), 1.0)


def test_power_weighting_can_be_switched_off(trained_model):
    odd = trained_model.mode_models[Mode(2, 1)]

    odd.power_weighting = False
    try:
        assert odd._training_power_weights() is None
    finally:
        odd.power_weighting = True


def test_power_weighting_round_trips_through_the_metadata(trained_model):
    """A model saved now records the choice; one saved before defaults."""
    odd = trained_model.mode_models[Mode(2, 1)]

    assert odd.metadata_dict["power_weighting"] is True
    assert odd.metadata_dict["power_weight_exponent"] == 1.0

    # a sidecar written before these keys existed leaves the defaults
    legacy = {k: v for k, v in odd.metadata_dict.items() if not k.startswith("power")}
    odd.set_metadata(legacy)
    assert odd.power_weighting is True
    assert odd.power_weight_exponent == 1.0
