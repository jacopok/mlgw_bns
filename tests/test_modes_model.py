import numpy as np
import pytest

from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model import ParametersWithExtrinsic
from mlgw_bns.modes_model import MODES_MODELS_AVAILABLE, ModesModel


def test_modes_model_requires_nonempty_modes():
    with pytest.raises(ValueError):
        ModesModel(modes=[])


def test_modes_model_mode_filename():
    mm = ModesModel(modes=[Mode(2, 2), Mode(2, 1)], filename="some_base")

    assert mm.mode_filename(Mode(2, 2)) == "some_base_l2_m2"
    assert mm.mode_filename(Mode(2, 1)) == "some_base_l2_m1"


def test_modes_model_filename_timeshifts():
    mm = ModesModel(modes=[Mode(2, 2)], filename="some_base")

    assert mm.filename_timeshifts == "some_base_timeshifts.pkl"


def test_modes_model_lazy_models_dict():
    mm = ModesModel(modes=[Mode(2, 2), Mode(2, 1)], filename="some_base")

    # nothing is materialized until accessed
    assert Mode(2, 2) not in mm.models
    assert Mode(2, 1) not in mm.models

    model_22 = mm.models[Mode(2, 2)]

    assert Mode(2, 2) in mm.models
    assert Mode(2, 1) not in mm.models
    assert model_22.mode == Mode(2, 2)
    assert model_22.filename == "some_base_l2_m2"

    # repeated access returns the same cached instance
    assert mm.models[Mode(2, 2)] is model_22


def test_modes_model_models_dict_rejects_excluded_mode():
    mm = ModesModel(modes=[Mode(2, 2)], filename="some_base")

    with pytest.raises(KeyError):
        mm.models[Mode(3, 3)]


def test_modes_model_base_filename_setter_propagates_to_materialized_models():
    mm = ModesModel(modes=[Mode(2, 2), Mode(2, 1)], filename="some_base")

    # only materialize (2, 2); (2, 1) is left lazy
    model_22 = mm.models[Mode(2, 2)]

    mm.base_filename = "new_base"

    assert mm.base_filename == "new_base"
    assert model_22.filename == "new_base_l2_m2"
    # accessing the still-lazy mode afterwards should reflect the new base
    assert mm.models[Mode(2, 1)].filename == "new_base_l2_m1"


def test_predict_amplitude_phase_mode_rejects_excluded_mode():
    mm = ModesModel(modes=[Mode(2, 2)], filename="some_base")

    with pytest.raises(ValueError):
        mm.predict_amplitude_phase_mode(
            Mode(3, 3), np.array([50.0]), params=None
        )


def test_modes_model_str_contains_modes_and_base_filename():
    mm = ModesModel(modes=[Mode(2, 2), Mode(2, 1)], filename="some_base")

    s = str(mm)
    assert "(2,2)" in s
    assert "(2,1)" in s
    assert "base_filename=some_base" in s
    assert "n_modes=2" in s


def test_modes_model_availability_flags_before_training():
    mm = ModesModel(modes=[Mode(2, 2)], filename="some_base")

    assert not mm.auxiliary_data_available
    assert not mm.nn_available
    assert not mm.training_dataset_available


@pytest.mark.requires_default
def test_modes_model_default_for_testing_smoke():
    """The packaged HOM checkpoints are not up to date / present, so this
    should degrade gracefully (with warnings) rather than raising."""
    model_name = MODES_MODELS_AVAILABLE[0]
    mm = ModesModel.default_for_testing(model_name)

    assert mm.base_filename == f"data/HOM/{model_name}"
    assert len(mm.modes) == 4


def test_modes_model_generate_sets_availability_flags(generated_modes_model):
    assert generated_modes_model.auxiliary_data_available
    assert generated_modes_model.training_dataset_available
    for mode in generated_modes_model.modes:
        model = generated_modes_model.models[mode]
        assert model.auxiliary_data_available
        assert model.training_dataset_available


def test_modes_model_set_hyper_and_train_nn(trained_modes_model):
    assert trained_modes_model.nn_available
    for mode in trained_modes_model.modes:
        assert trained_modes_model.models[mode].nn is not None


def test_modes_model_predict_returns_finite_waveform(
    trained_modes_model, parameters_with_extrinsic
):
    frequencies = np.linspace(30.0, 500.0, 50)

    h, hp, hc = trained_modes_model.predict(
        frequencies, parameters_with_extrinsic, time_shifts=0.0
    )

    assert h.shape == frequencies.shape
    assert np.all(np.isfinite(h))
    assert np.allclose(h, hp - 1j * hc)


def test_modes_model_predict_modes_dict_sums_to_predict(
    trained_modes_model, parameters_with_extrinsic
):
    frequencies = np.linspace(30.0, 500.0, 50)

    h, _, _ = trained_modes_model.predict(
        frequencies, parameters_with_extrinsic, time_shifts=0.0
    )
    modes_dict = trained_modes_model.predict_modes_dict(
        frequencies, parameters_with_extrinsic, time_shifts=0.0
    )

    assert set(modes_dict.keys()) == set(trained_modes_model.modes)
    assert np.allclose(h, sum(modes_dict.values()))


def test_modes_model_predict_amplitude_phase_mode(
    trained_modes_model, parameters_with_extrinsic
):
    frequencies = np.linspace(30.0, 500.0, 50)
    mode = trained_modes_model.modes[0]

    amp, phase = trained_modes_model.predict_amplitude_phase_mode(
        mode, frequencies, parameters_with_extrinsic
    )

    assert amp.shape == frequencies.shape
    assert phase.shape == frequencies.shape
    assert np.all(np.isfinite(amp))
    assert np.all(np.isfinite(phase))
    assert np.all(amp >= 0)


def test_modes_model_save_and_load_roundtrip(
    trained_modes_model, parameters_with_extrinsic
):
    trained_modes_model.save()

    reloaded = ModesModel(
        modes=trained_modes_model.modes,
        filename=trained_modes_model.base_filename,
        pca_components_number=10,
    )
    reloaded.load()

    assert reloaded.auxiliary_data_available
    assert reloaded.nn_available

    frequencies = np.linspace(30.0, 500.0, 50)
    h_before, _, _ = trained_modes_model.predict(
        frequencies, parameters_with_extrinsic, time_shifts=0.0
    )
    h_after, _, _ = reloaded.predict(
        frequencies, parameters_with_extrinsic, time_shifts=0.0
    )

    assert np.allclose(h_before, h_after)
