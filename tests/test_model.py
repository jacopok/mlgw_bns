import dataclasses

import numpy as np
import pytest

from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model import (
    DEFAULT_MODES,
    MODELS_AVAILABLE,
    PRETRAINED_MODEL_FOLDER,
    Model,
)
from mlgw_bns.mode_model import ParametersWithExtrinsic
from mlgw_bns.model_validation import ValidateModel


def assert_waveforms_close(first, second):
    """Compare two waveforms relative to their own scale.

    Strain amplitudes are of order 1e-21, so a plain `np.allclose`, whose
    default `atol` is 1e-8, would accept anything at all.
    """

    scale = np.max(np.abs(second))
    if scale == 0:
        assert np.all(first == 0)
        return
    assert np.allclose(first / scale, second / scale, atol=1e-8, rtol=0)


def test_model_requires_nonempty_modes():
    with pytest.raises(ValueError):
        Model(modes=[])


def test_model_mode_filename():
    mm = Model(modes=[Mode(2, 2), Mode(2, 1)], filename="some_base")

    assert mm.mode_filename(Mode(2, 2)) == "some_base_l2_m2"
    assert mm.mode_filename(Mode(2, 1)) == "some_base_l2_m1"


def test_model_filename_timeshifts():
    mm = Model(modes=[Mode(2, 2)], filename="some_base")

    assert mm.filename_timeshifts == "some_base_timeshifts.pkl"


def test_model_lazy_mode_models_dict():
    mm = Model(modes=[Mode(2, 2), Mode(2, 1)], filename="some_base")

    # nothing is materialized until accessed
    assert Mode(2, 2) not in mm.mode_models
    assert Mode(2, 1) not in mm.mode_models

    model_22 = mm.mode_models[Mode(2, 2)]

    assert Mode(2, 2) in mm.mode_models
    assert Mode(2, 1) not in mm.mode_models
    assert model_22.mode == Mode(2, 2)
    assert model_22.filename == "some_base_l2_m2"

    # repeated access returns the same cached instance
    assert mm.mode_models[Mode(2, 2)] is model_22


def test_model_mode_models_dict_rejects_excluded_mode():
    mm = Model(modes=[Mode(2, 2)], filename="some_base")

    with pytest.raises(KeyError):
        mm.mode_models[Mode(3, 3)]


def test_model_base_filename_setter_propagates_to_materialized_models():
    mm = Model(modes=[Mode(2, 2), Mode(2, 1)], filename="some_base")

    # only materialize (2, 2); (2, 1) is left lazy
    model_22 = mm.mode_models[Mode(2, 2)]

    mm.base_filename = "new_base"

    assert mm.base_filename == "new_base"
    assert model_22.filename == "new_base_l2_m2"
    # accessing the still-lazy mode afterwards should reflect the new base
    assert mm.mode_models[Mode(2, 1)].filename == "new_base_l2_m1"


def test_predict_amplitude_phase_mode_rejects_excluded_mode():
    mm = Model(modes=[Mode(2, 2)], filename="some_base")

    with pytest.raises(ValueError):
        mm.predict_amplitude_phase_mode(
            Mode(3, 3), np.array([50.0]), params=None
        )


def test_model_str_contains_modes_and_base_filename():
    mm = Model(modes=[Mode(2, 2), Mode(2, 1)], filename="some_base")

    s = str(mm)
    assert "(2,2)" in s
    assert "(2,1)" in s
    assert "base_filename=some_base" in s
    assert "n_modes=2" in s


def test_model_availability_flags_before_training():
    mm = Model(modes=[Mode(2, 2)], filename="some_base")

    assert not mm.auxiliary_data_available
    assert not mm.nn_available
    assert not mm.training_dataset_available


def test_default_for_testing_loads_every_mode(default_model):
    assert default_model.base_filename == (
        f"{PRETRAINED_MODEL_FOLDER}{MODELS_AVAILABLE[0]}"
    )
    assert default_model.modes == DEFAULT_MODES
    assert default_model.auxiliary_data_available
    assert default_model.nn_available
    assert default_model.time_shifts_predictor is not None


def test_default_for_testing_rejects_unknown_name():
    with pytest.raises(ValueError):
        Model.default_for_testing("not_a_model")


# The packaged model currently sits at a median full-waveform mismatch of
# ~1.5e-4 with a worst case of ~3e-3; these are those numbers with an order
# of magnitude of headroom, and should be tightened as the model improves.
DEFAULT_MODEL_MAX_MISMATCH = 1e-2
DEFAULT_MODEL_MEDIAN_MISMATCH = 1e-3


def test_default_model_full_waveform_mismatch(default_model):
    """Compare the summed multi-mode waveform against the EOB ground truth."""

    validator = ValidateModel(default_model.mode_models[Mode(2, 2)])
    frequencies = validator.frequencies
    parameter_generator = default_model.dataset.make_parameter_generator(seed=7)

    mismatches = []
    for _ in range(16):
        intrinsic = next(parameter_generator)
        params = ParametersWithExtrinsic(
            mass_ratio=intrinsic.mass_ratio,
            lambda_1=intrinsic.lambda_1,
            lambda_2=intrinsic.lambda_2,
            chi_1=intrinsic.chi_1,
            chi_2=intrinsic.chi_2,
            distance_mpc=100.0,
            inclination=1.0,
            total_mass=2.8,
        )
        # `time_shifts` is left out on purpose: the packaged model ships
        # its own predictor, and this is the way it is meant to be used.
        predicted = default_model.predict_modes_dict(frequencies, params)
        true = default_model.get_teob_modes_dict(frequencies, params)

        # The EOB modes are zero below the frequency at which the waveform
        # starts; only compare where all of them are defined.
        support = np.ones(len(frequencies), dtype=bool)
        for mode_array in true.values():
            support &= np.abs(mode_array) > 0
        assert support.sum() > 2

        mismatches.append(
            validator.full_waveform_mismatch(
                {k: v[support] for k, v in true.items()},
                {k: v[support] for k, v in predicted.items()},
                frequencies=frequencies[support],
            )
        )

    mismatches = np.array(mismatches)
    assert np.max(mismatches) < DEFAULT_MODEL_MAX_MISMATCH
    assert np.median(mismatches) < DEFAULT_MODEL_MEDIAN_MISMATCH


def test_model_generate_sets_availability_flags(generated_model):
    assert generated_model.auxiliary_data_available
    assert generated_model.training_dataset_available
    for mode in generated_model.modes:
        model = generated_model.mode_models[mode]
        assert model.auxiliary_data_available
        assert model.training_dataset_available


def test_model_set_hyper_and_train_nn(trained_model):
    assert trained_model.nn_available
    for mode in trained_model.modes:
        assert trained_model.mode_models[mode].nn is not None


def test_model_predict_returns_finite_waveform(
    trained_model, parameters_with_extrinsic
):
    frequencies = np.linspace(30.0, 500.0, 50)

    hp, hc = trained_model.predict(
        frequencies, parameters_with_extrinsic, time_shifts=0.0
    )

    assert hp.shape == frequencies.shape
    assert hc.shape == frequencies.shape
    assert np.all(np.isfinite(hp))
    assert np.all(np.isfinite(hc))


def test_model_predict_modes_dict_sums_to_predict(
    trained_model, parameters_with_extrinsic
):
    frequencies = np.linspace(30.0, 500.0, 50)

    hp, hc = trained_model.predict(
        frequencies, parameters_with_extrinsic, time_shifts=0.0
    )
    modes_dict = trained_model.predict_modes_dict(
        frequencies, parameters_with_extrinsic, time_shifts=0.0
    )

    assert set(modes_dict.keys()) == set(trained_model.modes)
    assert_waveforms_close(hp - 1j * hc, sum(modes_dict.values()))


def test_model_predict_modes_dict_sums_to_predict_at_other_total_mass(
    trained_model, parameters_with_extrinsic
):
    """The time shifts must be rescaled to the requested total mass.

    Both code paths agree trivially when the total mass is the reference
    one of the dataset, so check a mass which is not.
    """

    frequencies = np.linspace(30.0, 500.0, 50)
    total_mass = 4.0
    assert total_mass != trained_model.dataset.total_mass
    params = dataclasses.replace(
        parameters_with_extrinsic, total_mass=total_mass, inclination=1.0
    )

    hp, hc = trained_model.predict(frequencies, params, time_shifts=1e-3)
    modes_dict = trained_model.predict_modes_dict(
        frequencies, params, time_shifts=1e-3
    )

    assert_waveforms_close(hp - 1j * hc, sum(modes_dict.values()))


def test_model_predict_defaults_to_predicted_time_shifts(
    trained_model, parameters_with_extrinsic
):
    """Omitting `time_shifts` must be the same as querying the predictor."""

    frequencies = np.linspace(30.0, 500.0, 50)
    # at zero inclination every mode but the (2,2) vanishes
    params = dataclasses.replace(parameters_with_extrinsic, inclination=1.0)

    time_shifts = trained_model.time_shifts_predictor.predict(
        [params.intrinsic(trained_model.dataset).array]
    )[0]

    hp_explicit, hc_explicit = trained_model.predict(
        frequencies, params, time_shifts=time_shifts
    )
    hp_implicit, hc_implicit = trained_model.predict(frequencies, params)

    assert_waveforms_close(hp_explicit, hp_implicit)
    assert_waveforms_close(hc_explicit, hc_implicit)

    modes_explicit = trained_model.predict_modes_dict(
        frequencies, params, time_shifts=time_shifts
    )
    modes_implicit = trained_model.predict_modes_dict(frequencies, params)
    for mode in modes_explicit:
        assert_waveforms_close(modes_explicit[mode], modes_implicit[mode])


def test_model_predict_without_predictor_requires_time_shifts(
    trained_model, parameters_with_extrinsic
):
    """A model with no predictor must ask for `time_shifts`, not guess."""

    frequencies = np.linspace(30.0, 500.0, 50)
    predictor = trained_model.time_shifts_predictor
    trained_model.time_shifts_predictor = None

    try:
        with pytest.raises(ValueError):
            trained_model.predict(frequencies, parameters_with_extrinsic)

        # ... but providing them explicitly still works.
        hp, hc = trained_model.predict(
            frequencies, parameters_with_extrinsic, time_shifts=0.0
        )
        assert np.all(np.isfinite(hp))
        assert np.all(np.isfinite(hc))
    finally:
        trained_model.time_shifts_predictor = predictor


def test_model_predict_amplitude_phase_mode(
    trained_model, parameters_with_extrinsic
):
    frequencies = np.linspace(30.0, 500.0, 50)
    mode = trained_model.modes[0]

    amp, phase = trained_model.predict_amplitude_phase_mode(
        mode, frequencies, parameters_with_extrinsic
    )

    assert amp.shape == frequencies.shape
    assert phase.shape == frequencies.shape
    assert np.all(np.isfinite(amp))
    assert np.all(np.isfinite(phase))
    assert np.all(amp >= 0)


def test_model_save_and_load_roundtrip(
    trained_model, parameters_with_extrinsic
):
    trained_model.save()

    reloaded = Model(
        modes=trained_model.modes,
        filename=trained_model.base_filename,
        pca_components_number=10,
    )
    reloaded.load()

    assert reloaded.auxiliary_data_available
    assert reloaded.nn_available

    frequencies = np.linspace(30.0, 500.0, 50)
    hp_before, hc_before = trained_model.predict(
        frequencies, parameters_with_extrinsic, time_shifts=0.0
    )
    hp_after, hc_after = reloaded.predict(
        frequencies, parameters_with_extrinsic, time_shifts=0.0
    )

    assert_waveforms_close(hp_before, hp_after)
    assert_waveforms_close(hc_before, hc_after)
