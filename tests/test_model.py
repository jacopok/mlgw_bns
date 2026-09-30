import dataclasses

import numpy as np
import pytest

from mlgw_bns.data_management import ParameterRanges
from mlgw_bns.dataset_generation import ParameterSet
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model import (
    DEFAULT_MODES,
    MODELS_AVAILABLE,
    PRETRAINED_MODEL_FOLDER,
    Model,
)
from mlgw_bns.mode_model import ParametersWithExtrinsic
from mlgw_bns.model_validation import ValidateModel

# The packaged default model is trained with lambda up to 12000, but the
# PyPI-released TEOBResumS (unlike the locally patched one) crashes on
# root-bracketing for some high-q, high-Lambda combinations. These tests
# only need ground-truth EOB waveforms to validate against, so they draw
# from a narrower, crash-free slice of the model's own parameter space
# rather than its full training range.
REDUCED_TEOB_SAFE_RANGES = ParameterRanges(
    q_range=(1.0, 3.0), lambda1_range=(5.0, 5000.0), lambda2_range=(5.0, 5000.0)
)


def reduced_range_parameter_generator(model, seed):
    dataset = model.dataset
    return dataset.parameter_generator_class(
        parameter_ranges=REDUCED_TEOB_SAFE_RANGES, dataset=dataset, seed=seed
    )


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


def test_model_requires_the_22_mode():
    """The (2,2) sets the merger reference of every mode."""
    with pytest.raises(ValueError):
        Model(modes=[Mode(2, 1), Mode(3, 3)])


def test_model_mode_filename():
    mm = Model(modes=[Mode(2, 2), Mode(2, 1)], filename="some_base")

    assert mm.mode_filename(Mode(2, 2)) == "some_base_l2_m2"
    assert mm.mode_filename(Mode(2, 1)) == "some_base_l2_m1"


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


def test_default_for_testing_rejects_unknown_name():
    with pytest.raises(ValueError):
        Model.default_for_testing("not_a_model")


# Measured on the packaged (interim: public TEOBResumS, 8192 waveforms)
# model over the sixteen binaries this test draws (seed 7, inclination 1.0):
# median 6.8e-8, worst 4.0e-6. Nothing here is random (the model is loaded
# from disk and the parameters come from a fixed seed), so the factor of
# two or three is headroom for library changes rather than for the model.
DEFAULT_MODEL_MAX_MISMATCH = 1e-5
DEFAULT_MODEL_MEDIAN_MISMATCH = 2e-7

# Measured on the same model over the eight binaries
# `test_full_waveform_mismatch_is_flat_in_total_mass` draws (seed 7):
# medians of 1.5e-7, 2.3e-7 and 3.7e-7 at total_mass 2.2, 2.8 and 3.6,
# rising smoothly with mass rather than jumping at the 2.8 dataset reference.
FLAT_MASS_MEDIAN_MISMATCH = 8e-7

# The same sixteen binaries with nothing maximised --- the surrogate's own
# merger time and coalescence phase against TEOBResumS's, both from the
# tangent to the (2,2) phase at the top of the band. The median mismatch
# is 1.4e-4, but the tail follows the merger time, a derivative at the edge
# of the band that grows for small tidal deformabilities (see
# visualization/merger_reference_corner.py): over 2000 uniform draws |dt|
# has a median of 2 us, a 99th percentile of 50 us and a maximum of 0.35 ms,
# and 0.11 ms here (q=2.75, Lambda_1=31). Were the two references
# inconsistent it would be of the order of the waveform's duration.
UNOPTIMISED_MEDIAN_MISMATCH = 5e-4
MERGER_TIME_MAX_ERROR = 2e-4


def test_default_model_merger_reference_matches_teobresums(default_model):
    """The summed waveform against the EOB ground truth with nothing
    maximised: the merger time and coalescence phase are the model's."""

    mode_model = default_model.mode_models[Mode(2, 2)]
    validator = ValidateModel(mode_model)
    parameter_generator = reduced_range_parameter_generator(default_model, seed=7)
    intrinsics = [next(parameter_generator) for _ in range(16)]

    mismatches = []
    for intrinsic in intrinsics:
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
        predicted = sum(default_model.predict_modes_dict(validator.frequencies, params).values())
        true = sum(default_model.get_teob_modes_dict(validator.frequencies, params).values())
        mismatches.append(validator.unmaximised_mismatch(true, predicted))

    time_errors, _, _ = validator.merger_reference_errors(
        ParameterSet.from_list_of_waveform_parameters(intrinsics)
    )

    assert np.median(mismatches) < UNOPTIMISED_MEDIAN_MISMATCH
    assert np.max(np.abs(time_errors)) < MERGER_TIME_MAX_ERROR


@pytest.mark.parametrize("total_mass", [2.2, 2.8, 3.6])
def test_full_waveform_mismatch_is_flat_in_total_mass(default_model, total_mass):
    """The multi-mode mismatch must not jump when the total mass crosses
    the dataset reference (2.8): below it the rescaled grid dips under
    ``effective_initial_frequency_hz`` and the PN low-frequency extension
    fires, which used to overwrite each mode's inter-mode phase constant
    with the PN one and mis-phase the HOM modes by ~1e-4."""

    validator = ValidateModel(default_model.mode_models[Mode(2, 2)])
    frequencies = validator.frequencies
    band = (frequencies >= 20.0) & (frequencies <= 2048.0)
    parameter_generator = reduced_range_parameter_generator(default_model, seed=7)

    mismatches = []
    for _ in range(8):
        intrinsic = next(parameter_generator)
        params = ParametersWithExtrinsic(
            mass_ratio=intrinsic.mass_ratio,
            lambda_1=intrinsic.lambda_1,
            lambda_2=intrinsic.lambda_2,
            chi_1=intrinsic.chi_1,
            chi_2=intrinsic.chi_2,
            distance_mpc=100.0,
            inclination=1.0,
            total_mass=total_mass,
        )
        predicted = default_model.predict_modes_dict(frequencies, params)
        true = default_model.get_teob_modes_dict(frequencies, params)
        mismatches.append(
            validator.full_waveform_mismatch(
                {k: v[band] for k, v in true.items()},
                {k: v[band] for k, v in predicted.items()},
                frequencies=frequencies[band],
            )
        )

    assert np.median(mismatches) < FLAT_MASS_MEDIAN_MISMATCH


def test_coalescence_phase_and_merger_time(default_model):
    """``coalescence_phase`` rotates the ``(l, m)`` mode by
    ``exp(i m phi_c)`` --- not the same phase for every mode --- and
    ``merger_time`` shifts every mode by ``exp(-2 pi i f t_c)``."""

    frequencies = np.linspace(30.0, 1500.0, 400)
    phi_c, t_c = 0.37, 0.012
    base = ParametersWithExtrinsic(
        mass_ratio=1.6, lambda_1=500.0, lambda_2=500.0, chi_1=0.1, chi_2=0.05,
        distance_mpc=100.0, inclination=1.1, total_mass=2.8,
    )
    shifted = dataclasses.replace(base, coalescence_phase=phi_c, merger_time=t_c)

    modes_base = default_model.predict_modes_dict(frequencies, base)
    modes_shifted = default_model.predict_modes_dict(frequencies, shifted)

    for (l, m), array in modes_base.items():
        ratio = modes_shifted[(l, m)] / array
        np.testing.assert_allclose(
            ratio, np.exp(1j * (m * phi_c - 2 * np.pi * frequencies * t_c)), rtol=1e-6
        )


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
        frequencies, parameters_with_extrinsic
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
        frequencies, parameters_with_extrinsic
    )
    modes_dict = trained_model.predict_modes_dict(
        frequencies, parameters_with_extrinsic
    )

    assert set(modes_dict.keys()) == set(trained_model.modes)
    assert_waveforms_close(hp - 1j * hc, sum(modes_dict.values()))


def test_model_predict_modes_dict_sums_to_predict_at_other_total_mass(
    trained_model, parameters_with_extrinsic
):
    """At a total mass other than the reference one of the dataset, where
    the merger reference must be rescaled with the frequencies."""

    frequencies = np.linspace(30.0, 500.0, 50)
    total_mass = 4.0
    assert total_mass != trained_model.dataset.total_mass
    params = dataclasses.replace(
        parameters_with_extrinsic, total_mass=total_mass, inclination=1.0,
        merger_time=1e-3,
    )

    hp, hc = trained_model.predict(frequencies, params)
    modes_dict = trained_model.predict_modes_dict(frequencies, params)

    assert_waveforms_close(hp - 1j * hc, sum(modes_dict.values()))


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
        frequencies, parameters_with_extrinsic
    )
    hp_after, hc_after = reloaded.predict(
        frequencies, parameters_with_extrinsic
    )

    assert_waveforms_close(hp_before, hp_after)
    assert_waveforms_close(hc_before, hc_after)
