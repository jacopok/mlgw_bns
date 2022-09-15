import numpy as np
import pytest
from EOBRun_module import EOBRunPy  # type: ignore

from mlgw_bns.dataset_generation import ParameterSet, TEOBResumSGenerator
from mlgw_bns.model import Model, ParametersWithExtrinsic
from mlgw_bns.model_validation import ValidateModel

DEFAULT_MODEL_MAX_MISMATCH = 1e-5
TRAINED_MODEL_MAX_MISMATCH = 1.5e-2

# if the maximum mismatch is 1.5e-2,
# the average mismatch should be this many times
# smaller than the maximum mismatch;
# so, if this is 30 then the average
# mismatch will need to be smaller than 3e-3
AVERAGE_MISMATCH_REDUCTION_FACTOR = 10


def waveforms_are_close(
    h1, h2, validation_model, freqs_hz, tolerance_amp, tolerance_mismatch
):
    assert validation_model.mismatch(h1, h2, frequencies=freqs_hz) < tolerance_mismatch

    # the mismatch does not account for the magnitude of the waveforms,
    # so we check that separately.
    number_of_sample_points = len(h1)

    assert np.allclose(
        abs(h1)[: number_of_sample_points // 4],
        abs(h2)[: number_of_sample_points // 4],
        atol=0.0,
        rtol=tolerance_amp,
    )

    assert np.allclose(abs(h1), abs(h2), atol=0.0, rtol=tolerance_amp * 20)


def test_validating_model(generated_model):
    vm = ValidateModel(generated_model)


def test_model_saving(generated_model):

    generated_model.save()

    with generated_model.file_arrays as file:
        assert "downsampling/amplitude_indices" in file
        assert file["downsampling/amplitude_indices"][0] == 0
        assert "principal_component_analysis/eigenvalues" in file


def test_downsampling_indices_characteristics(generated_model):

    generated_model.save()

    with generated_model.file_arrays as file:
        assert 50 < file["downsampling/amplitude_indices"][10] < 10_000

        # this holds when training on residuals
        # assert 20_000 < file["downsampling/amplitude_indices"][10] < 30_000


def test_quick_model_with_validation_mismatches(trained_model):

    vm = ValidateModel(trained_model)

    mismatches = vm.validation_mismatches(16)

    assert all(m < TRAINED_MODEL_MAX_MISMATCH for m in mismatches)
    assert np.average(np.log(mismatches)) < np.log(
        TRAINED_MODEL_MAX_MISMATCH / AVERAGE_MISMATCH_REDUCTION_FACTOR
    )


@pytest.mark.requires_default
def test_default_model_with_validation_mismatches(default_model):

    vm = ValidateModel(default_model)

    mismatches = vm.validation_mismatches(16)

    assert all(m < DEFAULT_MODEL_MAX_MISMATCH for m in mismatches)
    assert np.average(np.log(mismatches)) < np.log(
        DEFAULT_MODEL_MAX_MISMATCH / AVERAGE_MISMATCH_REDUCTION_FACTOR
    )


@pytest.mark.requires_default
def test_default_model_residuals(default_model):

    vm = ValidateModel(default_model)

    params = vm.param_set(32)
    true_wfs = vm.true_waveforms(params)
    pred_wfs = vm.predicted_waveforms(params)

    amp_errors = np.log(true_wfs.amplitudes / pred_wfs.amplitudes)
    assert np.all(abs(amp_errors) < 6e-2)
    assert np.all(abs(amp_errors)[:, :10] < 1e-3)

    # TODO include phase errors, but subtracting the linear term
    # phase_errors = true_wfs.phases - pred_wfs.phases

    # for i, pe in enumerate(phase_errors):
    #     phase_errors[i] -= something

    # assert all(abs(phase_errors) < 1e-2)


@pytest.mark.benchmark(group="model-prediction")
@pytest.mark.parametrize(
    "number_of_sample_points",
    [128, 512, 2048, 8192],
)
@pytest.mark.parametrize(
    "model_name, tolerance_mismatch, tolerance_amp",
    [
        ("trained_model", TRAINED_MODEL_MAX_MISMATCH, 1e-1),
        pytest.param(
            "default_model",
            DEFAULT_MODEL_MAX_MISMATCH,
            2e-3,
            marks=pytest.mark.requires_default,
        ),
    ],
)
def test_model_nn_prediction(
    model_name,
    tolerance_mismatch,
    tolerance_amp,
    request,
    benchmark,
    number_of_sample_points,
):
    """Test whether the prediction for the plus and cross polarizations
    provided by the model matches the value given by
    """
    model = request.getfixturevalue(model_name)

    params = ParametersWithExtrinsic(
        mass_ratio=1.2,
        lambda_1=500.0,
        lambda_2=50.0,
        chi_1=0.1,
        chi_2=-0.1,
        distance_mpc=1.0,
        inclination=0.0,
        reference_phase=0.0,
        time_shift=0.0,
        total_mass=2.8,
    )

    teob_dict = params.teobresums_dict(model.dataset, use_effective_frequencies=False)
    teob_dict["use_geometric_units"] = "no"
    teob_dict["initial_frequency"] /= params.mass_sum_seconds
    teob_dict["srate_interp"] /= params.mass_sum_seconds
    teob_dict["df"] /= params.mass_sum_seconds

    # tweak initial frequency backward by a few samples
    # this is needed because of a bug in TEOBResumS
    # causing the phase evolution not to behave properly
    # at the beginning of integration
    # TODO remove this once the TEOB bug is fixed

    red_factor = 1024

    n_additional = 256 // red_factor
    f_0 = teob_dict["initial_frequency"]
    delta_f = teob_dict["df"] * red_factor
    new_f0 = f_0 - delta_f * n_additional
    teob_dict["initial_frequency"] = new_f0
    teob_dict["df"] = delta_f

    f_spa, rhp_teob, ihp_teob, rhc_teob, ihc_teob = EOBRunPy(teob_dict)

    hp_teob = (rhp_teob - 1j * ihp_teob)[n_additional:]
    hc_teob = (rhc_teob - 1j * ihc_teob)[n_additional:]
    f_spa = f_spa[n_additional:]

    n_downsample = round(len(f_spa) / number_of_sample_points)

    freqs_hz = f_spa[::n_downsample]

    assert (
        abs((len(freqs_hz) - number_of_sample_points) / number_of_sample_points) <= 0.1
    )

    hp, hc = benchmark(model.predict, freqs_hz, params)

    vm = ValidateModel(model)

    hp_teob = hp_teob[::n_downsample]
    hc_teob = hc_teob[::n_downsample]

    waveforms_are_close(hp, hp_teob, vm, freqs_hz, tolerance_amp, tolerance_mismatch)
    waveforms_are_close(hc, hc_teob, vm, freqs_hz, tolerance_amp, tolerance_mismatch)


@pytest.mark.parametrize("seed", list(range(5)))
@pytest.mark.parametrize(
    "model_name, tolerance_mismatch, tolerance_amp",
    [
        ("trained_model", TRAINED_MODEL_MAX_MISMATCH, 1e-1),
        pytest.param(
            "default_model",
            DEFAULT_MODEL_MAX_MISMATCH,
            2e-3,
            marks=pytest.mark.requires_default,
        ),
    ],
)
def test_model_nn_prediction_random_extrinsic(
    request, model_name, tolerance_mismatch, tolerance_amp, seed
):
    model = request.getfixturevalue(model_name)

    params = random_parameters(model, seed)

    teob_dict = params.teobresums_dict(model.dataset, use_effective_frequencies=False)

    teob_dict["use_geometric_units"] = "no"
    teob_dict["initial_frequency"] /= params.mass_sum_seconds
    teob_dict["srate_interp"] /= params.mass_sum_seconds
    teob_dict["df"] /= params.mass_sum_seconds / 1024.0

    # tweak initial frequency backward by a few samples
    # this is needed because of a bug in TEOBResumS
    # causing the phase evolution not to behave properly
    # at the beginning of integration
    # TODO remove this once the TEOB bug is fixed

    n_additional = 2
    f_0 = teob_dict["initial_frequency"]
    delta_f = teob_dict["df"]
    new_f0 = f_0 - delta_f * n_additional
    teob_dict["initial_frequency"] = new_f0

    freqs_hz, rhp_teob, ihp_teob, rhc_teob, ihc_teob = EOBRunPy(teob_dict)

    hp_teob = (rhp_teob - 1j * ihp_teob)[n_additional:]
    hc_teob = (rhc_teob - 1j * ihc_teob)[n_additional:]
    freqs_hz = freqs_hz[n_additional:]

    hp, hc = model.predict(freqs_hz, params)

    vm = ValidateModel(model)

    waveforms_are_close(hp, hp_teob, vm, freqs_hz, tolerance_amp, tolerance_mismatch)
    waveforms_are_close(hc, hc_teob, vm, freqs_hz, tolerance_amp, tolerance_mismatch)


def random_parameters(model: Model, seed: int) -> ParametersWithExtrinsic:
    param_generator = model.dataset.make_parameter_generator(seed)
    intrinsic_params = next(param_generator)

    return ParametersWithExtrinsic(
        mass_ratio=intrinsic_params.mass_ratio,
        lambda_1=intrinsic_params.lambda_1,
        lambda_2=intrinsic_params.lambda_2,
        chi_1=intrinsic_params.chi_1,
        chi_2=intrinsic_params.chi_2,
        distance_mpc=10 ** param_generator.rng.uniform(-1, 4),
        inclination=param_generator.rng.uniform(-np.pi, np.pi),
        total_mass=param_generator.rng.uniform(2.0, 4.0),
    )


@pytest.mark.parametrize(
    "param_name, value",
    [
        ("mass_ratio", 4),
        ("lambda_1", 0.0),
        ("lambda_2", 6000.0),
        ("chi_1", 2.0),
        ("chi_2", 2.0),
        ("total_mass", 5.0),
    ],
)
def test_parameters_out_of_bounds_error(trained_model, param_name, value):

    freqs = np.linspace(20.0, 2048.0)

    with pytest.raises(ValueError):
        params = ParametersWithExtrinsic(
            mass_ratio=2.0,
            lambda_1=500,
            lambda_2=400,
            chi_1=0.1,
            chi_2=-0.1,
            distance_mpc=10,
            inclination=0.5,
            total_mass=2.8,
        )
        setattr(params, param_name, value)
        trained_model.predict(freqs, params)


@pytest.mark.parametrize("total_mass", [2.0, 4.0])
def test_bounds_of_mass_range_work(trained_model, total_mass):
    freqs = np.linspace(20.0, 2048.0)
    params = ParametersWithExtrinsic(
        mass_ratio=2.0,
        lambda_1=500,
        lambda_2=400,
        chi_1=0.1,
        chi_2=-0.1,
        distance_mpc=10,
        inclination=0.5,
        total_mass=total_mass,
    )

    trained_model.predict(freqs, params)
