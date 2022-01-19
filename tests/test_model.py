import numpy as np
import pytest
from EOBRun_module import EOBRunPy  # type: ignore

from mlgw_bns.dataset_generation import TEOBResumSGenerator
from mlgw_bns.model import ExtendedWaveformParameters, Model
from mlgw_bns.model_validation import ValidateModel


def test_validating_model(generated_model):

    vm = ValidateModel(generated_model)


def test_model_saving(generated_model):

    generated_model.save()

    with generated_model.file_arrays as file:
        assert "downsampling/amplitude_indices" in file
        assert file["downsampling/amplitude_indices"][0] == 0
        assert 20_000 < file["downsampling/amplitude_indices"][10] < 30_000

        assert "principal_component_analysis/eigenvalues" in file


def test_quick_model_with_validation_mismatches(trained_model):

    vm = ValidateModel(trained_model)

    mismatches = vm.validation_mismatches(16)

    assert all(m < 1e-2 for m in mismatches)


def test_default_model_with_validation_mismatches(default_model):

    vm = ValidateModel(default_model)

    mismatches = vm.validation_mismatches(16)

    assert all(m < 3e-5 for m in mismatches)


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


@pytest.mark.xfail
@pytest.mark.benchmark(group="model-prediction")
@pytest.mark.parametrize(
    # "number_of_sample_points",
    # [16384]
    "number_of_sample_points",
    [128, 256, 512, 1024, 2048, 4096, 8192, 16384],
)
@pytest.mark.parametrize(
    "model_name, tolerance", [("trained_model", 1e-2), ("default_model", 1e-4)]
)
def test_model_nn_prediction(
    model_name, tolerance, request, benchmark, number_of_sample_points
):
    """Test whether the prediction for the plus and cross polarizations
    provided by the model matches the value given by
    """
    model = request.getfixturevalue(model_name)

    params = ExtendedWaveformParameters(
        mass_ratio=1.0,
        lambda_1=500.0,
        lambda_2=50.0,
        chi_1=0.1,
        chi_2=-0.1,
        dataset=model.dataset,
        distance_mpc=1.0,
        inclination=0.0,
        reference_phase=0.0,
        time_shift=0.0,
        total_mass=2.8,
    )

    teob_dict = params.teobresums
    teob_dict["use_geometric_units"] = 0
    teob_dict["initial_frequency"] /= params.mass_sum_seconds
    teob_dict["srate_interp"] /= params.mass_sum_seconds
    teob_dict["df"] /= params.mass_sum_seconds

    n_additional = 256

    # tweak initial frequency backward by a few samples
    # this is needed because of a bug in TEOBResumS
    # causing the phase evolution not to behave properly
    # at the beginning of integration
    # TODO remove this once the TEOB bug is fixed

    f_0 = teob_dict["initial_frequency"]
    delta_f = teob_dict["df"]
    new_f0 = f_0 - delta_f * n_additional
    teob_dict["initial_frequency"] = new_f0

    f_spa, rhp_teob, ihp_teob, rhc_teob, ihc_teob = EOBRunPy(teob_dict)

    hp_teob = (rhp_teob - 1j * ihp_teob)[n_additional:]
    hc_teob = (rhc_teob - 1j * ihc_teob)[n_additional:]
    f_spa = f_spa[n_additional:]

    n_downsample = len(f_spa) // number_of_sample_points

    freqs_hz = f_spa[::n_downsample]

    hp, hc = benchmark(model.predict, freqs_hz, params)

    vm = ValidateModel(model)

    hp_teob = hp_teob[::n_downsample]
    hc_teob = hc_teob[::n_downsample]

    # The model used here uses quite few training data, so we are not able to
    # achieve very small mismatches.
    # TODO is the mismatch found here too high?
    assert vm.mismatch(hp, hp_teob, frequencies=freqs_hz) < tolerance
    assert vm.mismatch(hc, hc_teob, frequencies=freqs_hz) < tolerance

    # the mismatch does not account for the magnitude of the waveforms,
    # so we check that separately.
    assert np.allclose(abs(hp), abs(hp_teob), atol=0.0, rtol=tolerance)
    assert np.allclose(abs(hc), abs(hc_teob), atol=0.0, rtol=tolerance)
