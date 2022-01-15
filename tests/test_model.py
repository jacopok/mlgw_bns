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
        assert 25_000 < file["downsampling/amplitude_indices"][10] < 30_000

        assert "principal_component_analysis/eigenvalues" in file


@pytest.mark.benchmark(group="nn-prediction")
@pytest.mark.parametrize(
    "number_of_sample_points", [512, 1024, 2048, 4096, 8192, 16384]
)
def test_model_nn_prediction(trained_model, benchmark, number_of_sample_points):

    params = ExtendedWaveformParameters(
        mass_ratio=1.0,
        lambda_1=500,
        lambda_2=50,
        chi_1=0.1,
        chi_2=-0.1,
        dataset=trained_model.dataset,
        distance_mpc=1.0,
        inclination=0.0,
        reference_phase=0.0,
        time_shift=0.0,
        total_mass=2.8,
    )

    f_spa, hp_teob, hc_teob, _, _ = EOBRunPy(params.teobresums)

    n_downsample = len(f_spa) // number_of_sample_points

    freqs_hz = f_spa[::n_downsample] / params.mass_sum_seconds

    hp, hc = benchmark(trained_model.predict, freqs_hz, params)

    vm = ValidateModel(trained_model)

    hp_teob = hp_teob[::n_downsample]
    hc_teob = hc_teob[::n_downsample]

    # The model used here uses quite few training data, so we are not able to
    # achieve very small mismatches.
    assert vm.mismatch(hp, hp_teob, frequencies=freqs_hz) < 1e-2
    assert vm.mismatch(hc, hc_teob, frequencies=freqs_hz) < 1e-2

    # the mismatch does not account for the magnitude of the waveforms,
    # so we check that separately.
    assert np.isclose(np.sum(hp ** 2), np.sum(hp_teob ** 2))
    assert np.isclose(np.sum(hc ** 2), np.sum(hc_teob ** 2))
