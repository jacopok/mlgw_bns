import numpy as np
import pytest
from EOBRun_module import EOBRunPy

from mlgw_bns.dataset_generation import WaveformParameters
from mlgw_bns.higher_order_modes import (
    BarePostNewtonianModeGenerator,
    Mode,
    ModeGenerator,
    TEOBResumSModeGenerator,
    spherical_harmonic_spin_2,
)
from mlgw_bns.model import ModesModel


def test_mode_initialization():
    m = Mode(2, 1)
    m2 = Mode(2, 1)
    m3 = Mode(2, 2)

    assert m == m2
    assert m != m3

    assert m.l == 2
    assert m.m == 1


def test_mode_checking():

    with pytest.raises(TypeError):
        gen = BarePostNewtonianModeGenerator()

    with pytest.raises(NotImplementedError):
        gen = BarePostNewtonianModeGenerator(Mode(-2, 0))

    with pytest.raises(NotImplementedError):
        gen = TEOBResumSModeGenerator(eobrun_callable=EOBRunPy, mode=Mode(1, 0))


def test_teob_mode_generation(parameters):
    gen = TEOBResumSModeGenerator(eobrun_callable=EOBRunPy, mode=Mode(2, 2))

    gen.effective_one_body_waveform(parameters)


def test_spherical_harmonics():
    """Test against known cases, eqs II.9--13 in https://arxiv.org/pdf/0709.0093.pdf"""

    inclination = np.linspace(0, np.pi, num=50)[:, np.newaxis]
    azimuth = np.linspace(0, 2 * np.pi, num=50)[np.newaxis, :]

    theoretical_22 = (
        np.sqrt(5 / 64 / np.pi) * (1 + np.cos(inclination)) ** 2 * np.exp(2j * azimuth)
    )

    theoretical_21 = (
        np.sqrt(5 / 16 / np.pi)
        * np.sin(inclination)
        * (1 + np.cos(inclination))
        * np.exp(1j * azimuth)
    )

    theoretical_20 = np.sqrt(15 / 32 / np.pi) * np.sin(inclination) ** 2

    theoretical_2_negative1 = (
        np.sqrt(5 / 16 / np.pi)
        * np.sin(inclination)
        * (1 - np.cos(inclination))
        * np.exp(-1j * azimuth)
    )

    theoretical_2_negative2 = (
        np.sqrt(5 / 64 / np.pi) * (1 - np.cos(inclination)) ** 2 * np.exp(-2j * azimuth)
    )

    theoretical_harmonics = {
        Mode(2, 2): theoretical_22,
        Mode(2, 1): theoretical_21,
        Mode(2, 0): theoretical_20,
        Mode(2, -1): theoretical_2_negative1,
        Mode(2, -2): theoretical_2_negative2,
    }

    for mode, theoretical_harm in theoretical_harmonics.items():
        computed_harm = spherical_harmonic_spin_2(mode, inclination, azimuth)

        assert np.allclose(computed_harm, theoretical_harm)


def test_modes_model_waveform_generators():
    model = ModesModel(
        modes=[Mode(2, 2), Mode(2, 1)],
    )

    assert set(model.models) == set([Mode(2, 2), Mode(2, 1)])

    assert model.models[Mode(2, 2)].waveform_generator.mode == Mode(2, 2)
    assert model.models[Mode(2, 1)].waveform_generator.mode == Mode(2, 1)

    assert isinstance(
        model.models[Mode(2, 2)].waveform_generator, TEOBResumSModeGenerator
    )


def test_modes_model_prediction(
    default_model, trained_modes_model, params_with_extrinsic, frequencies
):

    hp_modes, hc_modes = trained_modes_model.predict(frequencies, params_with_extrinsic)
    hp_default, hc_default = default_model.predict(frequencies, params_with_extrinsic)

    assert np.allclose(abs(hp_modes), abs(hp_default), atol=0.0, rtol=1)
    assert np.allclose(abs(hc_modes), abs(hc_default), atol=0.0, rtol=1)

    assert np.allclose(np.unwrap(np.angle(hp_modes)), np.unwrap(np.angle(hp_default)))
