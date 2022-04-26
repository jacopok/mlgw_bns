import pytest
from EOBRun_module import EOBRunPy

from mlgw_bns.dataset_generation import WaveformParameters
from mlgw_bns.higher_order_modes import (
    BarePostNewtonianModeGenerator,
    Mode,
    ModeGenerator,
    TEOBResumSModeGenerator,
)


def test_mode_initialization():
    m = Mode(2, 1)
    m2 = Mode(2, 1)
    m3 = Mode(2, 2)

    assert m == m2
    assert m != m3

    assert m.l == 2
    assert m.m == 1


def test_mode_is_required():
    with pytest.raises(TypeError):
        gen = BarePostNewtonianModeGenerator()

    with pytest.raises(NotImplementedError):
        gen = BarePostNewtonianModeGenerator(Mode(-2, 0))


def test_teob_mode_generation(parameters):
    gen = TEOBResumSModeGenerator(eobrun_callable=EOBRunPy, mode=Mode(2, 2))

    gen.effective_one_body_waveform(parameters)
