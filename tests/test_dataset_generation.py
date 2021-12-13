import pytest

from mlgw_bns.dataset_generation import TEOBResumSGenerator, SlowWaveformGenerator


def test_teobresums_generator_creation():
    tg = TEOBResumSGenerator()
    assert isinstance(tg, TEOBResumSGenerator)
    assert isinstance(tg, SlowWaveformGenerator)
