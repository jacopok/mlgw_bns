import numpy as np

from mlgw_bns.multibanding import (
    SEGLEN_20_HZ,
    high_frequency_grid,
    low_frequency_grid,
    reduced_frequency_array,
    seglen_from_freq,
)


def test_reduced_frequency_array_simplifies_to_high_or_low_frequency_grids():

    assert np.allclose(
        reduced_frequency_array(30.0, 40.0, 200.0), low_frequency_grid(30.0, 40.0)
    )
    assert np.allclose(
        reduced_frequency_array(30.0, 40.0, 10.0), high_frequency_grid(30.0, 40.0)
    )


def test_seglen_normalization():
    assert np.isclose(
        seglen_from_freq(
            20, maximum_mass_ratio=1.0, power_of_two=False, margin_percent=0.0
        ),
        SEGLEN_20_HZ,
    )
    assert np.isclose(
        seglen_from_freq(20, maximum_mass_ratio=1.0, margin_percent=0.0),
        2 ** (np.ceil(np.log2(SEGLEN_20_HZ))),
    )


def test_reduced_frequency_array_histogram_works():
    arr = reduced_frequency_array(30.0, 100.0, 40.0)

    hist, _ = np.histogram(arr, bins=np.arange(30, 101))

    assert np.allclose(
        hist[:10],
        seglen_from_freq(
            np.arange(30, 40) + 1 / 2, power_of_two=False, margin_percent=50.0
        ),
        rtol=5e-2,
    )
    assert np.allclose(hist[10:], seglen_from_freq(40.0), rtol=2e-2, atol=1.0)


def test_reduced_frequency_array_is_sorted():
    arr = reduced_frequency_array(30.0, 100.0, 40.0)
    assert np.allclose(arr, sorted(arr))
