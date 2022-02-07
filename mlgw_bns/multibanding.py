import numpy as np

EXTRA_FACTOR = 1.5

# this value refers to the eta=1/4, m_tot=2.8 case
SEGLEN_20_HZ = 157.86933774


def seglen_from_freq(
    f_0: float, m_tot: float = 2.8, maximum_mass_ratio: float = 2.0
) -> float:
    r"""
    The seglen has a closed-form expression in the Newtonian limit,
    see e.g. Maggiore (2007), eq. 4.21:

    :math:`t = 5/256 (\pi f)^{-8/3} (G M \eta ^{3/5} / c^3)^{-5/3}`

    for which we compute the highest possible value, with the smallest possible
    :math:`\eta`; since
    :math:`t \propto \eta^{-1}` this gives us a upper bound on the seglen.

    Parameters
    ----------
    f_0: float
            Initial frequency from which the waveform starts, in Hz.
    m_tot: float, optional
            Total mass of the binary,
            in solar masses.
            Defaults to 2.8.
    maximum_mass_ratio: float
            Maximum allowed mass ratio in the dataset;
            this is used to give an upper bound on the seglen.

    Returns
    -------
    seglen : float
            Approximate duration of the BNS waveform, in seconds.
    """

    eta = maximum_mass_ratio / (1 + maximum_mass_ratio) ** 2

    return SEGLEN_20_HZ * (f_0 / 20) ** (-8 / 3) * (m_tot / 2.8) ** (5 / 3) / (4 * eta)


def reduced_frequency_array(f_min: float, f_max: float, f_pivot: float) -> np.ndarray:
    r"""Compute an array of frequencies which are a good guess to represent
    a gravitational waveform starting at a given minimum frequency.

    Above ``f_pivot`` a uniform spacing is used; below it
    a non-uniform spacing is used, which follows the rule
    :math:`dN / df \sim T(f)`, where :math:`T(f)` is the seglen
    (time-domain length) of a waveform  which would start at
    that point, and which can be computed by

    Parameters
    ----------
    f_min : float
        Minimum frequency of the array, in Hz.
    f_max : float
        Maximum frequency of the array, in Hz.
    f_pivot : float
        Frequency at which to switch from the non-uniform
        to the uniform array, in Hz.

    Returns
    -------
    frequencies: np.ndrray
        Frequency array, in Hz.
    """

    if f_min <= f_pivot <= f_max:
        df_pivot = 1 / seglen_from_freq(f_pivot)
        return np.append(
            low_frequency_grid(f_min, f_pivot),
            high_frequency_grid(f_pivot + df_pivot, f_max),
        )
    elif f_max < f_pivot:
        return low_frequency_grid(f_min, f_max)
    else:
        return high_frequency_grid(f_min, f_max)


def high_frequency_grid(f_min: float, f_max: float):
    """Uniform grid for the high-frequency regime.

    Parameters
    ----------
    f_min : float
        Minimum frequency of the array, in Hz.
    f_max : float
        Maximum frequency of the array, in Hz.

    Returns
    -------
    frequencies: np.ndrray
        Frequency array, in Hz.
    """

    return np.arange(f_min, f_max, step=1 / seglen_from_freq(f_min) / EXTRA_FACTOR)


def low_frequency_grid(f_min: float, f_max: float):
    """Non-uniform grid for the low-frequency regime.

    This is achieved by generating points uniformly in
    :math:`f^{-8/3+1}`-space, and then transforming back.
    The normalization is computed by integrating
    :math:`dN/df = t(f)` in the frequency range of interest,
    where :math:`t(f) = t_0 (f/f_0)^{-8/3}` (as implemented
    in :func:`seglen_from_freq`): this yields
    :math:`N(f_1, f_2) = f_0^{8/3} t_0 (f_1^{-5/3} - f_2^{-5/3})`.

    Since we are already working in :math:`f^{-5/3}` space,
    to achieve this it is sufficient to use a step of
    :math:`1/(f_0^{8/3} t_0)`; this is easily computed by looking
    at :math:`t(1)` (since the function :math:`t(f)` is already implemented).

    Parameters
    ----------
    f_min : float
        Minimum frequency of the array, in Hz.
    f_max : float
        Maximum frequency of the array, in Hz.

    Returns
    -------
    frequencies: np.ndrray
        Frequency array, in Hz.
    """
    f_min_reduced, f_max_reduced = f_min ** (-5 / 3), f_max ** (-5 / 3)
    # iterate backwards since we are f^{-5/3} is a decreasing function of f
    grid = np.arange(
        f_max_reduced,
        f_min_reduced,
        step=1 / seglen_from_freq(1) * (5 / 3) / EXTRA_FACTOR,
    )
    return grid ** (-3 / 5)
