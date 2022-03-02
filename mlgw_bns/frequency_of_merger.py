from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dataset_generation import WaveformParameters

MTSUN_SI = 4.925491025543576e-06


def frequency_of_merger(params: "WaveformParameters") -> float:
    """Given a set of waveform parameters, return an estimate for when
    these reach merger.

    Parameters
    ----------
    params : WaveformParameters
        Binary parameters. For the total mass,

    Returns
    -------
    Mf_merger: float
        Dimensionless frequency for the merger: :math:`Mf` at merger.
    """

    qqq = 1 - 4 * params.eta
    kkk = lambda_2_kappa(params.m_1, params.m_2, params.lambda_1, params.lambda_2)
    sss = (params.m_1 / params.dataset.total_mass) ** 2 * params.chi_1 + (
        params.m_2 / params.dataset.total_mass
    ) ** 2 * params.chi_2

    # get coefficients and params
    a0, a1, b0, b1, n1, n2, d1, d2, q1, q2, q3, q4 = (
        0.22754806,
        0.92330138,
        0.59374838,
        -1.99378496,
        0.03445340731627873,
        5.5799962023491245e-06,
        0.08404652974611324,
        0.00011328992320789428,
        13.828175998146255,
        517.4149218303298,
        12.74661916436829,
        139.76057108600236,
    )

    p1s = b0 * (1.0 + b1 * qqq)
    _n1 = n1 * (1.0 + q1 * qqq)
    _n2 = n2 * (1.0 + q2 * qqq)
    _d1 = d1 * (1.0 + q3 * qqq)
    _d2 = d2 * (1.0 + q4 * qqq)
    _up = 1.0 + _n1 * kkk + _n2 * kkk ** 2.0
    _lo = 1.0 + _d1 * kkk + _d2 * kkk ** 2.0
    return a0 * (1.0 + a1 * qqq) * (1.0 + p1s * sss) * _up / _lo * params.eta


def lambda_2_kappa(
    mass_1: float, mass_2: float, lambda_1: float, lambda_2: float
) -> float:
    """A parameter describing the tidal interactions of the binary.

    Equations 2-3 in Zappa et al 2018 10.1103/PhysRevLett.120.111101"""
    Mt = mass_1 + mass_2
    k1 = 3.0 * lambda_1 * (mass_2 / mass_1) * (mass_1 / Mt) ** 5
    k2 = 3.0 * lambda_2 * (mass_1 / mass_2) * (mass_2 / Mt) ** 5
    return k1 + k2
