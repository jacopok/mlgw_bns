from typing import TYPE_CHECKING

import numpy as np

from .taylorf2 import compute_delta_lambda, compute_lambda_tilde

if TYPE_CHECKING:
    from .model import ParametersWithExtrinsic


def mlgw_bns_merger_time_shift(params: "ParametersWithExtrinsic") -> float:

    m1 = params.total_mass * params.mass_ratio / (1 + params.mass_ratio)
    m2 = params.total_mass / (1 + params.mass_ratio)
    nu = m1 * m2 / params.total_mass / params.total_mass
    X = 1 - 4 * nu
    sqX = np.sqrt(X)
    lt = compute_lambda_tilde(m1, m2, params.lambda_1, params.lambda_2)
    dl = compute_delta_lambda(m1, m2, params.lambda_1, params.lambda_2)
    ld = params.lambda_2 - params.lambda_1
    ls = params.lambda_1 + params.lambda_2
    ce = (m1 * params.chi_1 + m2 * params.chi_2) / params.total_mass

    cs = {
        "a0": 0.0020759621961942603,
        "a1": -0.003070536961079697,
        "a2": -9.629458339368572e-07,
        "a3": 0.000542447368295823,
        "a4": 6.604879000806726e-09,
        "b0": -6.585504061714901e-12,
        "b1": 5.652710843348325e-13,
        "bx": -2.293953484226671e-14,
        "c0": 12.882556914189799,
        "c1": 11.21937846139295,
        "c2": 11.449609847910668,
        "c3": -2.9681147636094796,
        "d0": -0.07143247198447567,
        "d1": -0.9359464525773223,
        "b2": 0.002444447726984135,
        "b3": -3.8982563658704605e-05,
        "b4": -0.00017640077699324582,
        "e0": 0.1812511561795588,
        "e1": -1.2949443130142897,
        "e2": -1.6835328420336901,
        "e3": -0.46264137662511656,
        "b5": -0.010454941422013362,
        "b6": 2.838006604314393,
        "b7": -0.00033645778450926226,
        "b8": -0.010507369168043812,
    }

    lt_fit = lt + cs["d0"] * sqX * ld + cs["d1"] * sqX * dl
    fx_q1_s0 = (
        cs["a0"]
        * (1.0 + cs["a1"] * lt_fit + cs["a2"] * lt_fit ** 2)
        / (1.0 + cs["a3"] * lt_fit + cs["a4"] * lt_fit ** 2)
    )
    X_corr = (1 + cs["c0"] * X + cs["c1"] * X * X) / (
        1 + cs["c2"] * X + cs["c3"] * X * X
    )
    S_corr = (1 + cs["e0"] * ce + cs["e1"] * ce ** 2 + cs["e2"] * X * ce) / (
        1 + cs["e3"] * ce
    ) ** 2

    correct1 = cs["b0"] * ld ** 2 + cs["b1"] * ls ** 2 + cs["bx"] * ls * ld
    correct2 = cs["b2"] * X * (1.0 + cs["b3"] * ld + cs["b4"] * ls)
    correct3 = cs["b5"] * ce * (1 + cs["b6"] * X + cs["b7"] * lt) + cs["b8"] * X * (
        params.chi_1 - params.chi_2
    )

    dt_m28 = fx_q1_s0 * X_corr * S_corr + correct1 + correct2 + correct3
    return dt_m28 * params.total_mass / 2.8
