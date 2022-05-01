r""" This module deals with the generation of waveforms coming from a single mode 
of a spherical harmonic decomposition of the GW emission from a binary system.

As described in `the TEOBResumS documentation <https://bitbucket.org/eob_ihes/teobresums/wiki/Conventions,%20parameters%20and%20output>`_,
the full waveform is expressed as 

:math:`h_+ - i h_\times = \sum_{\ell m} A_{\ell m} e^{-i \phi_{\ell m}} Y_{\ell m}(\iota, \varphi)`

where the pair :math:`\ell, m`, with :math:`\ell \geq m`, is known as a *mode*, and
it is implemented as the namedtuple :class:`Mode`.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Callable, NamedTuple, Optional

import numpy as np
from scipy.special import factorial  # type: ignore

from .dataset_generation import (
    WaveformGenerator,
    WaveformParameters,
    amplitude_3h_post_newtonian,
    phase_5h_post_newtonian_tidal,
)

WaveformCallable = Callable[[WaveformParameters, np.ndarray], np.ndarray]

# Mode = namedtuple("Mode", ["l", "m"])
class Mode(NamedTuple):
    """A mode in the harmonic decomposition of the GW emission from a system."""

    l: int
    m: int

    def opposite(self):
        return self.__class__(self.l, -self.m)


# all modes with l<5, m>0 are supported by TEOB
EOB_SUPPORTED_MODES = [Mode(l, m) for l in range(2, 5) for m in range(1, l + 1)]

# TODO fix these, but it's not so bad now -
# these are only wrong by a constant scaling

_post_newtonian_amplitudes_by_mode: dict[Mode, WaveformCallable] = {
    Mode(2, 2): amplitude_3h_post_newtonian
}
_post_newtonian_phases_by_mode: dict[Mode, WaveformCallable] = {
    Mode(2, 2): phase_5h_post_newtonian_tidal
}


class ModeGenerator(WaveformGenerator):
    """Generic generator of a single mode for a waveform."""

    supported_modes = list(_post_newtonian_amplitudes_by_mode.keys())

    def __init__(self, mode: Mode, *args, **kwargs):

        self._mode = None

        super().__init__(*args, **kwargs)  # type: ignore
        # see (https://github.com/python/mypy/issues/5887) for typing problem

        self.mode = mode

    @property
    def mode(self) -> Optional[Mode]:
        return self._mode

    @mode.setter
    def mode(self, val: Optional[Mode]) -> None:

        if val not in self.supported_modes and val is not None:
            raise NotImplementedError(
                f"{val} is not supported yet for {self.__class__}!"
            )

        self._mode = val


class BarePostNewtonianModeGenerator(ModeGenerator):
    def post_newtonian_amplitude(
        self, params: "WaveformParameters", frequencies: np.ndarray
    ) -> np.ndarray:

        assert self.mode is not None
        return _post_newtonian_amplitudes_by_mode[self.mode](params, frequencies)

    def post_newtonian_phase(
        self, params: "WaveformParameters", frequencies: np.ndarray
    ) -> np.ndarray:
        assert self.mode is not None
        return _post_newtonian_phases_by_mode[self.mode](params, frequencies)

    def effective_one_body_waveform(
        self, params: "WaveformParameters", frequencies: Optional[np.ndarray] = None
    ):
        raise NotImplementedError(
            "This generator does not include the possibility "
            "to generate effective one body waveforms"
        )


class TEOBResumSModeGenerator(BarePostNewtonianModeGenerator):
    supported_modes = EOB_SUPPORTED_MODES

    def __init__(self, eobrun_callable: Callable, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.eobrun_callable = eobrun_callable

    def effective_one_body_waveform(
        self, params: "WaveformParameters", frequencies: Optional[np.ndarray] = None
    ):
        assert self.mode is not None
        par_dict: dict = params.teobresums()

        # tweak initial frequency backward by a few samples
        # this is needed because of a bug in TEOBResumS
        # causing the phase evolution not to behave properly
        # at the beginning of integration
        # TODO remove this once the TEOB bug is fixed

        n_additional = 256
        f_0 = par_dict["initial_frequency"]
        delta_f = par_dict["df"]
        new_f0 = f_0 - delta_f * n_additional
        par_dict["initial_frequency"] = new_f0

        to_slice = (
            slice(-len(frequencies), None)
            if frequencies is not None
            else slice(n_additional, None)
        )

        if frequencies is not None:
            frequencies_list = list(
                np.insert(
                    frequencies,
                    0,
                    np.arange(f_0 - delta_f * n_additional, f_0, step=delta_f),
                )
            )
            par_dict.pop("df")
            par_dict["interp_freqs"] = "yes"
            par_dict["freqs"] = frequencies_list

        par_dict["arg_out"] = "yes"
        par_dict["use_mode_lm"] = [mode_to_k(self.mode)]

        f_spa, _, _, _, _, hflm, _, _ = self.eobrun_callable(par_dict)

        amplitude = hflm[str(mode_to_k(self.mode))][0][to_slice]
        phase = hflm[str(mode_to_k(self.mode))][1][to_slice]

        return (f_spa, amplitude, phase)


def spherical_harmonic_spin_2(
    mode: Mode, inclination: float, azimuth: float
) -> complex:
    r"""Returns the spin-2 spherical harmonic
    :math:`^{-2}Y_{\ell m}(\iota, \varphi) =
    (-1)^s \sqrt{\frac{2 \ell+1}{4 \pi}} d_{m, s}^{\ell} (\iota) e^{im \phi_0}`

    where :math:`s= -2`
    """

    return (
        np.sqrt((2 * mode.l + 1) / 4 / np.pi)
        * wigner_d_function_spin_2(mode, inclination)
        * np.exp(1j * mode.m * azimuth)
    )


def wigner_d_function_spin_2(mode: Mode, inclination: float) -> complex:
    """Equation II.8 in https://arxiv.org/pdf/0709.0093.pdf, with :math:`s=-2`."""

    return_value = 0

    cos_i_halves = np.cos(inclination / 2)
    sin_i_halves = np.sin(inclination / 2)

    ki = max(0, mode.m - 2)
    kf = min(mode.l + mode.m, mode.l - 2)

    for k in range(ki, kf + 1):
        norm = (
            factorial(k)
            * factorial(mode.l + mode.m - k)
            * factorial(mode.l - 2 - k)
            * factorial(k + 2 - mode.m)
        )
        return_value += (
            (-1) ** k
            * cos_i_halves ** (2 * mode.l + mode.m - 2 - 2 * k)
            * sin_i_halves ** (2 * k + 2 - mode.m)
        ) / norm

    const = np.sqrt(
        factorial(mode.l + mode.m)
        * factorial(mode.l - mode.m)
        * factorial(mode.l + 2)
        * factorial(mode.l - 2)
    )

    return const * return_value


def mode_to_k(mode: Mode) -> int:
    """
    Map a mode to a unique integer identifier; needed because of
    TEOBResumS conventions.

    Actually a bad idea, since it is non-injective when including m=0.
    """
    return int(mode.l * (mode.l - 1) / 2 + mode.m - 2)


ModeGeneratorFactory = Callable[[Mode], ModeGenerator]


def teob_mode_generator_factory(mode: Mode) -> ModeGenerator:
    try:
        from EOBRun_module import EOBRunPy  # type: ignore

        return TEOBResumSModeGenerator(eobrun_callable=EOBRunPy, mode=mode)
    except ModuleNotFoundError as e:

        return BarePostNewtonianModeGenerator(mode=mode)
