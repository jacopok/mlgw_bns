r"""Generation of waveforms from a single spherical-harmonic GW mode.

As described in `the TEOBResumS documentation
<https://bitbucket.org/eob_ihes/teobresums/wiki/Conventions,%20parameters%20and%20output>`_,
the full waveform is expressed as

.. math::

    h_+ - i h_\times = \sum_{\ell m} A_{\ell m} e^{-i \phi_{\ell m}}
    Y_{\ell m}(\iota, \varphi)

where the pair :math:`\ell, m`, with :math:`\ell \geq m`, is known as a *mode*,
and is implemented as the namedtuple :class:`~mlgw_bns.pn_modes.Mode`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.special import factorial  # type: ignore

from .data_management import phase_unwrapping
from .dataset_generation import WaveformGenerator, WaveformParameters
from .pn_modes import (
    Mode,
    _post_newtonian_amplitudes_by_mode,
    _post_newtonian_phases_by_mode,
)

#: Modes with :math:`\ell < 5` and :math:`m > 0` supported by TEOBResumS.
EOB_SUPPORTED_MODES: list[Mode] = [
    Mode(l, m) for l in range(2, 5) for m in range(1, l + 1)
]

# TODO fix these, but it's not so bad now -
# these are only wrong by a constant scaling

ModeGeneratorFactory = Callable[[Mode], "ModeGenerator"]


class ModeGenerator(WaveformGenerator):
    """Generic generator of a single mode for a waveform.

    Parameters
    ----------
    mode : Mode
        Spherical-harmonic mode :math:`(\ell, m)` to generate.
    """

    supported_modes = list(_post_newtonian_amplitudes_by_mode.keys())

    def __init__(self, mode: Mode, *args: Any, **kwargs: Any) -> None:
        self._mode: Optional[Mode] = None

        super().__init__(*args, **kwargs)  # type: ignore[misc]
        # see https://github.com/python/mypy/issues/5887 for typing problem

        self.mode = mode

    @property
    def mode(self) -> Optional[Mode]:
        """Currently selected :math:`(\ell, m)` mode."""
        return self._mode

    @mode.setter
    def mode(self, val: Mode) -> None:
        if val not in self.supported_modes and val is not None:
            raise NotImplementedError(
                f"{val} is not supported yet for {self.__class__}!"
            )

        self._mode = val


class BarePostNewtonianModeGenerator(ModeGenerator):
    """Single-mode waveform generator using post-Newtonian amplitudes and phases only."""

    def post_newtonian_amplitude(
        self, params: WaveformParameters, frequencies: np.ndarray
    ) -> np.ndarray:
        if self.mode not in _post_newtonian_amplitudes_by_mode:
            raise ValueError(
                f"No post-Newtonian amplitude defined for mode {self.mode}."
            )

        return _post_newtonian_amplitudes_by_mode[self.mode](params, frequencies)

    def post_newtonian_phase(
        self, params: WaveformParameters, frequencies: np.ndarray
    ) -> np.ndarray:
        if self.mode not in _post_newtonian_amplitudes_by_mode:
            raise ValueError(
                f"No post-Newtonian phase defined for mode {self.mode}."
            )

        return _post_newtonian_phases_by_mode[self.mode](params, frequencies)

    def effective_one_body_waveform(
        self,
        params: WaveformParameters,
        frequencies: Optional[np.ndarray] = None,
    ) -> None:
        raise NotImplementedError(
            "This generator does not include the possibility "
            "to generate effective one body waveforms"
        )


class TEOBResumSModeGenerator(BarePostNewtonianModeGenerator):
    """Single-mode generator backed by TEOBResumS via an ``eobrun`` callable.

    Parameters
    ----------
    eobrun_callable : Callable[[dict], tuple]
        Python wrapper around TEOBResumS (e.g. ``EOBRunPy``).
    mode : Mode
        Spherical-harmonic mode :math:`(\ell, m)` to generate.
    """

    supported_modes = EOB_SUPPORTED_MODES

    def __init__(
        self,
        eobrun_callable: Callable[[dict], Tuple[np.ndarray, ...]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.eobrun_callable = eobrun_callable

    def get_polarizations(
        self,
        params: WaveformParameters,
        frequencies: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.mode is not None

        par_dict: dict = params.teobresums()

        n_additional = 400000
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
        par_dict["use_mode_lm"] = [1, 4, 0, 8]  # (22, 33, 21, 44) summed
        par_dict["inclination"] = np.pi / 3

        # print(without_keys(par_dict, {"freqs"}))
        f_spa, hp_re, hp_im, hc_re, hc_im, _, _, _ = self.eobrun_callable(par_dict)

        hp = (hp_re - 1j * hp_im)[to_slice]
        hc = (hc_re - 1j * hc_im)[to_slice]
        h = hp - 1j * hc

        f_spa = f_spa[to_slice]

        return (
            f_spa,
            hp_re[to_slice],
            hp_im[to_slice],
            hc_re[to_slice],
            hc_im[to_slice],
        )

    def get_amplitude_phase_at_inclination(
        self,
        params: WaveformParameters,
        frequencies: Optional[np.ndarray] = None,
        inclination: float = np.pi / 3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return frequency, amplitude, and phase for this mode at fixed inclination.

        Used for per-mode extraction when building mode dicts for mismatch.

        Parameters
        ----------
        params : WaveformParameters
            Intrinsic and extrinsic source parameters.
        frequencies : np.ndarray, optional
            Frequency grid; if omitted, the generator default grid is used.
        inclination : float
            Inclination angle :math:`\iota` passed to TEOBResumS.

        Returns
        -------
        f_spa : np.ndarray
            Frequency samples.
        amplitude : np.ndarray
            Mode amplitude.
        phase : np.ndarray
            Unwrapped mode phase.
        """
        assert self.mode is not None

        par_dict: dict = params.teobresums()

        n_additional = 400000

        f_0 = par_dict["initial_frequency"]
        delta_f = par_dict["df"]
        par_dict["initial_frequency"] = f_0 - delta_f * n_additional
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
        par_dict["inclination"] = inclination

        f_spa, hp_re, hp_im, _, _, hflm, _, _ = self.eobrun_callable(par_dict)

        hp = (hp_re - 1j * hp_im)[to_slice]

        _, phase = phase_unwrapping(hp)
        amplitude = hflm[str(mode_to_k(self.mode))][0][to_slice] * params.eta

        return (f_spa[to_slice], amplitude, phase)

    def effective_one_body_waveform(
        self,
        params: WaveformParameters,
        frequencies: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.mode is not None

        par_dict: dict = params.teobresums()

        n_additional = 400000

        f_0 = par_dict["initial_frequency"]  # TODO: change it to par_dict["initial_frequency"]
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

        if self.mode == Mode(3, 3) or self.mode == Mode(2, 1) or self.mode == Mode(4, 4):
            par_dict["inclination"] = np.pi / 2

        # print(without_keys(par_dict, {"freqs"}))

        f_spa, hp_re, hp_im, _, _, hflm, _, _ = self.eobrun_callable(par_dict)

        hp = (hp_re - 1j * hp_im)[to_slice]
        # hc = (hc_re - 1j * hc_im)[to_slice]
        # h = hp - 1j * hc

        _, phase = phase_unwrapping(hp)
        amplitude = hflm[str(mode_to_k(self.mode))][0][to_slice] * params.eta
        # phase = hflm[str(mode_to_k(self.mode))][1][to_slice]

        f_spa = f_spa[to_slice]

        return (f_spa, amplitude, phase)


def spherical_harmonic_spin_2(
    mode: Mode, inclination: float, azimuth: float
) -> complex:
    r"""Spin-weighted spherical harmonic :math:`^{-2}Y_{\ell m}(\iota, \varphi)`.

    .. math::

        {}^{-2}Y_{\ell m}(\iota, \varphi) =
        (-1)^s \sqrt{\frac{2 \ell+1}{4 \pi}} d_{m, s}^{\ell} (\iota) e^{im \phi_0}

    with :math:`s = -2`.

    Parameters
    ----------
    mode : Mode
        Spherical-harmonic indices :math:`(\ell, m)`.
    inclination : float
        Inclination :math:`\iota`.
    azimuth : float
        Azimuth :math:`\varphi`.

    Returns
    -------
    complex
        Value of :math:`^{-2}Y_{\ell m}` at the given angles.
    """
    y_lm_const = np.sqrt((2 * mode.l + 1) / (4 * np.pi))
    d_lm = wigner_d_function_spin_2(mode, inclination)
    y_lm = y_lm_const * d_lm * np.exp(1j * mode.m * azimuth)

    return y_lm


def spherical_harmonic_spin_2_conjugate(  # TODO: change it's name
    mode: Mode, inclination: float, azimuth: float
) -> complex:
    r"""Complex conjugate partner of :math:`^{-2}Y_{\ell m}` via :math:`(\ell, -m)`.

    Parameters
    ----------
    mode : Mode
        Spherical-harmonic indices :math:`(\ell, m)`.
    inclination : float
        Inclination :math:`\iota`.
    azimuth : float
        Azimuth :math:`\varphi`.

    Returns
    -------
    complex
        Conjugate harmonic evaluated using :meth:`Mode.opposite`.
    """
    mode_opp = mode.opposite()

    y_lm_const = np.sqrt((2 * mode_opp.l + 1) / (4 * np.pi))
    d_star_lm = wigner_d_function_spin_2(mode_opp, inclination)
    y_star_lm_minus = y_lm_const * d_star_lm * np.exp(1j * mode_opp.m * azimuth)

    return y_star_lm_minus


def wigner_d_function_spin_2(mode: Mode, inclination: float) -> complex:
    r"""Wigner little-:math:`d` for spin :math:`s=-2` (Eq. II.8, arXiv:0709.0093).

    Parameters
    ----------
    mode : Mode
        Spherical-harmonic indices :math:`(\ell, m)`.
    inclination : float
        Inclination :math:`\iota`.

    Returns
    -------
    complex
        Little-:math:`d` matrix element.
    """
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
    """Map a :class:`Mode` to the integer index used by TEOBResumS.

    Notes
    -----
    Non-injective when modes with :math:`m = 0` are included.

    Parameters
    ----------
    mode : Mode
        Spherical-harmonic indices :math:`(\ell, m)`.

    Returns
    -------
    int
        TEOBResumS mode index.
    """
    return int(mode.l * (mode.l - 1) / 2 + mode.m - 2)


def teob_mode_generator_factory(mode: Mode) -> ModeGenerator:
    """Return a TEOB-backed generator, or a PN fallback if EOB is unavailable.

    Parameters
    ----------
    mode : Mode
        Spherical-harmonic mode to generate.

    Returns
    -------
    ModeGenerator
        :class:`TEOBResumSModeGenerator` when ``EOBRun_module`` is importable,
        otherwise :class:`BarePostNewtonianModeGenerator`.
    """
    try:
        from EOBRun_module import EOBRunPy  # type: ignore

        return TEOBResumSModeGenerator(eobrun_callable=EOBRunPy, mode=mode)
    except ModuleNotFoundError:
        return BarePostNewtonianModeGenerator(mode=mode)


def without_keys(d: Dict[Any, Any], keys: Iterable[Any]) -> Dict[Any, Any]:
    """Return a shallow copy of ``d`` omitting the given keys.

    Parameters
    ----------
    d : dict
        Source mapping.
    keys : iterable
        Keys to exclude.

    Returns
    -------
    dict
        Filtered mapping.
    """
    return {x: d[x] for x in d if x not in keys}
