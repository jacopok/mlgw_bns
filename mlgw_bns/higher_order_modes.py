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

from .data_management import phase_unwrapping
from .dataset_generation import WaveformGenerator, WaveformParameters
from .special_func import wigner_d_function
from .pn_modes import (
    Mode,
    _post_newtonian_amplitudes_by_mode,
    _post_newtonian_phases_by_mode,
)

#: Modes with :math:`\ell < 5` and :math:`m > 0` supported by TEOBResumS.
EOB_SUPPORTED_MODES: list[Mode] = [
    Mode(l, m) for l in range(2, 5) for m in range(1, l + 1)
]

#: Modes which are summed together by :meth:`TEOBResumSModeGenerator.get_polarizations`.
SUMMED_MODES: list[Mode] = [Mode(2, 2), Mode(3, 3), Mode(2, 1), Mode(4, 4)]

#: Number of frequency bins, below the start of the requested band, for which
#: the TEOBResumS output is generated and then discarded; see
#: :func:`start_integration_early`.
N_ADDITIONAL_BINS: int = 256


def initial_frequency_scaling(modes: Iterable[Mode]) -> float:
    r"""Factor by which the TEOBResumS ``initial_frequency`` must be lowered
    in order for all the given modes to have support down to it.

    TEOBResumS interprets ``initial_frequency`` as the *gravitational wave*
    frequency of the :math:`(2, 2)` mode at the start of the ODE integration,
    i.e. twice the orbital frequency. Since the :math:`(\ell, m)` mode
    oscillates at :math:`m` times the orbital frequency, its frequency-domain
    representation only has support above

    .. math::

        f^{\ell m}_{\text{min}} = \frac{m}{2} f_{\text{initial}}

    and is identically zero below it. Asking for the :math:`(4,4)` mode on a
    grid starting at :math:`f_{\text{initial}}` therefore yields garbage on the
    whole lower half of the band. To avoid this we start the integration at
    :math:`(2 / m_{\text{max}}) f_{\text{initial}}` instead.

    The factor is capped at 1: modes with :math:`m \leq 2` already have support
    below the requested initial frequency, and lowering it further would only
    make the integration needlessly long.

    Parameters
    ----------
    modes : Iterable[Mode]
        Modes which will be generated with this parameter dictionary.

    Returns
    -------
    float
        Multiplicative factor to apply to ``initial_frequency``.

    Examples
    --------
    >>> initial_frequency_scaling([Mode(2, 2)])
    1.0
    >>> initial_frequency_scaling([Mode(2, 1)])
    1.0
    >>> initial_frequency_scaling([Mode(3, 3)])
    0.6666666666666666
    >>> initial_frequency_scaling([Mode(2, 2), Mode(4, 4)])
    0.5
    """

    return min(1.0, 2.0 / max(mode.m for mode in modes))


def start_integration_early(
    par_dict: dict,
    frequencies: Optional[np.ndarray],
    modes: Iterable[Mode],
    n_additional: int = N_ADDITIONAL_BINS,
) -> slice:
    r"""Lower the TEOBResumS initial frequency in-place, and return the slice
    which discards the extra samples thus generated.

    This is a workaround for `a TEOBResumS bug
    <https://bitbucket.org/teobresums/teobresums/issues/11/phase-gradient-is-incorrect-at-the>`_
    which makes the phase gradient incorrect in the first few samples after the
    start of the ODE integration: we start earlier and crop the output, so that
    the corrupted section falls outside the band we care about.

    The margin is measured in :math:`(2,2)` frequency bins, and it is applied
    *after* the mode-dependent rescaling of :func:`initial_frequency_scaling`.
    In the frequency of the :math:`(\ell, m)` mode itself the margin is thus
    :math:`(m / 2) n_{\text{additional}} \Delta f`, i.e. it corresponds to the
    same interval of time at the start of the integration for every mode.

    Note that ``n_additional`` alone is *not* enough for modes with
    :math:`m > 2`: no matter how many bins are added, the mode has no support
    below :math:`(m/2) f_{\text{initial}}`, which is why the rescaling is
    needed. This shows up as a jump of hundreds (:math:`(3,3)`) or thousands
    (:math:`(4,4)`) of radians in the phase residuals at low frequency.

    Parameters
    ----------
    par_dict : dict
        TEOBResumS parameter dictionary, as returned by
        :meth:`WaveformParameters.teobresums`. Modified in-place: the
        ``initial_frequency`` is lowered, and if ``frequencies`` is given the
        ``df`` key is replaced by ``interp_freqs`` and ``freqs``.
    frequencies : np.ndarray, optional
        Frequencies (natural units) at which the waveform is required.
        If ``None``, TEOBResumS's own uniform grid with spacing ``df`` is used.
    modes : Iterable[Mode]
        Modes which will be generated with this parameter dictionary.
    n_additional : int
        Number of :math:`(2,2)` frequency bins of margin.

    Returns
    -------
    slice
        Slice to be applied to the TEOBResumS output arrays in order to
        restrict them to the originally requested frequency band.
    """

    f_0 = par_dict["initial_frequency"]
    delta_f = par_dict["df"]

    new_f0 = f_0 * initial_frequency_scaling(modes) - delta_f * n_additional
    par_dict["initial_frequency"] = new_f0

    if frequencies is not None:
        par_dict["freqs"] = list(
            np.insert(
                frequencies,
                0,
                np.arange(f_0 - delta_f * n_additional, f_0, step=delta_f),
            )
        )
        par_dict.pop("df")
        par_dict["interp_freqs"] = "yes"
        return slice(-len(frequencies), None)

    # TEOBResumS returns a uniform grid starting at ``initial_frequency``,
    # so the number of samples to discard grows when it is rescaled.
    return slice(int(round((f_0 - new_f0) / delta_f)), None)


# TODO fix these, but it's not so bad now -
# these are only wrong by a constant scaling

ModeGeneratorFactory = Callable[[Mode], "ModeGenerator"]


class ModeGenerator(WaveformGenerator):
    """Generic generator of a single mode for a waveform.

    Parameters
    ----------
    mode : Mode
        Spherical-harmonic mode :math:`(\\ell, m)` to generate.
    """

    supported_modes = list(_post_newtonian_amplitudes_by_mode.keys())

    def __init__(self, mode: Mode, *args: Any, **kwargs: Any) -> None:
        self._mode: Optional[Mode] = None

        super().__init__(*args, **kwargs)  # type: ignore[misc]
        # see https://github.com/python/mypy/issues/5887 for typing problem

        self.mode = mode

    @property
    def mode(self) -> Optional[Mode]:
        """Currently selected :math:`(\\ell, m)` mode."""
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
        Spherical-harmonic mode :math:`(\\ell, m)` to generate.
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

    @staticmethod
    def _merger_time(dyn: Any, htlm: Any) -> float:
        """Merger time of a TEOBResumS run, in natural units.

        Recent TEOBResumS versions put ``EOBPars->tc`` straight into the
        dynamics dictionary. Releases up to and including the 4.4.1 sdist
        on PyPI do not, so fall back on the last sample of the shared
        time-domain grid, which is the same quantity in the FD case
        (``EOBPars->tc`` is set to ``hlm.time[-1]`` there).
        """
        if "tc" in dyn:
            return float(dyn["tc"])
        return float(np.asarray(htlm["t"])[-1])

    @staticmethod
    def _align_mode_phase_to_merger(
        phase: np.ndarray, frequencies: np.ndarray, tc: float
    ) -> np.ndarray:
        """Correct the raw ``hflm`` phase for the merger-alignment shift.

        TEOBResumS's ``time_shift_FD`` option (always on, see
        :meth:`WaveformParameters.teobresums`) shifts the *summed*
        polarizations ``hp, hc`` so that the merger sits at :math:`t=0`,
        but it does **not** touch the per-mode ``hflm`` arrays, which stay
        referenced to the start of the ODE integration. For a waveform
        starting near 20 Hz this offset is on the order of :math:`10^7 M`,
        which would otherwise show up as a huge secular term in the phase.

        Parameters
        ----------
        phase : np.ndarray
            Raw ``hflm`` phase.
        frequencies : np.ndarray
            Frequencies (natural units) matching ``phase``.
        tc : float
            Merger time, as returned by :meth:`_merger_time`.

        Returns
        -------
        np.ndarray
            Phase aligned to the same merger-at-zero convention as ``hp, hc``.
        """
        merger_time = np.asarray(tc)
        return phase - 2 * np.pi * merger_time * frequencies

    def get_polarizations(
        self,
        params: WaveformParameters,
        frequencies: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.mode is not None

        par_dict: dict = params.teobresums()

        to_slice = start_integration_early(par_dict, frequencies, SUMMED_MODES)

        par_dict["arg_out"] = "yes"
        par_dict["use_mode_lm"] = [mode_to_k(mode) for mode in SUMMED_MODES]
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
            Inclination angle :math:`\\iota` passed to TEOBResumS.

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

        to_slice = start_integration_early(par_dict, frequencies, [self.mode])

        par_dict["arg_out"] = "yes"
        par_dict["use_mode_lm"] = [mode_to_k(self.mode)]
        par_dict["inclination"] = inclination

        f_spa, hp_re, hp_im, _, _, hflm, htlm, dyn = self.eobrun_callable(par_dict)

        hp = (hp_re - 1j * hp_im)[to_slice]
        f_spa = f_spa[to_slice]

        # _, phase = phase_unwrapping(hp)
        phase = hflm[str(mode_to_k(self.mode))][1][to_slice]
        amplitude = hflm[str(mode_to_k(self.mode))][0][to_slice] * params.eta
        phase = self._align_mode_phase_to_merger(
            phase, f_spa, self._merger_time(dyn, htlm)
        )
        phase = - (phase - phase[0])

        return (f_spa, amplitude, phase)

    def effective_one_body_waveform(
        self,
        params: WaveformParameters,
        frequencies: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.mode is not None

        par_dict: dict = params.teobresums()

        to_slice = start_integration_early(par_dict, frequencies, [self.mode])

        par_dict["arg_out"] = "yes"
        par_dict["use_mode_lm"] = [mode_to_k(self.mode)]

        if self.mode == Mode(3, 3) or self.mode == Mode(2, 1) or self.mode == Mode(4, 4):
            par_dict["inclination"] = np.pi / 2

        # print(without_keys(par_dict, {"freqs"}))

        f_spa, hp_re, hp_im, _, _, hflm, htlm, dyn = self.eobrun_callable(par_dict)

        hp = (hp_re - 1j * hp_im)[to_slice]
        # hc = (hc_re - 1j * hc_im)[to_slice]
        # h = hp - 1j * hc
        f_spa = f_spa[to_slice]

        # _, phase = phase_unwrapping(hp)
        amplitude = hflm[str(mode_to_k(self.mode))][0][to_slice] * params.eta
        phase = hflm[str(mode_to_k(self.mode))][1][to_slice]
        phase = self._align_mode_phase_to_merger(
            phase, f_spa, self._merger_time(dyn, htlm)
        )
        phase = - (phase - phase[0])

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
        Spherical-harmonic indices :math:`(\\ell, m)`.
    inclination : float
        Inclination :math:`\iota`.
    azimuth : float
        Azimuth :math:`\varphi`.

    Returns
    -------
    complex
        Value of :math:`^{-2}Y_{\\ell m}` at the given angles.
    """
    y_lm_const = np.sqrt((2 * mode.l + 1) / (4 * np.pi))
    d_lm = wigner_d_function_spin_2(mode, inclination)
    y_lm = y_lm_const * d_lm * np.exp(1j * mode.m * azimuth)

    return y_lm


def spherical_harmonic_spin_2_conjugate(  # TODO: change it's name
    mode: Mode, inclination: float, azimuth: float
) -> complex:
    r"""Complex conjugate partner of :math:`^{-2}Y_{\ell m}` via :math:`(\\ell, -m)`.

    Parameters
    ----------
    mode : Mode
        Spherical-harmonic indices :math:`(\\ell, m)`.
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

    Thin :math:`s=-2` specialisation of
    :func:`~mlgw_bns.special_func.wigner_d_function`, which holds the
    single implementation of the general :math:`d^{\ell}_{m,s}`.

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
    return wigner_d_function(mode.l, mode.m, 2, inclination)


def mode_to_k(mode: Mode) -> int:
    """Map a :class:`Mode` to the integer index used by TEOBResumS.

    Notes
    -----
    Non-injective when modes with :math:`m = 0` are included.

    Parameters
    ----------
    mode : Mode
        Spherical-harmonic indices :math:`(\\ell, m)`.

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
