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


def srate_interp_scaling(modes: Iterable[Mode]) -> float:
    r"""Factor by which the TEOBResumS ``srate_interp`` must be raised so that
    the stationary-phase-approximation band ceiling covers every requested mode.

    TEOBResumS's :c:func:`SPA` routine computes the native (chirping) SPA phase
    only up to ``srate_interp / 2``; above the mode's attachment (merger)
    frequency it prolongs the phase *linearly* in :math:`f`. That continuation
    is :math:`C^1` but not :math:`C^2` -- it kills the :math:`\mathrm{d}^2
    \Psi/\mathrm{d} f^2 = 2\pi/\dot F` curvature of the real SPA -- so the join
    is a curvature kink which the subsequent cubic-spline regrid turns into a
    localised oscillation in the output phase.

    ``srate_interp`` is tuned to the :math:`(2,2)` band, i.e. its Nyquist sits
    near the :math:`(2,2)` merger. The :math:`(\ell, m)` mode chirps at
    :math:`m/2` times the orbital rate, so its merger frequency is
    :math:`\sim (m/2) f^{22}_{\text{merger}}`, which for :math:`m > 2` lands
    *inside* the requested band -- putting the kink, and its spline ringing, in
    the data we keep. Raising ``srate_interp`` by :math:`m_{\text{max}}/2`
    pushes the ceiling (and the linear tail) back above the band for every
    mode, matching the treatment the :math:`(2,2)` already gets.

    Capped below at 1 (never lower the sample rate).

    Parameters
    ----------
    modes : Iterable[Mode]
        Modes which will be generated with this parameter dictionary.

    Returns
    -------
    float
        Multiplicative factor to apply to ``srate_interp``.

    Examples
    --------
    >>> srate_interp_scaling([Mode(2, 2)])
    1.0
    >>> srate_interp_scaling([Mode(2, 1)])
    1.0
    >>> srate_interp_scaling([Mode(3, 3)])
    1.5
    >>> srate_interp_scaling([Mode(2, 2), Mode(4, 4)])
    2.0
    """

    return max(1.0, max(mode.m for mode in modes) / 2.0)


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
        ``initial_frequency`` is lowered, ``srate_interp`` is raised (see
        :func:`srate_interp_scaling`), and if ``frequencies`` is given the
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

    # Raise the SPA band ceiling so the C1-but-not-C2 linear phase tail (and its
    # cubic-spline ringing) stays above the requested band for every mode, not
    # just the (2,2); see :func:`srate_interp_scaling`.
    par_dict["srate_interp"] *= srate_interp_scaling(modes)

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

    def all_modes_amplitude_phase(
        self,
        params: WaveformParameters,
        modes: Iterable[Mode],
        frequencies: Optional[np.ndarray] = None,
    ) -> Dict[Mode, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        r"""One TEOBResumS call, ``(f, amplitude, phase)`` for every requested mode.

        TEOBResumS computes each :math:`(\ell, m)` multipole independently and
        returns them all in a single ``hflm`` dict, so asking for several modes
        at once costs one EOB evaluation instead of one per mode.

        The ODE integration start is lowered by
        :func:`initial_frequency_scaling` of the *whole set* (i.e. by the
        highest-:math:`m` mode requested), so every mode has support across the
        band; see :func:`start_integration_early`. This differs from calling
        this method once per mode: e.g. asking for ``[(2,2),(4,4)]`` together
        starts the ``(2,2)`` integration at half the frequency it would use on
        its own. The per-mode ``hflm`` multipoles do not depend on
        ``par_dict["inclination"]`` (only the summed polarizations do), so it is
        left at the :meth:`~mlgw_bns.dataset_generation.WaveformParameters.teobresums`
        default.

        The phase convention matches :meth:`effective_one_body_waveform`:
        aligned to the merger by :meth:`_align_mode_phase_to_merger` and
        sign-flipped, keeping the additive ``arg H_lm(f0)`` constant for the
        shared :class:`~mlgw_bns.neural_network.ModePhasesNN` to learn.

        Returns
        -------
        dict[Mode, tuple[np.ndarray, np.ndarray, np.ndarray]]
            ``mode -> (f_spa, amplitude, phase)``, each on the requested grid.
        """
        modes = list(modes)

        par_dict: dict = params.teobresums()

        to_slice = start_integration_early(par_dict, frequencies, modes)

        par_dict["arg_out"] = "yes"
        par_dict["use_mode_lm"] = [mode_to_k(mode) for mode in modes]

        f_spa, _, _, _, _, hflm, htlm, dyn = self.eobrun_callable(par_dict)

        f_spa = f_spa[to_slice]
        merger_time = self._merger_time(dyn, htlm)

        result: Dict[Mode, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for mode in modes:
            key = str(mode_to_k(mode))
            amplitude = hflm[key][0][to_slice] * params.eta
            phase = self._align_mode_phase_to_merger(
                hflm[key][1][to_slice], f_spa, merger_time
            )
            result[mode] = (f_spa, amplitude, -phase)
        return result

    def get_amplitude_phase_at_inclination(
        self,
        params: WaveformParameters,
        frequencies: Optional[np.ndarray] = None,
        inclination: float = np.pi / 3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Frequency, amplitude and phase for this generator's single mode.

        Thin wrapper over :meth:`all_modes_amplitude_phase`. ``inclination``
        is accepted for backwards compatibility but ignored: the per-mode
        ``hflm`` multipoles do not depend on it.
        """
        assert self.mode is not None
        return self.all_modes_amplitude_phase(params, [self.mode], frequencies)[
            self.mode
        ]

    def effective_one_body_waveform(
        self,
        params: WaveformParameters,
        frequencies: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(f, amplitude, phase)`` for this generator's single mode.

        Thin wrapper over :meth:`all_modes_amplitude_phase`; kept so that the
        single-mode :class:`~mlgw_bns.dataset_generation.WaveformGenerator`
        contract (used by ``Dataset.generate_residuals`` and every non-batched
        caller) is unchanged.
        """
        assert self.mode is not None
        return self.all_modes_amplitude_phase(params, [self.mode], frequencies)[
            self.mode
        ]


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

    Parameters
    ----------
    mode : Mode
        Spherical-harmonic indices :math:`(\\ell, m)`.
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
