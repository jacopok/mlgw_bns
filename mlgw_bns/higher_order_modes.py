from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .dataset_generation import (
    WaveformGenerator,
    WaveformParameters,
    amplitude_3h_post_newtonian,
    phase_5h_post_newtonian_tidal,
)

WaveformCallable = Callable[[WaveformParameters, np.ndarray], np.ndarray]

_post_newtonian_amplitudes_by_mode: dict[tuple[int, int], WaveformCallable] = {
    (2, 2): amplitude_3h_post_newtonian
}
_post_newtonian_phases_by_mode: dict[tuple[int, int], WaveformCallable] = {
    (2, 2): phase_5h_post_newtonian_tidal
}


class ModeGenerator(WaveformGenerator):
    """Generic generator of a single mode for a waveform."""

    def __init__(self, mode: tuple[int, int], *args, **kwargs):
        super().__init__(*args, **kwargs)  # type: ignore
        # see (https://github.com/python/mypy/issues/5887) for typing problem
        self.mode = mode

        # TODO improve the way these are handled
        self.supported_modes = list(_post_newtonian_amplitudes_by_mode.keys())

    def __post_init__(self):
        if self.mode not in self.supported_modes:
            raise NotImplementedError(f"Mode {self.mode} is not supported yet!")


class BarePostNewtonianModeGenerator(ModeGenerator):
    def post_newtonian_amplitude(
        self, params: "WaveformParameters", frequencies: np.ndarray
    ) -> np.ndarray:
        return _post_newtonian_amplitudes_by_mode[self.mode](params, frequencies)

    def post_newtonian_phase(
        self, params: "WaveformParameters", frequencies: np.ndarray
    ) -> np.ndarray:
        return _post_newtonian_phases_by_mode[self.mode](params, frequencies)

    def effective_one_body_waveform(
        self, params: "WaveformParameters", frequencies: Optional[np.ndarray] = None
    ):
        raise NotImplementedError(
            "This generator does not include the possibility "
            "to generate effective one body waveforms"
        )


class EffectiveOneBodyModeGenerator(BarePostNewtonianModeGenerator):
    def __init__(self, eobrun_callable: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eobrun_callable = eobrun_callable

    def effective_one_body_waveform(
        self, params: "WaveformParameters", frequencies: Optional[np.ndarray] = None
    ):
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
        par_dict["output_multipoles"] = "yes"
        par_dict["use_mode_lm"] = [mode_to_k(self.mode)]
        par_dict["output_lm"] = [mode_to_k(self.mode)]

        f_spa, _, _, _, _, hflm, _, _ = self.eobrun_callable(par_dict)

        amplitude = hflm[mode_to_k(self.mode)][0]
        phase = hflm[mode_to_k(self.mode)][1]

        return (f_spa, amplitude, phase)


def mode_to_k(mode: tuple[int, int]):
    """
    Map (l,m) -> k
    """
    return int(mode[0] * (mode[0] - 1) / 2 + mode[1] - 2)
