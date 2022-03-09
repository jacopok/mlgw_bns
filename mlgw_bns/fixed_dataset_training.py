from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional

import numpy as np

from .data_management import FDWaveforms
from .dataset_generation import (
    BarePostNewtonianGenerator,
    Dataset,
    ParameterGenerator,
    ParameterSet,
    WaveformParameters,
)
from .downsampling_interpolation import DownsamplingTraining


class IndexedWaveformParameters(WaveformParameters):
    """A simple subclass of WaveformParameters
    including an extra index, used to communicate
    information between a FixedParameterGenerator
    and a FixedWaveformGenerator.
    """

    def __init__(
        self,
        index: int,
        parameter_generator: "FixedParameterGenerator",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.index = index
        self.parameter_generator = parameter_generator


class FixedParameterGenerator(ParameterGenerator):
    """Generate waveform parameters by going through a fixed list.

    Parameters
    ----------
    dataset : Dataset
        Reference dataset.
    parameter_set : ParameterSet
        The parameters which will be sequentially generated.
    seed : Optional[int], optional
        RNG seed - included for compatibility, do not use.
        By default None.
    """

    def __init__(
        self,
        dataset: "Dataset",
        parameter_set: ParameterSet,
        seed: Optional[int] = None,
    ):
        super().__init__(dataset=dataset, seed=seed)
        self.parameter_set = parameter_set
        self.reset()

    def reset(self):
        self.waveform_parameters: Iterable[IndexedWaveformParameters] = (
            IndexedWaveformParameters(index, self, *params, self.dataset)
            for index, params in enumerate(self.parameter_set.parameter_array)
        )

    def __next__(self):
        return next(self.waveform_parameters)


class FixedWaveformGenerator(BarePostNewtonianGenerator):
    """Generate waveforms corresponding to

    Parameters
    ----------
    BarePostNewtonianGenerator : _type_
        _description_
    """

    def __init__(
        self,
        frequencies: np.ndarray,
        waveforms: FDWaveforms,
        parameter_generator: FixedParameterGenerator,
    ):
        self.frequencies = frequencies
        self.waveforms = waveforms
        self.parameter_generator = parameter_generator

    def effective_one_body_waveform(  # type: ignore[override]
        self,
        params: IndexedWaveformParameters,  # type: ignore[override]
        frequencies: Optional[list[float]] = None,
    ):

        if params.parameter_generator is not self.parameter_generator:
            raise NotImplementedError(
                "The parameter generator corresponding to these "
                "waveforms is not the same one which generated these parameters."
            )

        if frequencies is None:
            return (
                self.frequencies,
                self.waveforms.amplitudes[params.index],
                self.waveforms.phases[params.index],
            )

        resampled_amplitudes = DownsamplingTraining.resample(
            self.frequencies,
            np.array(frequencies),
            self.waveforms.amplitudes[params.index],
        )
        resampled_phases = DownsamplingTraining.resample(
            self.frequencies, np.array(frequencies), self.waveforms.phases[params.index]
        )

        return frequencies, resampled_amplitudes, resampled_phases


def make_fixed_generation_pair(
    frequencies: np.ndarray,
    parameter_set: ParameterSet,
    waveforms: FDWaveforms,
    reference_mass: float = Dataset.total_mass,
) -> tuple[FixedParameterGenerator, FixedWaveformGenerator]:
    """Make a fixed parameter and waveform generator pair:

    Parameters
    ----------
    frequencies : np.ndarray
        _description_
    parameter_set : ParameterSet
        _description_
    waveforms : FDWaveforms
        _description_
    reference_mass : float, optional
        _description_, by default Dataset.total_mass

    Returns
    -------
    tuple[FixedParameterGenerator, FixedWaveformGenerator]
        _description_
    """

    dataset = Dataset(initial_frequency_hz=frequencies[0], srate_hz=2 * frequencies[-1])
    dataset.total_mass = reference_mass

    parameter_generator = FixedParameterGenerator(dataset, parameter_set)

    return (
        parameter_generator,
        FixedWaveformGenerator(frequencies, waveforms, parameter_generator),
    )
