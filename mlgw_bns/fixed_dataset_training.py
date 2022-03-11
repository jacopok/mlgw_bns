from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional

import numpy as np

from .data_management import FDWaveforms
from .dataset_generation import (
    SUN_MASS_SECONDS,
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
    """Generate waveforms from a given dataset.

    Parameters
    ----------
    frequencies: np.ndarray
        In natural units.
    waveforms: FDWaveforms
        Reference waveforms.
    parameter_generator: FixedParameterGenerator
        Reference parameter generator.
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
        frequencies: Optional[np.ndarray] = None,
    ):

        assert self.frequencies is not None

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
            frequencies,
            self.waveforms.amplitudes[params.index],
        )
        resampled_phases = DownsamplingTraining.resample(
            self.frequencies, frequencies, self.waveforms.phases[params.index]
        )

        return frequencies, resampled_amplitudes, resampled_phases


def make_fixed_generation_pair(
    frequencies: np.ndarray,
    parameter_set: ParameterSet,
    waveforms: FDWaveforms,
    reference_mass: float = Dataset.total_mass,
) -> tuple[FixedParameterGenerator, FixedWaveformGenerator]:
    """Make a fixed parameter and waveform generator pair.

    Parameters
    ----------
    frequencies : np.ndarray
        In natural units.
    parameter_set : ParameterSet
    waveforms : FDWaveforms
    reference_mass : float, optional
        Reference mass in solar masses, by default Dataset.total_mass

    Returns
    -------
    tuple[FixedParameterGenerator, FixedWaveformGenerator]
        _description_
    """

    dataset = Dataset(
        initial_frequency_hz=frequencies[0] / SUN_MASS_SECONDS / reference_mass,
        srate_hz=2 * frequencies[-1] / SUN_MASS_SECONDS / reference_mass,
        parameter_generator_class=FixedParameterGenerator,
    )
    dataset.total_mass = reference_mass
    parameter_generator = FixedParameterGenerator(dataset, parameter_set)

    waveform_generator = FixedWaveformGenerator(
        frequencies,
        waveforms,
        parameter_generator,
    )

    dataset.waveform_generator = waveform_generator
    dataset.parameter_generator = parameter_generator

    return (parameter_generator, waveform_generator)
