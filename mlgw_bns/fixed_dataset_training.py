from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .dataset_generation import Dataset

from .data_management import FDWaveforms
from .dataset_generation import (
    BarePostNewtonianGenerator,
    ParameterGenerator,
    ParameterSet,
    WaveformParameters,
)


class IndexedWaveformParameters(WaveformParameters):
    """A simple subclass of WaveformParameters
    including an extra index, used to communicate
    information between a FixedParameterGenerator
    and a FixedWaveformGenerator.
    """

    def __init__(self, index: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = index


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
        waveform_parameters = parameter_set.waveform_parameters(dataset)

        self.waveform_parameters = [
            IndexedWaveformParameters(index, *params, dataset)
            for index, params in enumerate(parameter_set.parameter_array)
        ]

    def __next__(self):
        return next(self.waveform_parameters)


class FixedWaveformGenerator(BarePostNewtonianGenerator):
    def __init__(self, waveforms: FDWaveforms):
        self.waveforms = waveforms

    def effective_one_body_waveform(self, params):
        pass
