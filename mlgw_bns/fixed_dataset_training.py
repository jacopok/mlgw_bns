from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .dataset_generation import Dataset

from .dataset_generation import (
    ParameterGenerator,
    BarePostNewtonianGenerator,
    ParameterSet,
    WaveformParameters,
)
from .data_management import FDWaveforms


class IndexedWaveformParameters(WaveformParameters):
    def __init__(self, index: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = index


class FixedParameterGenerator(ParameterGenerator):
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
        pass

    def effective_one_body_waveform(self, params):
        pass
