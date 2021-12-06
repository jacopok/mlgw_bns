from abc import ABC, abstractmethod


class Dataset:
    pass


class SlowWaveformGenerator(ABC):
    @abstractmethod
    def post_newtonian_waveform(self):
        pass

    @abstractmethod
    def effective_one_body_waveform(self):
        pass


class TEOBResumSGenerator(SlowWaveformGenerator):
    def post_newtonian_waveform(self):
        pass

    def effective_one_body_waveform(self):
        pass
