#%%

from abc import ABC, abstractmethod


class Dataset:
    pass


class SlowWaveformGenerator(ABC):

    # abstract:
    # PN waveform
    # EOB waveform
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


# %%
