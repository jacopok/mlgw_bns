from abc import ABC, abstractmethod

import numpy as np
import h5py


class Dataset:
    # saving to file to be managed with https://docs.h5py.org/en/stable/quick.html

    def __init__(self, name):
        """
        Initialize dataset.
        """

        self.name = name

    def save(self) -> None:
        pass

    def load(self) -> None:
        with h5py.File(self.name, "r") as file:
            self.frequencies = file["data"]


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
