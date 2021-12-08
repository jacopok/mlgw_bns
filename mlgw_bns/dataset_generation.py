from abc import ABC, abstractmethod

import numpy as np
import h5py


class Dataset:
    """A dataset of data generated with some model.

    Includes:

    * frequencies at which the data are sampled
    * frequency indices
    * amplitude and phase residuals for all modes

    """

    # saving to file to be managed with https://docs.h5py.org/en/stable/quick.html

    def __init__(self, filename: str):
        r"""
        Initialize dataset.

        It runs in :math:`\mathcal{O}(1)` time.

        Parameters
        ----------
        filename : str
                Name of the file where to save the data
                in hdf5 format.

        Examples
        --------
        >>> dataset = Dataset('data')
        >>> print(dataset.filename)
        data
        """

        self.filename = filename

    def save(self) -> None:
        pass

    def load(self) -> None:
        with h5py.File(self.filename, "r") as file:
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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
