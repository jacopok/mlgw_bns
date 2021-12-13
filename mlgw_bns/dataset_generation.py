from abc import ABC, abstractmethod

import numpy as np
import h5py

import EOBRun_module  # type: ignore


class Dataset:
    r"""A dataset of data generated with some model.

    Includes:

    * frequencies at which the data are sampled
    * frequency indices
    * amplitude and phase residuals for all modes

    The amplitude residuals are defined as
    :math:`\log(A _{\text{EOB}} / A_{\text{PN}})`,
    while the phase residuals are defined as
    :math:`\phi _{\text{EOB}} - \phi_{\text{PN}}`.

    """

    # saving to file to be managed with https://docs.h5py.org/en/stable/quick.html

    def __init__(self, filename: str):
        r"""
        Initialize dataset.

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
        r"""Generate an EOB waveform with TEOB.

        Examples:
        >>> tg = TEOBResumSGenerator()
        >>> res = tg.effective_one_body_waveform()
        """

        pass
