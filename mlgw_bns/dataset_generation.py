from abc import ABC, abstractmethod

import numpy as np
import h5py

import EOBRun_module  # type: ignore


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
        """Generate an EOB waveform with TEOB.

        Examples:
        >>> tg = TEOBResumSGenerator()
        >>> res = tg.effective_one_body_waveform()
        >>> print(isinstance(res, tuple))
        True
        """

        return tuple(
            # return EOBRun_module.EOBRunPy(
            {
                "M": 2.8,
                "distance": 1.0,
                "initial_frequency": 40,
                "use_geometric_units": 1,
                "interp_uniform_grid": 0,
                "domain": 1,
                "srate_interp": 4096.0,
                "df": 1 / 2 ** 8,
                "interp_FD_waveform": 1,
                "inclination": 0.0,
                "time_shift_FD": 1,
            }
        )
