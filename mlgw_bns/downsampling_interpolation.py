"""Functionality for the generation of a set of indices 
which accurately represent a waveform. 

The default implementation is a greedy one, as defined in 
:class:`GreedyDownsamplingTraining`. 

To provide an alternate method, just subclass
:class:`DownsamplingTraining`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import h5py
import numpy as np
from scipy import interpolate  # type: ignore
from sortedcontainers import SortedList  # type: ignore

from .data_management import DownsamplingIndices
from .dataset_generation import Dataset


class DownsamplingTraining(ABC):
    """Selection of the downsampling indices.

    Parameters
    ----------
    dataset : Dataset
            dataset to which to refer for the generation
            of training waveforms for the downsampling.
    degree : int
            degree for the interpolation.
            Defaults to 3.
    tol : float
            Tolerance for the interpolation error.
            Defaults to ``1e-5``.
    """

    degree: int = 3

    def __init__(self, dataset: Dataset, tol: float = 1e-5):

        self.dataset = dataset
        self.tol = tol

    @abstractmethod
    def train(self, training_dataset_size: int) -> DownsamplingIndices:
        """Calcalate downsampling with a generic algoritm,
        training on a dataset with a given sizes."""

    def validate_downsampling(
        self, training_dataset_size: int, validating_dataset_size: int
    ) -> tuple[list[float], list[float]]:
        r"""Check that the downsampling is working by looking at the
        reconstruction error on a fresh dataset.

        Parameters
        ----------
        training_dataset_size : int
            How many waveforms to train the downsampling on.
        validating_dataset_size : int
            How many waveforms to validate on.

        Returns
        -------
        tuple[list[float], list[float]]
            Amplitude and phase validation errors;
            these are reported as :math:`L_\infty` errors:
            the absolute maximum of the difference.
        """

        amp_indices, phi_indices = self.train(training_dataset_size)

        frequencies, _, residuals = self.dataset.generate_residuals(
            size=validating_dataset_size
        )

        amp_residuals, phi_residuals = residuals

        amp_validation = self.validate_indices(
            amp_indices, frequencies, amp_residuals[-validating_dataset_size:]
        )
        phi_validation = self.validate_indices(
            phi_indices, frequencies, phi_residuals[-validating_dataset_size:]
        )

        return amp_validation, phi_validation

    @classmethod
    def resample(
        cls, x_ds: np.ndarray, new_x: np.ndarray, y_ds: np.ndarray
    ) -> np.ndarray:
        """Resample a function :math:`y(x)` from its values
        at certain points :math:`y_{ds} = y(x_{ds})`.

        Parameters
        ----------
        x_ds : np.ndarray
                Old, sparse :math:`x` values.
        new_x : np.ndarray
                New :math:`x` coordinates at which to evaluate the function.
        y_ds : np.ndarrays
                Old, sparse :math:`y` values.

        Returns
        -------
        new_y : np.ndarray
            Function evaluated at the coordinates ``new_x``.
        """

        if x_ds.shape != y_ds.shape:
            raise ValueError(
                f"""Shape mismatch in the downsampling arrays! 
                The shape of x_ds is {x_ds.shape} while the shape of y_ds is {y_ds.shape}."""
            )

        return interpolate.splev(
            new_x, tck=interpolate.splrep(x_ds, y_ds, s=0, k=cls.degree), der=0
        )

    def validate_indices(
        self, indices: list[int], x_val: np.ndarray, ys_val: list[np.ndarray]
    ) -> list[float]:

        validation = []
        for y_val in ys_val:
            ypred = self.resample(x_val[indices], x_val, y_val[indices])
            validation.append(max(abs(y_val - ypred)))

        return validation


class GreedyDownsamplingTraining(DownsamplingTraining):
    def indices_error(
        self, ytrue: np.ndarray, ypred: np.ndarray, current_indices: SortedList
    ) -> tuple[list[int], list[float]]:
        """Find new indices to add to the sampling.

        Arguments
        ---------
        ytrue : np.ndarray
                True values of y.
        ypred : np.ndarray
                Predicted values of y through interpolation.
                The algorithm minimizes the difference ``abs(y - ypred)``.
        current_indices : SortedList
                Indices to which the algorithm should add.
        tol : float
                Tolerance for the reconstruction error ---
                new indices are not added if the reconstruction error is below this value.

        Returns
        -------
        new_indices : list[int]
                Indices to insert among the current ones.
        errors : list[float]
                Errors (``abs(y - y_pred)``) at the points where the
                algorithm inserted the new indices.
        """

        arr = np.abs(ytrue - ypred)

        new_indices = []
        errors = []

        for key in range(len(current_indices) - 1):

            i = (
                np.argmax(arr[current_indices[key] : current_indices[key + 1]])
                + current_indices[key]
            )

            err = arr[i]

            if err > self.tol:
                new_indices.append(i)
                errors.append(err)
        return new_indices, errors

    def find_indices(
        self,
        x_train: np.ndarray,
        ys_train: list[np.ndarray],
        seeds_number: int = 4,
    ) -> list[int]:
        """Greedily downsample y(x) by making sure that the reconstruction error of each of
        the ys (instances of y(x)) is smaller than tol.

        Arguments
        ---------
            x_train : np.ndarray
                    x array
            ys : np.ndarray
                    a list of y arrays
            seeds_number : np.ndarray, optional
                    number of "seed" indices. Defaults to 4.
                    These are placed as equally spaced along the array.
                    Note: this should always be larger than the degree
                    for the interpolation.

        Returns
        -------
        indices : np.ndarray
                indices which make the interpolation errors smaller than
                the tolerance on the training dataset.
        """

        indices = SortedList(
            list(np.linspace(0, len(x_train) - 1, num=seeds_number, dtype=int))
        )

        err = self.tol + 1

        done_with_wf = np.zeros(len(ys_train), dtype=bool)

        logging.info("Starting interpolation")
        while not all(done_with_wf):

            for i, y in enumerate(ys_train):
                if done_with_wf[i]:
                    continue
                ypred = self.resample(x_train[indices], x_train, y[indices])

                indices_batch, errs = self.indices_error(y, ypred, indices)

                if len(errs) < 1:
                    done_with_wf[i] = True

                else:
                    indices.update(set(indices_batch))

                    err = min(max(errs), err)

            logging.info(
                "%i indices, error = %f = %f times the tol",
                len(indices),
                err,
                err / self.tol,
            )

        return list(indices)

    def train(self, training_dataset_size: int) -> DownsamplingIndices:
        """Compute a close-to-optimal set of indices at which to sample
        waveforms, so that the reconstruction stays below a certain tolerance.

        Parameters
        ----------
        training_dataset_size : int
            Number of waveforms to generate and with which to train.

        Returns
        -------
        tuple[list[int], list[int]]
                Indices for amplitude and phase, respectively.
        """

        generator = self.dataset.make_parameter_generator()
        param_set = self.dataset.parameter_set_cls.from_parameter_generator(
            generator, training_dataset_size
        )

        waveforms = self.dataset.generate_waveforms_from_params(param_set)
        frequencies = self.dataset.frequencies

        amp_indices = self.find_indices(frequencies, list(waveforms.amplitudes))
        phi_indices = self.find_indices(frequencies, list(waveforms.phases))

        return DownsamplingIndices(amp_indices, phi_indices)


class GreedyDownsamplingTrainingWithResiduals(GreedyDownsamplingTraining):
    def train(self, training_dataset_size: int) -> DownsamplingIndices:
        """Compute a close-to-optimal set of indices at which to sample
        waveforms, so that the reconstruction stays below a certain tolerance.

        Parameters
        ----------
        training_dataset_size : int
            Number of waveforms to generate and with which to train.

        Returns
        -------
        tuple[list[int], list[int]]
                Indices for amplitude and phase, respectively.
        """

        frequencies, _, residuals = self.dataset.generate_residuals(
            size=training_dataset_size
        )
        amp_residuals, phi_residuals = residuals

        amp_indices = self.find_indices(
            frequencies, amp_residuals[:training_dataset_size]
        )
        phi_indices = self.find_indices(
            frequencies, phi_residuals[:training_dataset_size]
        )

        return DownsamplingIndices(amp_indices, phi_indices)
