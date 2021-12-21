"""Functionality for the management of """

import logging
from typing import Optional, Tuple

import h5py
import numpy as np
from scipy import interpolate  # type: ignore
from sortedcontainers import SortedList  # type: ignore

from .dataset_generation import Dataset, save_arrays_to_file


class DownsamplingTraining:
    """Selection of the downsampling indices with a greedy algorithm.

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
            Defaults to ``1e-6``.
    """

    def __init__(self, dataset: Dataset, degree: int = 3, tol: float = 1e-6):

        self.dataset = dataset
        self.tol = tol
        self.degree = degree

    def _indices_error(
        self, ytrue: np.ndarray, ypred: np.ndarray, current_indices: SortedList
    ) -> Tuple[list[int], list[float]]:
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

        Args:
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

                indices_batch, errs = self._indices_error(y, ypred, indices)

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

    def validate_indices(
        self, indices: list[int], x_val: np.ndarray, ys_val: list[np.ndarray]
    ) -> list[float]:

        validation = []
        for y_val in ys_val:
            ypred = self.resample(x_val[indices], x_val, y_val[indices])
            validation.append(max(abs(y_val - ypred)))

        return validation

    def resample(
        self, x_ds: np.ndarray, new_x: np.ndarray, y_ds: np.ndarray
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

        return interpolate.splev(
            new_x, tck=interpolate.splrep(x_ds, y_ds, s=0, k=self.degree), der=0
        )

    def calculate_downsampling(
        self, training_dataset_size: int
    ) -> tuple[list[int], list[int]]:
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

        frequencies, amp_residuals, phi_residuals = self.dataset.generate_residuals(
            size=training_dataset_size
        )

        amp_indices = self.find_indices(
            frequencies, amp_residuals[:training_dataset_size]
        )
        phi_indices = self.find_indices(
            frequencies, phi_residuals[:training_dataset_size]
        )

        return amp_indices, phi_indices

    def save_downsampling(self, training_dataset_size: int, file: h5py.File) -> None:
        """Call the :func:`calculate_downsampling` function
        and save its result to the provided file.

        Parameters
        ----------
        training_dataset_size : int
            See :func:`calculate_downsampling`.
        file : h5py.File
            File to save the indices to.
        """
        amp_indices, phi_indices = self.calculate_downsampling(training_dataset_size)

        dict_to_save = {
            "amplitude_indices": amp_indices,
            "phase_indices": phi_indices,
            "amplitude_frequencies": self.dataset.frequencies[amp_indices],
            "phase_frequencies": self.dataset.frequencies[phi_indices],
        }

        with file:
            save_arrays_to_file(file, "downsampling_indices", dict_to_save)

    def validate_downsampling(
        self, training_dataset_size: int, validating_dataset_size: int
    ) -> tuple[list[float], list[float]]:
        """Check that the downsampling is working by looking at the
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

        amp_indices, phi_indices = self.calculate_downsampling(training_dataset_size)

        frequencies, amp_residuals, phi_residuals = self.dataset.generate_residuals(
            size=validating_dataset_size
        )
        amp_validation = self.validate_indices(
            amp_indices, frequencies, amp_residuals[-validating_dataset_size:]
        )
        phi_validation = self.validate_indices(
            phi_indices, frequencies, phi_residuals[-validating_dataset_size:]
        )

        return amp_validation, phi_validation
