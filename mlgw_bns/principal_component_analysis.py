"""Functionality for the PCA-decomposition of arbitrary data.

The classes defined here are meant to be lightweight: they do not store 
the data, instead deferring its management to the higher-level :class:`Model` class.
"""

from __future__ import annotations

import logging  # type: ignore

import numpy as np

from .data_management import DownsamplingIndices, PrincipalComponentData
from .dataset_generation import Dataset


class PrincipalComponentTraining:
    """Training and usage of a Principal Component Analysis models.

    Parameters
    ----------
    dataset: Dataset
            Used to generate the data to be used for training.
    downsampling_indices
    number_of_components: int
            Number of components to keep when reducing the dimensionality
            of the data.
    """

    def __init__(
        self,
        dataset: Dataset,
        downsampling_indices: DownsamplingIndices,
        number_of_components: int,
    ):

        self.dataset = dataset
        self.downsampling_indices = downsampling_indices
        self.pca_model = PrincipalComponentAnalysisModel(number_of_components)

    def train(self, number_of_training_waveforms: int) -> PrincipalComponentData:

        if number_of_training_waveforms < self.pca_model.number_of_components:
            logging.warn(
                "PCA can not be trained with K=%s but only %s waveforms. Aborting.",
                self.pca_model.number_of_components,
                number_of_training_waveforms,
            )
            raise ValueError

        logging.info(
            "Generating %s waveforms for PCA training", number_of_training_waveforms
        )

        _, _, residuals = self.dataset.generate_residuals(
            number_of_training_waveforms,
            self.downsampling_indices,
        )

        logging.info("Fitting PCA model")

        return self.pca_model.fit(residuals.combined)


class PrincipalComponentAnalysisModel:
    def __init__(self, number_of_components: int):
        self.number_of_components = number_of_components

    def fit(self, data: np.ndarray) -> PrincipalComponentData:
        """Fit the PCA model to this dataset.

        Parameters
        ----------
        data : np.ndarray
                Data to fit. Does not need to have zero mean.
                Should have shape
                ``(number_of_datapoints, number_of_dimensions)``

        Returns
        -------
        PrincipalComponentData
                Data describing the trained PCA model.
        """

        mean = np.mean(data, axis=0)

        zero_mean_data = data - mean[np.newaxis, :]

        # compute eigendecomposition with SVD, which is much faster!

        # eigenvalues, eigenvectors = np.linalg.eig(np.cov(zero_mean_data.T))
        U, S, V = np.linalg.svd(zero_mean_data.T, full_matrices=False)
        eigenvalues = S ** 2
        eigenvectors = U

        indices_by_magnitude = np.argsort(eigenvalues)[::-1]

        # selecting only the real part is required since in general,
        # due to potential floating point errors, these will be complex
        eigenvectors_to_keep = eigenvectors[
            :, indices_by_magnitude[: self.number_of_components]
        ].real
        eigenvalues_to_keep = eigenvalues[
            indices_by_magnitude[: self.number_of_components]
        ].real

        reduced_training_data = zero_mean_data @ eigenvectors_to_keep

        principal_components_scaling = np.max(np.abs(reduced_training_data), axis=0)

        return PrincipalComponentData(
            eigenvectors_to_keep,
            eigenvalues_to_keep,
            mean,
            principal_components_scaling,
        )

    @staticmethod
    def reduce_data(data: np.ndarray, pca_data: PrincipalComponentData) -> np.ndarray:
        """Reduce a dataset to its principal-component representation.

        Parameters
        ----------
        data : np.ndarray
            With shape ``(number_of_points, number_of_dimensions)``.
        pca_data : PrincipalComponentData
            To use in the reduction.

        Returns
        -------
        reduced_data : np.ndarray
            With shape ``(number_of_points, number_of_components)``.
        """

        zero_mean_data = data - pca_data.mean

        reduced_data = zero_mean_data @ pca_data.eigenvectors

        return reduced_data / pca_data.principal_components_scaling[np.newaxis, :]

    @staticmethod
    def reconstruct_data(
        reduced_data: np.ndarray, pca_data: PrincipalComponentData
    ) -> np.ndarray:
        """Reconstruct the data.

        Parameters
        ----------
        reduced_data : np.ndarray
            With shape ``(number_of_points, number_of_components)``.
        pca_data : PrincipalComponentData
            To use in the reconstruction.

        Returns
        -------
        reconstructed_data: np.ndarray
            With shape ``(number_of_points, number_of_dimensions)``.
        """

        # (npoints, npca) = (npoints, npca) * (npca)
        scaled_data = (
            reduced_data * pca_data.principal_components_scaling[np.newaxis, :]
        )

        # (npoints, ndims) = (npoints, npca) @ (npca, ndims)
        zero_mean_data = scaled_data @ pca_data.eigenvectors.T

        return zero_mean_data + pca_data.mean
