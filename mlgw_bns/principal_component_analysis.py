"""Functionality for the PCA-decomposition of arbitrary data.

The classes defined here are meant to be lightweight: they do not store 
the data, instead deferring its management to the higher-level :class:`Model` class.
"""

from __future__ import annotations

import logging  # type: ignore

import numpy as np

from typing import Union

from .data_management import (
    DownsamplingIndices,
    PrincipalComponentData,
    array_memory,
    format_bytes,
    peak_memory_usage,
)
from .dataset_generation import Dataset
from .neural_network import TimeshiftsGPR, TimeshiftsNN


def log_expected_svd_memory(data: np.ndarray) -> None:
    """Log the memory footprint expected for the PCA of this data matrix.

    The dominant allocations of :meth:`PrincipalComponentAnalysisModel.fit`
    are the zero-mean copy of the data, the copy LAPACK makes of it
    internally, and the two factors :math:`U` and :math:`V` of the SVD, all
    of which are alive at the same time. The number of sample points per
    waveform is set by the downsampling training, so this is not predictable
    before a model is generated.

    Parameters
    ----------
    data : np.ndarray
        Matrix to be decomposed, with shape
        ``(number_of_waveforms, number_of_sample_points)``.
    """

    number_of_waveforms, number_of_points = data.shape
    smaller_dimension = min(data.shape)

    # np.linalg.svd(zero_mean_data.T, full_matrices=False) produces
    # U with shape (number_of_points, k) and V with shape (k, number_of_waveforms),
    # with k the smaller of the two dimensions, and works on its own copy
    # of the (transposed, hence non-contiguous) input.
    svd_memory = array_memory(
        (
            2 * number_of_waveforms * number_of_points
            + smaller_dimension * (number_of_waveforms + number_of_points),
        ),
        data.dtype,
    )

    logging.info(
        "Fitting PCA on a %i x %i matrix of %s residuals taking up %s; "
        "the SVD is expected to need roughly %s more, "
        "on top of the %s currently in use",
        number_of_waveforms,
        number_of_points,
        data.dtype,
        format_bytes(data.nbytes),
        format_bytes(svd_memory),
        format_bytes(peak_memory_usage()),
    )


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
        timeshifts_predictor: Union[TimeshiftsGPR, TimeshiftsNN],
    ):

        self.dataset = dataset
        self.downsampling_indices = downsampling_indices
        self.pca_model = PrincipalComponentAnalysisModel(number_of_components)
        self.timeshifts_predictor = timeshifts_predictor

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

        logging.info(
            "PCA training on %i amplitude and %i phase sample points",
            self.downsampling_indices.amp_length,
            self.downsampling_indices.phi_length,
        )

        freq_downsampled, parameters, residuals = self.dataset.generate_residuals(
            number_of_training_waveforms,
            self.downsampling_indices,
            flatten_phase=False
        )

        residuals.phase_residuals = remove_linear_trend(
            parameters=parameters,
            phi_diff=residuals.phase_residuals,
            frq=self.dataset.natural_units_to_hz(freq_downsampled),
            timeshifts_predictor=self.timeshifts_predictor,
        )

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

        log_expected_svd_memory(data)

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

        logging.info(
            "PCA fit done, keeping %i of %i components "
            "(peak memory usage so far: %s)",
            self.number_of_components,
            len(eigenvalues),
            format_bytes(peak_memory_usage()),
        )

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
    
    @staticmethod
    def calculate_total_variance(pca_data: PrincipalComponentData) -> float:
        """Calculate the total variance explained by all principal components.

        Parameters
        ----------
        pca_data : PrincipalComponentData
            The PCA data containing eigenvalues.

        Returns
        -------
        float
            The total variance (sum of all eigenvalues).
        """
        return np.sum(pca_data.eigenvalues)
    
    @staticmethod
    def calculate_individual_variance_ratio(pca_data: PrincipalComponentData) -> np.ndarray:
        """Calculate the individual variance ratio explained by each principal component.

        Parameters
        ----------
        pca_data : PrincipalComponentData
            The PCA data containing eigenvalues.

        Returns
        -------
        np.ndarray
            Array of individual variance ratios, where each element represents the
            proportion of variance explained by that principal component.
            Shape is (number_of_components,)
        """
        total_variance = PrincipalComponentAnalysisModel.calculate_total_variance(pca_data)
        return pca_data.eigenvalues / total_variance

    @staticmethod
    def calculate_cumulative_variance_ratio(pca_data: PrincipalComponentData) -> np.ndarray:
        """Calculate the cumulative variance ratio explained by principal components.

        Parameters
        ----------
        pca_data : PrincipalComponentData
            The PCA data containing eigenvalues.

        Returns
        -------
        np.ndarray
            Array of cumulative variance ratios, where each element represents the
            proportion of variance explained up to that principal component.
            Shape is (number_of_components,)
        """
        total_variance = PrincipalComponentAnalysisModel.calculate_total_variance(pca_data)
        return np.cumsum(pca_data.eigenvalues) / total_variance
    
def remove_linear_trend(parameters, phi_diff, frq, timeshifts_predictor):

    for i in range(parameters.parameter_array.shape[0]):
        phi_diff[i] = (
            phi_diff[i]
            - 2 * np.pi * (frq - frq[0]) * timeshifts_predictor.predict([parameters.parameter_array[i]])
            - phi_diff[i,0]
        )

    return phi_diff
