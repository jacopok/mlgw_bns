import numpy as np

from mlgw_bns.data_management import PrincipalComponentData
from mlgw_bns.principal_component_analysis import PrincipalComponentAnalysisModel


def test_pca_model_reconstruction_exact(random_array):
    """If the number of principal components used is the number of dimensions,
    the reconstruction should be exact."""

    n_data, n_dims = random_array.shape

    pca_model = PrincipalComponentAnalysisModel(n_dims)
    pca_data = pca_model.fit(random_array)
    reduced_array = pca_model.reduce_data(random_array, pca_data)
    reconstructed_array = pca_model.reconstruct_data(reduced_array, pca_data)

    assert np.allclose(reconstructed_array, random_array, atol=0, rtol=1e-8)


def test_pca_model_reconstruction_inexact(random_array):
    """If the number of principal components used is lower than
    the number of dimensions, the reconstruction will not be exact.

    Since the data used is very close to fully independent,
    we still must use many components.
    """

    n_data, n_dims = random_array.shape

    pca_model = PrincipalComponentAnalysisModel(80)
    pca_data = pca_model.fit(random_array)
    reduced_array = pca_model.reduce_data(random_array, pca_data)
    reconstructed_array = pca_model.reconstruct_data(reduced_array, pca_data)

    assert np.allclose(reconstructed_array, random_array, atol=1e-2, rtol=1e-2)
    assert np.average(abs(reconstructed_array - random_array)) < 1e-3


def test_pca_in_model(generated_model):
    pca_data = generated_model.pca_data

    assert isinstance(pca_data, PrincipalComponentData)
