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


def test_pca_in_model(generated_mode_model):
    pca_data = generated_mode_model.pca_data

    assert isinstance(pca_data, PrincipalComponentData)


def test_streamed_covariance_gives_the_principal_components_of_the_svd():
    """The PCA accumulated over chunks of rows (CovarianceAccumulator) is the
    one fitted on all of them at once by an SVD: the same eigenvalues and
    mean, and the same eigenvectors but for their signs."""
    from mlgw_bns.principal_component_analysis import CovarianceAccumulator

    rng = np.random.default_rng(0)
    # a decaying spectrum about a large mean
    data = 5.0 + rng.normal(size=(600, 12)) @ np.diag(np.geomspace(1, 1e-3, 12)) @ rng.normal(size=(12, 12))
    expected = PrincipalComponentAnalysisModel(6).fit(data)
    accumulator = CovarianceAccumulator()
    for chunk in np.array_split(data, 7):
        accumulator.add(chunk)
    eigenvectors, eigenvalues, mean = accumulator.principal_components(6)
    assert np.allclose(eigenvalues, expected.eigenvalues, rtol=1e-9)
    assert np.allclose(mean, expected.mean, rtol=1e-12)
    assert np.allclose(np.abs(np.sum(eigenvectors * expected.eigenvectors, axis=0)), 1.0, atol=1e-9)
