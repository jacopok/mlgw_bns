from __future__ import annotations

from abc import ABC
from functools import cached_property
from pathlib import Path
from typing import Callable, Optional, Type

import numpy as np
from scipy import integrate  # type: ignore
from scipy.interpolate import interp1d  # type: ignore
from scipy.optimize import minimize_scalar  # type: ignore
from tqdm import tqdm  # type: ignore

from .data_management import FDWaveforms
from .dataset_generation import Dataset, ParameterSet, WaveformGenerator
from .downsampling_interpolation import DownsamplingIndices, DownsamplingTraining
from .model import Model
from .resample_residuals import cartesian_waveforms_at_frequencies

PSD_PATH = Path(__file__).parent / "data"


class ValidateModel:
    r"""Functionality for the validation of a model.

    Parameters
    ----------
    model: Model
            Model to validate.
    psd_name: str
            Name of the power spectral density to use in the computation
            of the mismatches.
            Currently only 'ET' (default) is supported
    """

    def __init__(
        self,
        model: Model,
        psd_name: str = "ET",
    ):

        self.model = model
        self.psd_name: str = psd_name
        self.psd_data = np.loadtxt(PSD_PATH / f"{self.psd_name}_psd.txt")

        all_frequencies = self.psd_data[:, 0]
        mask = np.where(
            np.logical_and(
                all_frequencies < self.model.dataset.effective_srate_hz / 2,
                all_frequencies > self.model.dataset.effective_initial_frequency_hz,
            )
        )

        self.frequencies = self.psd_data[:, 0][mask]
        self.psd_values = self.psd_data[:, 1][mask]

    @cached_property
    def psd_at_frequencies(self) -> Callable[[np.ndarray], np.ndarray]:
        """Compute the given PSD

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies at which to compute the PSD, in Hz.

        Returns
        -------
        np.ndarray
            Values of the PSD, :math:`S_n(f_i)`.
        """
        return interp1d(
            self.frequencies,
            self.psd_values,
        )

    def param_set(
        self, number_of_parameter_tuples: int, seed: Optional[int] = None
    ) -> ParameterSet:
        """Generate a random set of parameters, using the
        parameter generator in :attr:`model.dataset`.

        Parameters
        ----------
        number_of_parameter_tuples : int
            How many tuples of parameters to generate
        seed : int, optional
            Seed used to initialize the parameter generator.
            By default None, which means the seed is computed
            from the dataset's default RNG.

        Returns
        -------
        ParameterSet
            A set of uniform parameter tuples
        """

        parameter_generator = self.model.dataset.make_parameter_generator(seed)

        return ParameterSet.from_parameter_generator(
            parameter_generator, number_of_parameter_tuples
        )

    def true_waveforms(self, param_set: ParameterSet) -> FDWaveforms:
        """Waveforms corresponding to the given parameter set,
        computed according to the EOB generator.

        Parameters
        ----------
        param_set : ParameterSet
            Parameters at which to generate the waveforms.

        Returns
        -------
        FDWaveforms
            EOB waveforms at the given parameters.
        """

        return self.model.dataset.generate_waveforms_from_params(
            param_set, self.model.downsampling_indices
        )

    def predicted_waveforms(self, param_set: ParameterSet) -> FDWaveforms:
        """Waveforms corresponding to the given parameter set,
        reconstruced by the :attr:`model`.

        Parameters
        ----------
        param_set : ParameterSet
            Parameters at which to generate the waveforms.

        Returns
        -------
        FDWaveforms
            Reconstructed waveforms at the given parameters.
        """

        return self.model.predict_waveforms_bulk(param_set, self.model.nn)

    def post_newtonian_waveforms(self, param_set: ParameterSet) -> FDWaveforms:
        """Waveforms corresponding to the given parameter set,
        computed according to the post-Newtonian baseline.

        Parameters
        ----------
        param_set : ParameterSet
            Parameters at which to generate the waveforms.

        Returns
        -------
        FDWaveforms
            PN Waveforms at the given parameters.
        """

        assert self.model.nn is not None
        residuals = self.model.predict_residuals_bulk(param_set, self.model.nn)
        residuals.amplitude_residuals[:] = 0
        residuals.phase_residuals[:] = 0

        return self.model.dataset.recompose_residuals(
            residuals, param_set, self.model.downsampling_indices
        )

    def validation_mismatches(
        self,
        number_of_validation_waveforms: int,
        seed: Optional[int] = None,
        true_waveforms: Optional[FDWaveforms] = None,
        zero_residuals: bool = False,
    ) -> list[float]:
        """Validate the model by computing the mismatch between
        theoretical waveforms and waveforms reconstructed by the model.

        Parameters
        ----------
        number_of_validation_waveforms : int
            How many validation waveforms to use.
        true_waveforms: FDWaveforms, optional
            True waveforms to compare to.
            This parameter should be used in order to not recompute
            the true waveforms each time when comparing different models;
            do not use the same waveforms for different models,
            since the downsampling indices may be different.
            Defaults to None, which means the true waveforms are recomputed.
        zero_residuals: bool
            Whether to set the residuals to zero, meaning that
            the model is not used at all, instead just comparing
            the EOB waveforms to the PN baseline.
            Defaults to False.
        seed : int, optional
            Seed to give to the parameter generation.
            Defaults to None.

        Returns
        -------
        list[float]
            List of mismatches.
        """

        param_set = self.param_set(number_of_validation_waveforms, seed)

        if true_waveforms is None:
            true_waveforms = self.true_waveforms(param_set)

        if zero_residuals:
            predicted_waveforms = self.post_newtonian_waveforms(param_set)
        else:
            predicted_waveforms = self.predicted_waveforms(param_set)

        return self.mismatch_array(true_waveforms, predicted_waveforms)

    def mismatch_array(
        self, waveform_array_1: FDWaveforms, waveform_array_2: FDWaveforms
    ) -> list[float]:
        """Compute the mismatches between each of the waveforms in these two lists.

        Parameters
        ----------
        waveform_array_1 : FDWaveforms
            First set of waveforms to compare.
        waveform_array_2 : FDWaveforms
            Second set of waveforms to compare.

        Returns
        -------
        list[float]
            Mismatches between the waveforms, in order.
        """
        assert self.model.downsampling_indices is not None

        cartesian_1 = cartesian_waveforms_at_frequencies(
            waveform_array_1,
            self.model.dataset.hz_to_natural_units(self.frequencies),
            self.model.dataset,
            self.model.downsampling_training,
            self.model.downsampling_indices,
        )

        cartesian_2 = cartesian_waveforms_at_frequencies(
            waveform_array_2,
            self.model.dataset.hz_to_natural_units(self.frequencies),
            self.model.dataset,
            self.model.downsampling_training,
            self.model.downsampling_indices,
        )

        return [
            self.mismatch(waveform_1, waveform_2)
            for waveform_1, waveform_2 in tqdm(
                zip(cartesian_1, cartesian_2), unit="mismatches"
            )
        ]

    def mismatch(
        self,
        waveform_1: np.ndarray,
        waveform_2: np.ndarray,
        frequencies: Optional[np.ndarray] = None,
        max_delta_t: float = 0.07,
    ) -> float:
        r"""Compute the mismatch between two Cartesian waveforms.

        The mismatch between waveforms :math:`a` and :math:`b` is defined as the
        minimum value of :math:`1 - (a|b) / \sqrt{(a|a)(b|b)}`, where
        the time shift of one of the two waveforms is changed arbitrarily,
        and where the product :math:`(a|b)` is the Wiener product.

        A custom implementation is used as opposed to the
        `pycbc one <https://pycbc.org/pycbc/latest/html/pycbc.filter.html?highlight=match#module-pycbc.filter.matchedfilter>`_
        since that one is not accurate enough, see
        `this issue <https://github.com/gwastro/pycbc/issues/3817>`_.

        The implementation here uses scipy's `scalar minimizer <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html>`_ to find the minimum of the mismatch.

        Parameters
        ----------
        waveform_1 : np.ndarray
            First Cartesian waveform to compare.
        waveform_2 : np.ndarray
            Second Cartesian waveform to compare.
        frequencies : np.ndarray, optional
            Frequencies at which the two waveforms are sampled, in Hz.
            If None (default), it is assumed that the waveforms are sampled at
            the attribute :attr:`frequencies` of this object.
        max_delta_t: float
            Maximum time shift for the two waveforms which are being compared,
            in seconds.
            Defaults to 0.07.
        """

        if frequencies is None:
            frequencies = self.frequencies
            psd_values = self.psd_values
        else:
            psd_values = self.psd_at_frequencies(frequencies)

        def product(a: np.ndarray, b: np.ndarray) -> float:
            integral = integrate.trapezoid(np.conj(a) * b / psd_values, x=frequencies)
            return abs(integral)

        norm = np.sqrt(
            product(waveform_1, waveform_1) * product(waveform_2, waveform_2)
        )

        def to_minimize(t_c: float) -> float:
            assert frequencies is not None
            offset = np.exp(2j * np.pi * (frequencies * t_c))
            return -product(waveform_1, waveform_2 * offset)

        res = minimize_scalar(
            to_minimize, method="brent", bracket=(-max_delta_t, max_delta_t)
        )

        if not res.success:
            raise ValueError("Mismatch optimization did not succeed!")

        return 1 - (-res.fun) / norm
