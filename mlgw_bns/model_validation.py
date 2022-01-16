from abc import ABC
from typing import Optional, Type

import numpy as np
import pycbc.psd  # type: ignore
from scipy import integrate  # type: ignore
from scipy.optimize import minimize_scalar  # type: ignore
from tqdm import tqdm  # type: ignore

from .data_management import FDWaveforms
from .dataset_generation import Dataset, ParameterSet, WaveformGenerator
from .downsampling_interpolation import DownsamplingIndices, DownsamplingTraining
from .model import Model, cartesian_waveforms_at_frequencies


class ValidateModel:
    """Functionality for the validation of a model.

    Includes:
    - mismatch computation between model and reference
    - noise psd management
    - computation of reconstruction residuals
    """

    def __init__(
        self,
        model: Model,
        psd_name: str = "aLIGOZeroDetLowPower",
        downsample_by: int = 256,
    ):

        self.model = model

        self.pds_name: str = psd_name
        self.psd: pycbc.types.FrequencySeries = pycbc.psd.from_string(
            psd_name,
            length=len(model.dataset.frequencies) // downsample_by,
            delta_f=model.dataset.delta_f_hz * downsample_by,
            low_freq_cutoff=model.dataset.initial_frequency_hz,
        )

        psd_frequencies: pycbc.types.array.Array = self.psd.sample_frequencies

        mask = self.psd > 0

        self.frequencies = psd_frequencies[mask]
        self.psd_values = self.psd[mask]

    def psd_at_frequencies(self, frequencies: np.ndarray) -> np.ndarray:
        return np.array([self.psd.at_frequency(freq) for freq in frequencies])

    def true_and_predicted_waveforms(
        self, number_of_validation_waveforms: int
    ) -> tuple[FDWaveforms, FDWaveforms]:

        parameter_generator = self.model.dataset.make_parameter_generator()

        param_set = ParameterSet.from_parameter_generator(
            parameter_generator, number_of_validation_waveforms
        )

        true_waveforms: FDWaveforms = self.model.dataset.generate_waveforms_from_params(
            param_set, self.model.downsampling_indices
        )
        predicted_waveforms: FDWaveforms = self.model.predict_waveforms_bulk(
            param_set, self.model.nn, self.model.hyper
        )

        return true_waveforms, predicted_waveforms

    def validation_mismatches(self, number_of_validation_waveforms: int) -> list[float]:
        """Validate the model by computing the mismatch between
        theoretical waveforms and waveforms reconstructed by the model.

        Parameters
        ----------
        number_of_validation_waveforms : int
            How many validation waveforms to use.

        Returns
        -------
        list[float]
            List of mismatches.
        """

        true_waveforms, predicted_waveforms = self.true_and_predicted_waveforms(
            number_of_validation_waveforms
        )

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
    ) -> float:
        """Compute the mismatch between two Cartesian waveforms.

        The mismatch between waveforms :math:`a` and :math:`b` is defined as the
        minimum value of :math:`1 - (a|b) / \sqrt{(a|a)(b|b)}`, where
        the time shift of one of the two waveforms is changed arbitrarily,
        and where the product :math:`(a|b)` is the Wiener product.

        A custom implementation is used as opposed to the
        `pycbc one <https://pycbc.org/pycbc/latest/html/pycbc.filter.html?highlight=match#module-pycbc.filter.matchedfilter>`_
        since that one is not accurate enough, see
        `this issue <https://github.com/gwastro/pycbc/issues/3817>`_.

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
        """

        if frequencies is None:
            psd_values = self.psd_values
            frequencies = self.frequencies
        else:
            # TODO deal with the possibility that the frequencies may
            # lie outside the range of the PSD's frequencies
            # (recompute whole PSD?)
            # (or just ignore the part outside of the bounds?)
            # for now we do the latter
            mask = np.bitwise_and(
                min(self.frequencies) < frequencies,
                frequencies < max(self.frequencies) - 2 * self.model.dataset.delta_f_hz,
            )
            psd_values = self.psd_at_frequencies(frequencies[mask])
            frequencies = frequencies[mask]
            waveform_1 = waveform_1[mask]
            waveform_2 = waveform_2[mask]

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

        res = minimize_scalar(to_minimize, method="brent", bracket=(-0.1, 0.1))
        return 1 - (-res.fun) / norm
