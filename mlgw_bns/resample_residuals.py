import numpy as np
from numba import njit  # type: ignore

from .dataset_generation import Dataset, FDWaveforms, ParameterSet
from .downsampling_interpolation import DownsamplingIndices, DownsamplingTraining
from .model import Model, ParametersWithExtrinsic, compute_polarizations


class ModelPredictingInverted(Model):
    def predict(self, frequencies: np.ndarray, params: ParametersWithExtrinsic):
        """Calculate the waveforms in the plus and cross polarizations,
        accounting for extrinsic parameters

        Parameters
        ----------
        frequencies : np.ndarray
                Frequencies where to compute the waveform, in Hz.
        params : ParametersWithExtrinsic
                Parameters for the waveform, both intrinsic and extrinsic.

        Returns
        -------
        hp, hc (complex np.ndarray)
                Cartesian plus and cross-polarized waveforms, computed
                at the given frequencies, measured in 1/Hz.

        """
        assert self.downsampling_indices is not None
        assert self.nn is not None

        intrinsic_params = params.intrinsic(self.dataset)

        residuals = self.predict_residuals_bulk(
            ParameterSet.from_list_of_waveform_parameters([intrinsic_params]), self.nn
        )

        amp_residuals = self.downsampling_training.resample(
            self.dataset.frequencies_hz[self.downsampling_indices.amplitude_indices],
            frequencies,
            residuals.amplitude_residuals[0],
        )
        phi_residuals = self.downsampling_training.resample(
            self.dataset.frequencies_hz[self.downsampling_indices.phase_indices],
            frequencies,
            residuals.phase_residuals[0],
        )

        pn_amplitude = self.dataset.waveform_generator.post_newtonian_amplitude(
            intrinsic_params, frequencies * params.mass_sum_seconds
        )
        pn_phase = self.dataset.waveform_generator.post_newtonian_phase(
            intrinsic_params, frequencies * params.mass_sum_seconds
        )

        cartesian_waveform = compute_cartesian_waveform(
            amp_residuals,
            pn_amplitude,
            phi_residuals,
            pn_phase,
            params.reference_phase,
            params.time_shift,
            frequencies,
        )

        pre = self.dataset.mlgw_bns_prefactor(intrinsic_params.eta, params.total_mass)
        cosi = np.cos(params.inclination)
        pre_plus = (1 + cosi ** 2) / 2 * pre / params.distance_mpc
        pre_cross = cosi * pre * (-1j) / params.distance_mpc

        return compute_polarizations(cartesian_waveform, pre_plus, pre_cross)


@njit
def compute_cartesian_waveform(
    amplitude_residuals: np.ndarray,
    amp_pn: np.ndarray,
    phi_residuals: np.ndarray,
    phi_pn: np.ndarray,
    reference_phase: float,
    time_shift: float,
    frequencies: np.ndarray,
) -> np.ndarray:
    r"""Compute the Cartesian form of the waveform, starting
    from the residuals, the post-Newtonian baseline,
    and the parameters to add a linear term to the phase.

    This function is separated out so that it can be
    decorated with :func:`numba.njit`.

    Parameters
    ----------
    amplitude_residuals : np.ndarray
        Amplitude residuals, in the form :math:`\log (A / A_{\text{PN}})`.
    amp_pn : np.ndarray
        Post-Newtonian baseline amplitude.
    phi_residuals : np.ndarray
        Phase residuals.
    phi_pn : np.ndarray
        Post-Newtonian baseline phase
    reference_phase : float
        Overall phase to add.
    time_shift : float
        Time-domain shift in seconds, corresponds to
        a linear term added to the phase.
    frequencies : np.ndarray
        Reference frequencies, in Hz.

    Returns
    -------
    np.ndarray
        Cartesian waveform, :math:`h = A e^{i \phi}`.
    """

    amp = np.exp(amplitude_residuals) * amp_pn
    phi = (
        phi_residuals
        + phi_pn
        + reference_phase
        + (2 * np.pi * time_shift) * frequencies
    )

    return amp * np.exp(1j * phi)


def cartesian_waveforms_at_frequencies(
    waveforms: FDWaveforms,
    new_frequencies: np.ndarray,
    dataset: Dataset,
    downsampling_training: DownsamplingTraining,
    downsampling_indices: DownsamplingIndices,
) -> np.ndarray:
    """Starting from an array of downsampled waveforms decomposed into
    amplitude and phase, interpolate them to a new frequency grid and
    put them in Cartesian form.

    Parameters
    ----------
    waveforms : FDWaveforms
        Waveforms to put in Cartesian form.
    new_frequencies : np.ndarray
        Frequencies to resample to, in natural units.
    dataset : Dataset
        Reference dataset.
    downsampling_training : DownsamplingTraining
        Training model for the downsampling, contains metadata needed for the resampling.
    downsampling_indices : DownsamplingIndices
        Downsampling indices, needed for the resampling.

    Returns
    -------
    np.ndarray
        Cartesian waveforms: complex array with shape
        ``(n_waveforms, len(new_frequencies))``.
    """

    amp_frequencies = dataset.frequencies[downsampling_indices.amplitude_indices]
    phi_frequencies = dataset.frequencies[downsampling_indices.phase_indices]

    amps = np.array(
        [
            downsampling_training.resample(amp_frequencies, new_frequencies, amp)
            for amp in waveforms.amplitudes
        ]
    )

    phis = np.array(
        [
            downsampling_training.resample(phi_frequencies, new_frequencies, phi)
            for phi in waveforms.phases
        ]
    )

    return amps * np.exp(1j * phis)
