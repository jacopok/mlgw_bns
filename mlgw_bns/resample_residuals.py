import numpy as np
from numba import njit  # type: ignore
from scipy.spatial import distance
import matplotlib.pyplot as plt


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

def pad_zeros_right(s, padding_length):
    # https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
    return np.pad(s, (0, padding_length), mode = 'constant', constant_values=0)

def pad_mean_right(s, padding_length):
    # https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
    return np.pad(s, (0, padding_length), mode = 'mean')

# def compare_and_plot_signals(a, b, distance_function = distance.euclidean, alignment_function = None):
def plot_signals_with_alignment(a, b, pad_function = None):
    if(len(a) != len(b) and pad_function is None):
        raise Exception(f"Signal 'a' and 'b' must be the same size; len(a)={len(a)} and len(b)={len(b)} or pad_function must not be None")
    elif(len(a) != len(b) and pad_function is not None):
        if(len(a) < len(b)):
            a = pad_function(a, len(b) - len(a))
        else:
            b = pad_function(b, len(a) - len(b))
    
    correlate_result = np.correlate(a, b, 'full')
    shift_positions = np.arange(-len(a) + 1, len(b))
    
    print("len(a)", len(a), "len(b)", len(b), "len(correlate_result)", len(correlate_result))

    fig, axes = plt.subplots(5, 1, figsize=(10, 18))
    
    axes[0].plot(a, alpha=0.7, label="a", marker="o")
    axes[0].plot(b, alpha=0.7, label="b", marker="D")
    axes[0].legend()
    axes[0].set_title("Raw Signals 'a' and 'b'")
    
    if len(shift_positions) < 20:
        # useful for debugging and showing correlation results
        print(shift_positions)
        print(correlate_result)

    best_correlation_index = np.argmax(correlate_result)
    shift_amount_debug = shift_positions[best_correlation_index]
    shift_amount = (-len(a) + 1) + best_correlation_index
    print("best_correlation_index", best_correlation_index, "shift_amount_debug", shift_amount_debug, "shift_amount", shift_amount)
    
    axes[1].stem(shift_positions, correlate_result, use_line_collection=True, label="Cross-correlation of a and b")
    axes[1].set_title(f"Cross-Correlation Result | Best Match Index: {best_correlation_index} Signal 'b' Shift Amount: {shift_amount}")
    axes[1].set_ylabel("Cross Correlation")
    axes[1].set_xlabel("'b' Signal Shift Amount")
    
    best_match_ymin = 0
    best_match_ymin_normalized = makelab.signal.map(best_match_ymin, axes[1].get_ylim()[0], axes[1].get_ylim()[1], 0, 1)
    best_match_ymax = correlate_result[best_correlation_index]
    best_match_ymax_normalized = makelab.signal.map(best_match_ymax, axes[1].get_ylim()[0], axes[1].get_ylim()[1], 0, 1)
    axes[1].axvline(shift_positions[best_correlation_index], ymin=best_match_ymin_normalized, ymax=best_match_ymax_normalized, 
                    linewidth=2, color='orange', alpha=0.8, linestyle='-.', 
                    label=f"Best match ({shift_amount}, {best_match_ymax:.2f})")
    axes[1].legend()
    
    b_shifted_mean_fill = makelab.signal.shift_array(b, shift_amount, np.mean(b))
    axes[2].plot(a, alpha=0.7, label="a", marker="o")
    axes[2].plot(b_shifted_mean_fill, alpha=0.7, label="b_shifted_mean_fill", marker="D")
    axes[2].legend()
    axes[2].set_title("Signals 'a' and 'b_shifted_mean_fill'")
    
    b_shifted_zero_fill = makelab.signal.shift_array(b, shift_amount, 0)
    axes[3].plot(a, alpha=0.7, label="a", marker="o")
    axes[3].plot(b_shifted_zero_fill, alpha=0.7, label="b_shifted_zero_fill", marker="D")
    axes[3].legend()
    axes[3].set_title("Signals 'a' and 'b_shifted_zero_fill'")
    
    b_shifted_roll = np.roll(b, shift_amount)
    axes[4].plot(a, alpha=0.7, label="a", marker="o")
    axes[4].plot(b_shifted_roll, alpha=0.7, label="b_shifted_roll", marker="D")
    axes[4].legend()
    axes[4].set_title("Signals 'a' and 'b_shifted_roll'")
    
    fig.tight_layout()
    
def compare_and_plot_signals_with_alignment(a, b, bshift_method = 'mean_fill', pad_function = None):
    '''Aligns signals using cross correlation and then plots
    
       bshift_method can be: 'mean_fill', 'zero_fill', 'roll', or 'all'. Defaults to 'mean_fill'
    '''
    
    if(len(a) != len(b) and pad_function is None):
        raise Exception(f"Signal 'a' and 'b' must be the same size; len(a)={len(a)} and len(b)={len(b)} or pad_function must not be None")
    elif(len(a) != len(b) and pad_function is not None):
        if(len(a) < len(b)):
            a = pad_function(a, len(b) - len(a))
        else:
            b = pad_function(b, len(a) - len(b))
    
    correlate_result = np.correlate(a, b, 'full')
    shift_positions = np.arange(-len(a) + 1, len(b))
    print("len(a)", len(a), "len(b)", len(b), "len(correlate_result)", len(correlate_result))
    
    euclid_distance_a_to_b = distance.euclidean(a, b)
    
    num_charts = 3
    chart_height = 3.6
    if bshift_method is 'all':
        num_charts = 5
    
    fig, axes = plt.subplots(num_charts, 1, figsize=(10, num_charts * chart_height))
    
    # Turn on markers only if < 50 points
    a_marker = None
    b_marker = None
    if len(a) < 50:
        a_marker = "o"
        b_marker = "D"
        
    axes[0].plot(a, alpha=0.7, label="a", marker=a_marker)
    axes[0].plot(b, alpha=0.7, label="b", marker=b_marker)
    axes[0].legend()
    axes[0].set_title(f"Raw Signals | Euclidean Distance From 'a' to 'b' = {euclid_distance_a_to_b:.2f}")
    
    if len(shift_positions) < 20:
        # useful for debugging and showing correlation results
        print(shift_positions)
        print(correlate_result)
    
    best_correlation_index = np.argmax(correlate_result)
    shift_amount_debug = shift_positions[best_correlation_index]
    shift_amount = (-len(a) + 1) + best_correlation_index
    print("best_correlation_index", best_correlation_index, "shift_amount_debug", shift_amount_debug, "shift_amount", shift_amount)
    
    #axes[1].plot(shift_positions, correlate_result)
    axes[1].stem(shift_positions, correlate_result, use_line_collection=True, label="Cross-correlation of a and b")
    axes[1].set_title(f"Cross-correlation result | Best match index: {best_correlation_index}; Signal 'b' shift amount: {shift_amount}")
    axes[1].set_ylabel("Cross Correlation")
    axes[1].set_xlabel("'b' Signal Shift Amount")
    
    best_match_ymin = 0
    best_match_ymin_normalized = makelab.signal.map(best_match_ymin, axes[1].get_ylim()[0], axes[1].get_ylim()[1], 0, 1)
    best_match_ymax = correlate_result[best_correlation_index]
    best_match_ymax_normalized = makelab.signal.map(best_match_ymax, axes[1].get_ylim()[0], axes[1].get_ylim()[1], 0, 1)
    axes[1].axvline(shift_positions[best_correlation_index], ymin=best_match_ymin_normalized, ymax=best_match_ymax_normalized, 
                    linewidth=2, color='orange', alpha=0.8, linestyle='-.', 
                    label=f"Best match ({shift_amount}, {best_match_ymax:.2f})")
    axes[1].legend()
    
    if bshift_method is 'mean_fill' or bshift_method is 'all':
        b_shifted_mean_fill = makelab.signal.shift_array(b, shift_amount, np.mean(b))
        euclid_distance_a_to_b_shifted_mean_fill = distance.euclidean(a, b_shifted_mean_fill)
        axes[2].plot(a, alpha=0.7, label="a", marker=a_marker)
        axes[2].plot(b_shifted_mean_fill, alpha=0.7, label="b_shifted_mean_fill", marker=b_marker)
        axes[2].legend()
        axes[2].set_title(f"Euclidean distance From 'a' to 'b_shifted_mean_fill' = {euclid_distance_a_to_b_shifted_mean_fill:.2f}")
    
    ax_idx = 0
    if bshift_method is 'zero_fill' or bshift_method is 'all':
        if bshift_method is 'zero_fill':
            ax_idx = 2
        else:
            ax_idx = 3
    
        b_shifted_zero_fill = makelab.signal.shift_array(b, shift_amount, 0)
        euclid_distance_a_to_b_shifted_zero_fill = distance.euclidean(a, b_shifted_zero_fill)
        axes[ax_idx].plot(a, alpha=0.7, label="a", marker=a_marker)
        axes[ax_idx].plot(b_shifted_zero_fill, alpha=0.7, label="b_shifted_zero_fill", marker=b_marker)
        axes[ax_idx].legend()
        axes[ax_idx].set_title(f"Euclidean distance From 'a' to 'b_shifted_zero_fill' = {euclid_distance_a_to_b_shifted_zero_fill:.2f}")
    
    
    if bshift_method is 'roll' or bshift_method is 'all':
        if bshift_method is 'roll':
            ax_idx = 2
        else:
            ax_idx = 4
        b_shifted_roll = np.roll(b, shift_amount)
        euclid_distance_a_to_b_shifted_roll = distance.euclidean(a, b_shifted_roll)
        axes[ax_idx].plot(a, alpha=0.7, label="a", marker=a_marker)
        axes[ax_idx].plot(b_shifted_roll, alpha=0.7, label="b_shifted_roll", marker=b_marker)
        axes[ax_idx].legend()
        axes[ax_idx].set_title(f"Euclidean distance From 'a' to 'b_shifted_roll' = {euclid_distance_a_to_b_shifted_roll:.2f}")
    
    fig.tight_layout()
