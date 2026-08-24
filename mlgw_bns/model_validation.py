r"""Validation utilities for trained waveform surrogate models.

This module provides :class:`ValidateModel`, the main entry point for
quantifying the accuracy of a trained :class:`~mlgw_bns.model.Model` by
computing noise-weighted mismatches between EOB reference waveforms and
their surrogate reconstructions.

Three flavours of mismatch are exposed:

:meth:`ValidateModel.mismatch` computes the single dominant-mode
Cartesian waveform mismatch, marginalised analytically over a global
phase and numerically over a time shift.
:meth:`ValidateModel.full_waveform_mismatch` computes the multi-mode
mismatch with numerical marginalisation over both a time shift and a
reference azimuthal phase, suitable for waveforms with higher-order
modes.
:meth:`ValidateModel.sky_maximized_mismatch` additionally maximises
over the polarisation angle :math:`\kappa`.

The class also wraps the random generation of validation parameter
sets, the application of the learned merger time-shift correction
between modes (cached :meth:`ValidateModel.time_shifts_predictor`),
and the PSD-weighted inner product used by all of the above.
"""
from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import integrate  # type: ignore
from scipy.interpolate import interp1d  # type: ignore
from scipy.optimize import dual_annealing, minimize, minimize_scalar  # type: ignore
from tqdm import tqdm  # type: ignore

from .data_management import FDWaveforms
from .dataset_generation import ParameterSet
from .model import Model
from .neural_network import (
    NeuralNetwork,
    TimeshiftsGPR,
    TimeshiftsNN,
    load_timeshifts_predictor,
)
from .resample_residuals import cartesian_waveforms_at_frequencies


PSD_PATH: Path = Path(__file__).parent / "data"
TIMESHIFTS_NN_PATH: Path = (
    Path(__file__).resolve().parent.parent / "timeshifts_rff_surrogate.pkl"
)
TIMESHIFTS_GPR_PATH: Path = (
    Path(__file__).resolve().parent.parent / "timeshifts_model_HOM.pkl"
)


class ValidateModel:
    r"""Functionality for the validation of a trained waveform model.

    Parameters
    ----------
    model : Model
            Model to validate.
    psd_name : str
            Name of the power spectral density to use in the computation
            of the mismatches. Currently only ``"ET"`` (default) is
            supported.
    custom_frequencies : bool
            Whether the model is trained on the custom frequencies of
            the dataset, or the default frequencies of the dataset.
            Defaults to ``True``. If ``False``, the model is trained on
            the default frequencies of the dataset. This is useful for
            comparing the model to EOB waveforms, since the EOB
            waveforms are computed at the default frequencies of the
            dataset.
    """

    def __init__(
        self,
        model: Model,
        psd_name: str = "ET",
        custom_frequencies: bool = True,
    ) -> None:
        self.model = model
        self.psd_name: str = psd_name
        self.psd_data = np.loadtxt(PSD_PATH / f"{self.psd_name}_psd.txt")

        all_frequencies = self.psd_data[:, 0]
        if custom_frequencies:
            mask = np.where(
                np.logical_and(
                    all_frequencies < self.model.dataset.frequencies_hz[-1] + 10,
                    all_frequencies > self.model.dataset.frequencies_hz[0] - 0.5,
                )
            )
        else:
            mask = np.where(
                np.logical_and(
                    all_frequencies < self.model.dataset.effective_srate_hz / 2,
                    all_frequencies > self.model.dataset.effective_initial_frequency_hz,
                )
            )

        self.frequencies = self.psd_data[:, 0][mask]
        self.psd_values = self.psd_data[:, 1][mask]

        # Lazy cache for the merger-time-shift predictor, populated on
        # first call to :meth:`time_shifts_predictor`.
        self._time_shifts_predictor: Optional[Union[TimeshiftsNN, TimeshiftsGPR]] = None

    @cached_property
    def psd_at_frequencies(self) -> Callable[[np.ndarray], np.ndarray]:
        """Interpolator returning the PSD :math:`S_n(f)` at arbitrary frequencies.

        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
                Linear interpolator over :attr:`frequencies` that, when
                called with an array of frequencies in Hz, returns the
                corresponding PSD values.
        """
        return interp1d(
            self.frequencies,
            self.psd_values,
        )

    def time_shifts_predictor(self) -> Union[TimeshiftsNN, TimeshiftsGPR]:
        """Return the merger-time-shift predictor, loading it on first use.

        The lightweight RFF+Ridge :class:`TimeshiftsNN` checkpoint is
        tried first, and the heavier :class:`TimeshiftsGPR` checkpoint
        is used as a fallback. The result is cached on the instance so
        that subsequent calls do not re-read the pickle from disk.

        Returns
        -------
        Union[TimeshiftsNN, TimeshiftsGPR]
                Loaded predictor instance.

        Raises
        ------
        ValueError
                If neither checkpoint at :data:`TIMESHIFTS_NN_PATH` nor
                :data:`TIMESHIFTS_GPR_PATH` could be loaded.
        """
        if self._time_shifts_predictor is None:
            self._time_shifts_predictor = load_timeshifts_predictor(
                str(TIMESHIFTS_NN_PATH),
                str(TIMESHIFTS_GPR_PATH),
            )
        return self._time_shifts_predictor

    def param_set(
        self,
        number_of_parameter_tuples: int,
        seed: Optional[int] = None,
    ) -> ParameterSet:
        """Generate a random set of parameters from the model's parameter generator.

        Parameters
        ----------
        number_of_parameter_tuples : int
                How many tuples of parameters to generate.
        seed : int, optional
                Seed used to initialize the parameter generator. By
                default ``None``, which means the seed is computed from
                the dataset's default RNG.

        Returns
        -------
        ParameterSet
                A set of uniform parameter tuples.
        """
        parameter_generator = self.model.dataset.make_parameter_generator(seed)

        return ParameterSet.from_parameter_generator(
            parameter_generator, number_of_parameter_tuples
        )

    def mismatches_for_params(
        self,
        param_set: ParameterSet,
        nn: NeuralNetwork,
        include_time_shifts: bool = True,
        disable_tqdm: bool = True,
    ) -> List[float]:
        """Compute mismatches between EOB and model-predicted waveforms.

        Used during hyperparameter optimization to evaluate validation
        mismatch without attaching ``nn`` to the model.

        Parameters
        ----------
        param_set : ParameterSet
                Parameters at which to compute mismatches.
        nn : NeuralNetwork
                Trained network to use for predictions.
        include_time_shifts : bool
                Whether to apply time-shift correction to predicted
                waveforms. Defaults to ``True``.
        disable_tqdm : bool
                Whether to disable the tqdm progress bar. Defaults to
                ``True`` for use in optimization loops.

        Returns
        -------
        List[float]
                Mismatch for each waveform in the parameter set.
        """
        true_waveforms, valid_param_set = self.true_waveforms(param_set)
        phase0_eob = np.copy(true_waveforms.phases)
        true_waveforms.phases -= phase0_eob[:, 0].reshape(-1, 1)

        predicted_waveforms = self.model.predict_waveforms_bulk(valid_param_set, nn)

        if include_time_shifts:
            self._apply_predicted_time_shifts(predicted_waveforms, valid_param_set)

        return self._mismatch_array_internal(
            true_waveforms, predicted_waveforms, disable_tqdm=disable_tqdm
        )

    def true_waveforms(
        self,
        param_set: ParameterSet,
    ) -> Tuple[FDWaveforms, ParameterSet]:
        """EOB waveforms corresponding to the given parameter set.

        For modes :math:`(2,1)` and :math:`(3,3)`, parameter tuples with
        ``amp_teob <= 0`` are discarded. The returned :class:`ParameterSet`
        contains only the valid parameters.

        Parameters
        ----------
        param_set : ParameterSet
                Parameters at which to generate the waveforms.

        Returns
        -------
        Tuple[FDWaveforms, ParameterSet]
                EOB waveforms and the parameter set that produced them
                (filtered for modes 21/33 when ``amp_teob <= 0``).
        """
        return self.model.dataset.generate_waveforms_from_params(
            param_set, self.model.downsampling_indices
        )

    def predicted_waveforms(self, param_set: ParameterSet) -> FDWaveforms:
        """Waveforms reconstructed by :attr:`model` at the given parameters.

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
        """Post-Newtonian baseline waveforms for the given parameter set.

        Parameters
        ----------
        param_set : ParameterSet
                Parameters at which to generate the waveforms.

        Returns
        -------
        FDWaveforms
                PN waveforms at the given parameters.
        """
        assert self.model.nn is not None
        residuals = self.model.predict_residuals_bulk(param_set, self.model.nn)
        residuals.amplitude_residuals[:] = 0
        residuals.phase_residuals[:] = 0

        return self.model.dataset.recompose_residuals(
            residuals, param_set, self.model.downsampling_indices
        )

    def waveforms(
        self,
        waveform_array_1: FDWaveforms,
        waveform_array_2: FDWaveforms,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample two :class:`FDWaveforms` arrays to the validator frequencies.

        Both waveform arrays are converted from the model's downsampled
        amplitude/phase representation to Cartesian (complex-valued)
        waveforms on :attr:`frequencies`. This is the common preprocessing
        step used by every mismatch routine on this class.

        Parameters
        ----------
        waveform_array_1 : FDWaveforms
                First set of waveforms.
        waveform_array_2 : FDWaveforms
                Second set of waveforms.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
                The two arrays of Cartesian waveforms sampled on
                :attr:`frequencies`.
        """
        assert self.model.downsampling_indices is not None

        natural_frequencies = self.model.dataset.hz_to_natural_units(self.frequencies)

        cartesian_1 = cartesian_waveforms_at_frequencies(
            waveform_array_1,
            natural_frequencies,
            self.model.dataset,
            self.model.downsampling_training,
            self.model.downsampling_indices,
        )

        cartesian_2 = cartesian_waveforms_at_frequencies(
            waveform_array_2,
            natural_frequencies,
            self.model.dataset,
            self.model.downsampling_training,
            self.model.downsampling_indices,
        )

        return cartesian_1, cartesian_2

    def validation_mismatches(
        self,
        number_of_validation_waveforms: int,
        seed: Optional[int] = None,
        include_time_shifts: bool = False,
        true_waveforms: Optional[FDWaveforms] = None,
        zero_residuals: bool = False,
        save_params: bool = False,
    ) -> List[float]:
        """Validate the model by computing the dominant-mode mismatch.

        Parameters
        ----------
        number_of_validation_waveforms : int
                How many validation waveforms to use.
        seed : int, optional
                Seed to give to the parameter generation. Defaults to
                ``None``.
        include_time_shifts : bool
                Whether to apply learned merger-time-shift correction to
                the predicted waveforms. Defaults to ``False``.
        true_waveforms : FDWaveforms, optional
                True waveforms to compare to. Use this in order not to
                recompute the true waveforms each time when comparing
                different models; do not use the same waveforms for
                different models, since the downsampling indices may be
                different. Defaults to ``None``, which means the true
                waveforms are recomputed.
        zero_residuals : bool
                Whether to set the residuals to zero, meaning that the
                model is not used at all, instead just comparing the EOB
                waveforms to the PN baseline. Defaults to ``False``.
        save_params : bool
                Kept for API compatibility. Parameters are now stored
                per-model in the mismatch JSON (see ``mismatch_calc.py``)
                because each mode (e.g. ``model_21``) can have different
                parameter ranges, so this flag no longer writes
                ``parameter_set.json``.

        Returns
        -------
        List[float]
                List of mismatches.
        """
        # For modes (2,1) and (3,3), oversample parameters to compensate
        # for amp_teob <= 0 discards in :meth:`true_waveforms`.
        n_params = int(number_of_validation_waveforms * self._oversample_factor())
        self.parameter_set = self.param_set(n_params, seed)

        if true_waveforms is None:
            true_waveforms, valid_param_set = self.true_waveforms(self.parameter_set)
            # Use filtered params for the subsequent prediction step.
            self.parameter_set = valid_param_set
            phase0_eob = np.copy(true_waveforms.phases)
            true_waveforms.phases -= phase0_eob[:, 0].reshape(-1, 1)

        if zero_residuals:
            predicted_waveforms = self.post_newtonian_waveforms(self.parameter_set)
        else:
            predicted_waveforms = self.predicted_waveforms(self.parameter_set)

        if include_time_shifts:
            self._apply_predicted_time_shifts(predicted_waveforms, self.parameter_set)

        return self.mismatch_array(true_waveforms, predicted_waveforms)

    def full_waveform_validation_mismatches(
        self,
        number_of_validation_waveforms: int,
        seed: Optional[int] = None,
        include_time_shifts: bool = False,
        true_waveforms: Optional[FDWaveforms] = None,
        zero_residuals: bool = False,
    ) -> List[float]:
        """Validate the model using the full multi-mode waveform mismatch.

        This method uses :meth:`full_waveform_mismatch`, which
        marginalises over both time shift and reference phase, making it
        suitable for waveforms with higher modes.

        Parameters
        ----------
        number_of_validation_waveforms : int
                How many validation waveforms to use.
        seed : int, optional
                Seed to give to the parameter generation. Defaults to
                ``None``.
        include_time_shifts : bool
                Whether to apply learned merger-time-shift correction to
                the predicted waveforms. Defaults to ``False``.
        true_waveforms : FDWaveforms, optional
                True waveforms to compare to. Use this in order not to
                recompute the true waveforms each time when comparing
                different models; do not use the same waveforms for
                different models, since the downsampling indices may be
                different. Defaults to ``None``, which means the true
                waveforms are recomputed.
        zero_residuals : bool
                Whether to set the residuals to zero, meaning that the
                model is not used at all, instead just comparing the EOB
                waveforms to the PN baseline. Defaults to ``False``.

        Returns
        -------
        List[float]
                List of full waveform mismatches.
        """
        n_params = int(number_of_validation_waveforms * self._oversample_factor())
        self.parameter_set = self.param_set(n_params, seed)

        if true_waveforms is None:
            true_waveforms, valid_param_set = self.true_waveforms(self.parameter_set)
            self.parameter_set = valid_param_set
            phase0_eob = np.copy(true_waveforms.phases)
            true_waveforms.phases -= phase0_eob[:, 0].reshape(-1, 1)

        if zero_residuals:
            predicted_waveforms = self.post_newtonian_waveforms(self.parameter_set)
        else:
            predicted_waveforms = self.predicted_waveforms(self.parameter_set)

        if include_time_shifts:
            self._apply_predicted_time_shifts(predicted_waveforms, self.parameter_set)

        return self.full_waveform_mismatch_array(true_waveforms, predicted_waveforms)

    def mismatch_array(
        self,
        waveform_array_1: FDWaveforms,
        waveform_array_2: FDWaveforms,
        parameters_set: Optional[ParameterSet] = None,
    ) -> List[float]:
        """Compute the mismatches between each waveform pair in two arrays.

        Parameters
        ----------
        waveform_array_1 : FDWaveforms
                First set of waveforms to compare.
        waveform_array_2 : FDWaveforms
                Second set of waveforms to compare.
        parameters_set : ParameterSet, optional
                Unused; kept for API consistency with
                :meth:`full_waveform_mismatch_array`.

        Returns
        -------
        List[float]
                Mismatches between the waveforms, in order.
        """
        return self._mismatch_array_internal(
            waveform_array_1, waveform_array_2, disable_tqdm=False
        )

    def full_waveform_mismatch_array(
        self,
        waveform_array_1: FDWaveforms,
        waveform_array_2: FDWaveforms,
        parameters_set: Optional[ParameterSet] = None,
    ) -> List[float]:
        r"""Compute the full multi-mode mismatches between each waveform pair.

        Marginalises over both time shift and reference phase. This
        method is specifically designed for waveforms with higher modes
        where analytical phase marginalization is not possible.

        Parameters
        ----------
        waveform_array_1 : FDWaveforms
                First set of waveforms to compare.
        waveform_array_2 : FDWaveforms
                Second set of waveforms to compare.
        parameters_set : ParameterSet, optional
                Currently unused but kept for API consistency.

        Returns
        -------
        List[float]
                Full waveform mismatches between the waveforms, in order.
        """
        cartesian_1, cartesian_2 = self.waveforms(waveform_array_1, waveform_array_2)

        mode_key = (self.model.mode.l, self.model.mode.m)
        return [
            self.full_waveform_mismatch(
                modes_1={mode_key: c1},
                modes_2={mode_key: c2},
                frequencies=self.frequencies,
            )
            for c1, c2 in tqdm(
                zip(cartesian_1, cartesian_2), unit="full waveform mismatches"
            )
        ]

    def mismatch(
        self,
        waveform_1: np.ndarray,
        waveform_2: np.ndarray,
        frequencies: Optional[np.ndarray] = None,
        max_delta_t: float = 0.007,
    ) -> float:
        r"""Compute the mismatch between two Cartesian waveforms.

        The mismatch between waveforms :math:`a` and :math:`b` is defined
        as the minimum value of
        :math:`1 - (a|b) / \sqrt{(a|a)(b|b)}`,
        where the time shift of one of the two waveforms is changed
        arbitrarily, and where the product :math:`(a|b)` is the Wiener
        product.

        A custom implementation is used as opposed to the
        `pycbc one <https://pycbc.org/pycbc/latest/html/pycbc.filter.html?highlight=match#module-pycbc.filter.matchedfilter>`_
        since that one is not accurate enough, see
        `this issue <https://github.com/gwastro/pycbc/issues/3817>`_.

        The implementation uses scipy's
        `scalar minimizer <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html>`_
        to find the minimum of the mismatch.

        Parameters
        ----------
        waveform_1 : np.ndarray
                First Cartesian waveform to compare.
        waveform_2 : np.ndarray
                Second Cartesian waveform to compare.
        frequencies : np.ndarray, optional
                Frequencies at which the two waveforms are sampled, in
                Hz. If ``None`` (default), it is assumed that the
                waveforms are sampled at :attr:`frequencies` of this
                object.
        max_delta_t : float
                Maximum time shift for the two waveforms which are
                being compared, in seconds. Defaults to ``0.007``.

        Returns
        -------
        float
                The mismatch between the two waveforms.

        Raises
        ------
        ValueError
                If the underlying scipy optimization does not succeed.
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
            to_minimize,
            method="bounded",
            bounds=(-max_delta_t, max_delta_t),
            options={"maxiter": 1000, "xatol": 1e-12},
        )

        if not res.success:
            raise ValueError("Mismatch optimization did not succeed!")

        return 1 - (-res.fun) / norm

    def full_waveform_mismatch(
        self,
        modes_1: Dict[Tuple[int, int], np.ndarray],
        modes_2: Dict[Tuple[int, int], np.ndarray],
        frequencies: Optional[np.ndarray] = None,
        max_delta_t: float = 0.07,
        max_delta_phi: float = 2 * np.pi,
    ) -> float:
        r"""Mismatch between two multi-mode waveforms, with time/phase marginalisation.

        For full waveforms with higher modes, the mismatch cannot be
        computed analytically by marginalising over the reference phase,
        since different modes have different phase relationships. This
        function performs numerical marginalisation over both the time
        shift and the reference phase.

        The mismatch between waveforms :math:`a` and :math:`b` is
        defined as the minimum value of
        :math:`1 - (a|b) / \sqrt{(a|a)(b|b)}`,
        where the time shift and reference phase of one of the two
        waveforms are changed arbitrarily, and where the product
        :math:`(a|b)` is the Wiener product.

        Parameters
        ----------
        modes_1 : Dict[Tuple[int, int], np.ndarray]
                Per-mode contributions for the first waveform. Keys are
                :math:`(\ell, m)`, values are complex arrays already
                weighted by :math:`Y_{\ell m}(\iota, \varphi)`.
        modes_2 : Dict[Tuple[int, int], np.ndarray]
                Per-mode contributions for the second waveform. Same
                format as ``modes_1``.
        frequencies : np.ndarray, optional
                Frequencies at which the two waveforms are sampled, in
                Hz. If ``None`` (default), it is assumed that the
                waveforms are sampled at :attr:`frequencies` of this
                object.
        max_delta_t : float
                Maximum time shift for the two waveforms which are
                being compared, in seconds. Defaults to ``0.07``.
        max_delta_phi : float
                Maximum reference phase shift for the two waveforms
                which are being compared, in radians. Defaults to
                :math:`2\pi`.

        Returns
        -------
        float
                The mismatch between the two waveforms, marginalised
                over time and phase. Returns ``1.0`` as a fallback if
                the optimisation fails.
        """
        if frequencies is None:
            frequencies = self.frequencies
            psd_values = self.psd_values
        else:
            psd_values = self.psd_at_frequencies(frequencies)

        def product_complex(a: np.ndarray, b: np.ndarray) -> complex:
            """Wiener inner product (complex), preserves phase for overlap maximization."""
            return integrate.trapezoid(np.conj(a) * b / psd_values, x=frequencies)

        def product(a: np.ndarray, b: np.ndarray) -> float:
            """Wiener inner product magnitude (for norms)."""
            return abs(product_complex(a, b))

        waveform_1_summed = sum(modes_1.values())
        waveform_2_summed = sum(modes_2.values())
        norm = np.sqrt(
            product(waveform_1_summed, waveform_1_summed)
            * product(waveform_2_summed, waveform_2_summed)
        )

        def to_minimize(params: np.ndarray) -> float:
            """Negative overlap, as a function of ``[time_shift, phase_shift]``."""
            t_c, phi_c = params
            assert frequencies is not None

            # Each (l,m) mode transforms as e^{i*m*phi} under azimuthal
            # rotation; hence we apply e^{i*m*phi_c} per mode before
            # summing.
            waveform_2_shifted = sum(
                modes_2[(l, m)]
                * np.exp(2j * np.pi * frequencies * t_c + 1j * m * phi_c)
                for (l, m) in modes_2
            )
            overlap = product_complex(waveform_1_summed, waveform_2_shifted)
            return -abs(overlap)

        bounds = [(-max_delta_t, max_delta_t), (-max_delta_phi, max_delta_phi)]

        try:
            # Two-stage: coarse grid search followed by L-BFGS-B refinement.
            t_grid = np.linspace(-max_delta_t, max_delta_t, 20)
            phi_grid = np.linspace(-max_delta_phi, max_delta_phi, 12)
            best_val = np.inf
            best_params = np.array([0.0, 0.0])

            for t_c in t_grid:
                for phi_c in phi_grid:
                    val = to_minimize(np.array([t_c, phi_c]))
                    if val < best_val:
                        best_val = val
                        best_params = np.array([t_c, phi_c])

            res = minimize(
                to_minimize,
                best_params,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-8},
            )

            if not res.success:
                logging.warning(
                    "Full waveform mismatch optimization did not succeed: %s. "
                    "Returning fallback mismatch 1.0.",
                    res.message,
                )
                return 1.0

            return 1 - (-res.fun) / norm

        except Exception as e:
            logging.warning(
                "Full waveform mismatch computation failed (%s). "
                "Returning fallback mismatch 1.0.",
                e,
            )
            return 1.0

    def sky_maximized_mismatch(
        self,
        modes_1: Dict[Tuple[int, int], np.ndarray],
        modes_2: Dict[Tuple[int, int], np.ndarray],
        frequencies: Optional[np.ndarray] = None,
        max_delta_t: float = 0.07,
        max_delta_phi: float = 2 * np.pi,
    ) -> float:
        r"""Sky-maximised mismatch between two multi-mode waveforms.

        Extends :meth:`full_waveform_mismatch` by also maximising over
        the polarisation angle :math:`\kappa`, so that
        :math:`h_2 = \cos\kappa\, h_{2+} + \sin\kappa\, h_{2\times}`.

        Parameters
        ----------
        modes_1 : Dict[Tuple[int, int], np.ndarray]
                Per-mode contributions for the first waveform. Keys are
                :math:`(\ell, m)`, values are complex arrays
                (:math:`h_+ - i h_\times` for that mode).
        modes_2 : Dict[Tuple[int, int], np.ndarray]
                Per-mode contributions for the second waveform. Same
                format as ``modes_1``.
        frequencies : np.ndarray, optional
                Frequencies at which the two waveforms are sampled, in
                Hz. If ``None`` (default), it is assumed that the
                waveforms are sampled at :attr:`frequencies` of this
                object.
        max_delta_t : float
                Maximum time shift in seconds. Defaults to ``0.07``.
        max_delta_phi : float
                Maximum reference phase shift in radians. Defaults to
                :math:`2\pi`.

        Returns
        -------
        float
                The sky-maximised mismatch. Returns ``1.0`` as a
                fallback if the optimisation fails.
        """
        if frequencies is None:
            frequencies = self.frequencies
            psd_values = self.psd_values
        else:
            psd_values = self.psd_at_frequencies(frequencies)

        def product_complex(a: np.ndarray, b: np.ndarray) -> complex:
            """Wiener inner product (complex)."""
            return integrate.trapezoid(np.conj(a) * b / psd_values, x=frequencies)

        def build_polarizations(
            modes: Dict[Tuple[int, int], np.ndarray],
            phi_ref: float,
        ) -> Tuple[np.ndarray, np.ndarray]:
            r"""Build :math:`h_+` and :math:`h_\times` from modes at given ``phi_ref``.

            Convention: ``modes[(l,m)] = h_+ - i h_\times`` for that
            mode, so :math:`h_+ = \mathrm{Re}(h)` and
            :math:`h_\times = -\mathrm{Im}(h)`.
            """
            shifted = {
                (l, m): v * np.exp(1j * m * phi_ref)
                for (l, m), v in modes.items()
            }
            h_sum = sum(shifted.values())
            h_plus = np.real(h_sum)
            h_cross = -np.imag(h_sum)
            return h_plus, h_cross

        # Reference waveform fixed at kappa=0 (i.e. pure h_+).
        h1_plus, _h1_cross = build_polarizations(modes_1, phi_ref=0.0)
        h1 = h1_plus

        norm1 = abs(product_complex(h1, h1))

        def match_at_kappa_phi(kappa: float, phi_ref: float) -> float:
            """Max overlap over time shift, return normalized match."""
            h2_plus, h2_cross = build_polarizations(modes_2, phi_ref=phi_ref)
            h2 = np.cos(kappa) * h2_plus + np.sin(kappa) * h2_cross

            norm2 = abs(product_complex(h2, h2))
            if norm2 <= 0:
                return 0.0

            def neg_overlap(tau: float) -> float:
                phase = np.exp(2j * np.pi * frequencies * tau)
                return -abs(product_complex(h1, h2 * phase))

            res_t = minimize_scalar(
                neg_overlap,
                bounds=(-max_delta_t, max_delta_t),
                method="bounded",
                options={"maxiter": 500, "xatol": 1e-12},
            )
            overlap = -res_t.fun
            return overlap / np.sqrt(norm1 * norm2)

        def to_minimize(params: np.ndarray) -> float:
            kappa, phi_ref = params
            return -match_at_kappa_phi(float(kappa), float(phi_ref))

        try:
            res = dual_annealing(
                to_minimize,
                bounds=[(0, np.pi), (0, max_delta_phi)],
                maxiter=500,
                seed=None,
            )
            if not res.success:
                logging.warning(
                    "Sky-maximized mismatch optimization did not succeed: %s. "
                    "Returning fallback mismatch 1.0.",
                    res.message,
                )
                return 1.0
            return 1 - (-res.fun)
        except Exception as e:
            logging.warning(
                "Sky-maximized mismatch computation failed (%s). "
                "Returning fallback mismatch 1.0.",
                e,
            )
            return 1.0

    def _oversample_factor(self) -> float:
        """Oversampling factor for parameter generation.

        Modes :math:`(2,1)` and :math:`(3,3)` discard tuples for which
        ``amp_teob <= 0`` in :meth:`true_waveforms`, so we sample
        additional parameters to compensate.

        Returns
        -------
        float
                ``1.5`` for modes 21/33, ``1.0`` otherwise.
        """
        current_mode = self.model.dataset.current_mode
        if current_mode is None:
            return 1.0
        if (current_mode.l, current_mode.m) in ((2, 1), (3, 3)):
            return 1.5
        return 1.0

    def _apply_predicted_time_shifts(
        self,
        predicted_waveforms: FDWaveforms,
        param_set: ParameterSet,
    ) -> None:
        r"""Apply the learned merger-time-shift correction in place.

        Mutates ``predicted_waveforms.phases`` by adding
        :math:`2\pi (f - f_0)\, t_{\mathrm{shift}}` and then subtracting
        the per-waveform initial phase, so that the corrected phases
        start at zero.

        Parameters
        ----------
        predicted_waveforms : FDWaveforms
                Predicted waveforms whose ``phases`` field is mutated
                in place.
        param_set : ParameterSet
                Parameters at which to evaluate the time-shift
                predictor.
        """
        assert self.model.downsampling_indices is not None

        pred_phase_0 = np.copy(predicted_waveforms.phases)
        phase_freqs = self.model.dataset.frequencies_hz[
            self.model.downsampling_indices.phase_indices
        ]
        time_shifts = (
            self.time_shifts_predictor()
            .predict(param_set.parameter_array)
            .reshape(-1, 1)
        )

        predicted_waveforms.phases += (
            2 * np.pi * (phase_freqs - phase_freqs[0]) * time_shifts
            - pred_phase_0[:, 0].reshape(-1, 1)
        )

    def _mismatch_array_internal(
        self,
        waveform_array_1: FDWaveforms,
        waveform_array_2: FDWaveforms,
        disable_tqdm: bool = False,
    ) -> List[float]:
        """Compute pairwise mismatches, with an optional tqdm progress bar.

        Parameters
        ----------
        waveform_array_1 : FDWaveforms
                First set of waveforms.
        waveform_array_2 : FDWaveforms
                Second set of waveforms.
        disable_tqdm : bool
                Whether to disable the tqdm progress bar. Defaults to
                ``False``.

        Returns
        -------
        List[float]
                Mismatches between the waveforms, in order.
        """
        cartesian_1, cartesian_2 = self.waveforms(waveform_array_1, waveform_array_2)

        pairs = list(zip(cartesian_1, cartesian_2))
        iterator = tqdm(pairs, unit="mismatches", disable=disable_tqdm)
        return [self.mismatch(w1, w2) for w1, w2 in iterator]
