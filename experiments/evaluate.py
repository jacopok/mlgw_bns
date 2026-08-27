"""Score a :class:`~experiments.pipeline.Surrogate` by mismatch.

The metric is the one the package already uses: the PSD-weighted
mismatch of :class:`mlgw_bns.model_validation.ValidateModel`, minimised
over a time shift and (implicitly, through the modulus of the inner
product) over a reference phase. The validation waveforms are the EOB
ground truth cached by :mod:`experiments.cache`, so scoring a variant
costs no waveform generation at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from mlgw_bns.data_management import FDWaveforms, Residuals
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model_validation import ValidateModel
from mlgw_bns.neural_network import TimeshiftsNN
from mlgw_bns.resample_residuals import cartesian_waveforms_at_frequencies

from . import cache as cache_module
from .pipeline import Config, Surrogate


@dataclass
class Scores:
    """Mismatch summary for one variant."""

    label: str
    median: float
    percentile_90: float
    worst: float
    mismatches: np.ndarray

    def __str__(self) -> str:
        return (
            f"median {self.median:.3e}  p90 {self.percentile_90:.3e}  "
            f"worst {self.worst:.3e}   {self.label}"
        )


class Experiment:
    """Everything needed to train and score variants for one mode."""

    def __init__(self, mode: Mode, n_train: int, sampling: str = "uniform"):
        self.mode = mode
        self.n_train = n_train
        self.sampling = sampling
        self.cache = cache_module.load(mode, n_train, sampling)
        self.downsampling_indices = cache_module.downsampling_indices(self.cache)

        self.model = cache_module.make_mode_model(mode)
        self.model.downsampling_indices = self.downsampling_indices
        self.validator = ValidateModel(self.model)

        dataset = self.model.dataset
        self.amplitude_frequencies_hz = dataset.natural_units_to_hz(
            dataset.frequencies[self.downsampling_indices.amplitude_indices]
        )
        self.phase_frequencies_hz = dataset.natural_units_to_hz(
            dataset.frequencies[self.downsampling_indices.phase_indices]
        )
        self.psd = self.validator.psd_data

        # The production model removes the linear phase trend using a
        # predictor trained on the same residuals; fit it once here so
        # that every variant that asks for it gets the same one.
        self.timeshifts_predictor = self._fit_timeshifts_predictor()

        self.true_waveforms = self._true_waveforms()
        self.true_cartesian = self._to_cartesian(self.true_waveforms)

    # -- data views --------------------------------------------------------

    @property
    def train_parameters(self) -> np.ndarray:
        return self.cache["train_parameters"][: self.n_train]

    @property
    def train_amplitude_residuals(self) -> np.ndarray:
        return self.cache["train_amplitude_residuals"][: self.n_train]

    @property
    def train_phase_residuals(self) -> np.ndarray:
        return self.cache["train_phase_residuals"][: self.n_train]

    @property
    def validation_parameters(self) -> np.ndarray:
        return self.cache["validation_parameters"]

    def _fit_timeshifts_predictor(self) -> TimeshiftsNN:
        residuals = Residuals(
            np.array(self.cache["train_amplitude_residuals"]),
            np.array(self.cache["train_phase_residuals"]),
        )
        timeshifts = residuals.phase_timeshifts(frequencies=self.phase_frequencies_hz)
        return TimeshiftsNN(
            training_params=self.cache["train_parameters"],
            training_timeshifts=timeshifts,
        ).fit()

    def _true_waveforms(self) -> FDWaveforms:
        amplitudes = (
            self.cache["validation_amplitude_residuals"]
            * self.cache["validation_pn_amplitudes"]
        )
        phases = (
            self.cache["validation_phase_residuals"] + self.cache["validation_pn_phases"]
        )
        return FDWaveforms(amplitudes, phases - phases[:, :1])

    def _to_cartesian(self, waveforms: FDWaveforms) -> np.ndarray:
        dataset = self.model.dataset
        return cartesian_waveforms_at_frequencies(
            waveforms,
            dataset.hz_to_natural_units(self.validator.frequencies),
            dataset,
            self.model.downsampling_training,
            self.downsampling_indices,
        )

    # -- scoring -----------------------------------------------------------

    def mismatches(
        self, amplitude_residuals: np.ndarray, phase_residuals: np.ndarray
    ) -> np.ndarray:
        amplitudes = amplitude_residuals * self.cache["validation_pn_amplitudes"]
        phases = phase_residuals + self.cache["validation_pn_phases"]
        predicted = FDWaveforms(amplitudes, phases - phases[:, :1])
        predicted_cartesian = self._to_cartesian(predicted)

        # Serial: the per-waveform mismatch is a scalar minimisation over
        # a 3000-point quadrature, which is far cheaper than shipping the
        # validator (and with it a 5 Hz dataset and a TEOB generator) to
        # a pool of worker processes.
        return np.array(
            [
                self.validator.mismatch(true, prediction)
                for true, prediction in zip(self.true_cartesian, predicted_cartesian)
            ]
        )

    def make_surrogate(self, config: Config, fit_regressor: bool = True) -> Surrogate:
        surrogate = Surrogate(
            config,
            self.amplitude_frequencies_hz,
            self.phase_frequencies_hz,
            self.psd,
            self.timeshifts_predictor,
        )
        n_train = config.n_train or self.n_train
        surrogate.fit(
            self.train_parameters[:n_train],
            self.train_amplitude_residuals[:n_train],
            self.train_phase_residuals[:n_train],
            self.cache["train_pn_amplitudes"][:n_train].mean(axis=0),
            fit_regressor=fit_regressor,
            pn_amplitudes=self.cache["train_pn_amplitudes"][:n_train],
        )
        return surrogate

    def score(self, config: Config, projection_only: bool = False) -> Scores:
        surrogate = self.make_surrogate(config, fit_regressor=not projection_only)
        if projection_only:
            amplitude, phase = surrogate.project_residuals(
                self.validation_parameters,
                self.cache["validation_amplitude_residuals"],
                self.cache["validation_phase_residuals"],
                pn_amplitudes=self.cache["validation_pn_amplitudes"],
            )
        else:
            amplitude, phase = surrogate.predict_residuals(
                self.validation_parameters,
                pn_amplitudes=self.cache["validation_pn_amplitudes"],
            )

        values = self.mismatches(amplitude, phase)
        label = config.label() + (" [PCA floor]" if projection_only else "")
        return Scores(
            label=label,
            median=float(np.median(values)),
            percentile_90=float(np.percentile(values, 90)),
            worst=float(np.max(values)),
            mismatches=values,
        )
