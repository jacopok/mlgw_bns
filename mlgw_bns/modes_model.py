r"""Higher-order-mode surrogate model.

This module defines :class:`ModesModel`, a thin orchestrator that owns one
:class:`Model` instance per spherical-harmonic mode :math:`(\ell, m)` and
combines their predictions into the two observer-frame polarizations
:math:`h_+` and :math:`h_\times`.

The waveform from a quasi-circular binary is decomposed on the basis of
spin-weighted spherical harmonics :math:`{}_{-2}Y_{\ell m}(\iota, \varphi)`
as

.. math::
    h_+ - i\, h_\times = \sum_{\ell m} A_{\ell m}(f)\, e^{-i \phi_{\ell m}(f)}
                          \; {}_{-2}Y_{\ell m}(\iota, \varphi)\,,

see for instance Appendix E of `arXiv:2004.06503
<https://arxiv.org/pdf/2004.06503.pdf>`_. Each individual mode amplitude
and phase is reconstructed by a :class:`Model`, while the mode-relative
time shifts that align the mergers across modes are supplied by an
external predictor (``time_shifts_predictor``).

The module also exposes two summation kernels --- a Numba parallel kernel
and a NumPy ``einsum`` kernel --- that perform the per-frequency sum over
modes, weighted by the appropriate combinations of the spin-weighted
spherical harmonics.
"""

from __future__ import annotations

import copy
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import IO, Optional, Union

import numpy as np
import pkg_resources
from numba import njit, prange  # type: ignore

from .dataset_generation import Dataset
from .higher_order_modes import (
    Mode,
    ModeGeneratorFactory,
    teob_mode_generator_factory,
    _post_newtonian_amplitudes_by_mode,
    _post_newtonian_phases_by_mode,
)
from .model import Model, ParametersWithExtrinsic
from .neural_network import (
    Hyperparameters,
    TimeshiftsGPR,
    TimeshiftsNN,
    load_timeshifts_predictor,
)
from .special_func import spinsphericalharm

#: Subfolder, relative to the package, holding pretrained higher-order-mode models.
PRETRAINED_MODES_MODEL_FOLDER = "data/HOM/"

#: Names of the pretrained higher-order-mode models shipped with the package.
MODES_MODELS_AVAILABLE = ["pp_small_default_l2_m2", "pp_large_default_l2_m2"]

#: Default fallback locations for the time-shift predictor checkpoints.
_DEFAULT_TIMESHIFTS_NN_PATH = (
    "/scratch/shire/data/nj/personal/Prasoon/mlgw_bns_HOM/"
    "timeshifts_rff_surrogate.pkl"
)
_DEFAULT_TIMESHIFTS_GPR_PATH = (
    "/scratch/shire/data/nj/personal/Prasoon/mlgw_bns_HOM/"
    "timeshifts_model_HOM.pkl"
)


@njit(parallel=True, fastmath=True)
def _sum_modes_numba(
    amp: np.ndarray,
    cosphi: np.ndarray,
    sinphi: np.ndarray,
    coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Sum the per-mode contributions into the two polarizations.

    Each mode contributes a term of the form
    :math:`A_{\ell m}(f)\, [\cos\phi_{\ell m}(f)\, c_{\rm cos}
    + \sin\phi_{\ell m}(f)\, c_{\rm sin}]`
    to the real/imaginary parts of :math:`h_+` and :math:`h_\times`, where
    the coefficients :math:`c_{\rm cos}, c_{\rm sin}` are built from the
    spin-weighted spherical harmonics by :func:`_build_mode_coeffs`.

    Numba-compiled with ``parallel=True`` and ``fastmath=True`` to vectorize
    over frequency.

    Parameters
    ----------
    amp : np.ndarray
        Shape ``(n_modes, n_freq)``. Amplitude :math:`A_{\ell m}(f)`
        of each mode, evaluated at each frequency.
    cosphi : np.ndarray
        Shape ``(n_modes, n_freq)``. :math:`\cos\phi_{\ell m}(f)`.
    sinphi : np.ndarray
        Shape ``(n_modes, n_freq)``. :math:`\sin\phi_{\ell m}(f)`.
    coeffs : np.ndarray
        Shape ``(n_modes, 8)``. Spherical-harmonic-derived coefficients
        per mode, packed as
        ``[c_cos, c_sin]`` blocks for the four quantities
        ``h_plus_real, h_plus_imag, h_cross_real, h_cross_imag``
        (columns 0/1, 2/3, 4/5, 6/7 respectively).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(h_plus_real, h_plus_imag, h_cross_real, h_cross_imag)``,
        each of shape ``(n_freq,)``.
    """

    n_modes, n_freq = amp.shape
    h_plus_real = np.zeros(n_freq)
    h_plus_imag = np.zeros(n_freq)
    h_cross_real = np.zeros(n_freq)
    h_cross_imag = np.zeros(n_freq)

    for f in prange(n_freq):
        for m in range(n_modes):
            h_plus_real[f] += amp[m, f] * (
                cosphi[m, f] * coeffs[m, 0] + sinphi[m, f] * coeffs[m, 1]
            )
            h_plus_imag[f] += amp[m, f] * (
                cosphi[m, f] * coeffs[m, 2] + sinphi[m, f] * coeffs[m, 3]
            )
            h_cross_real[f] += amp[m, f] * (
                cosphi[m, f] * coeffs[m, 4] + sinphi[m, f] * coeffs[m, 5]
            )
            h_cross_imag[f] += amp[m, f] * (
                cosphi[m, f] * coeffs[m, 6] + sinphi[m, f] * coeffs[m, 7]
            )

    return h_plus_real, h_plus_imag, h_cross_real, h_cross_imag


def _sum_modes_einsum(
    amp: np.ndarray,
    cosphi: np.ndarray,
    sinphi: np.ndarray,
    coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure-NumPy counterpart of :func:`_sum_modes_numba`.

    Uses :func:`numpy.einsum` to perform the mode summation; this is
    typically as fast as the Numba kernel for moderate ``n_modes`` and
    avoids the one-shot JIT compilation cost.

    Parameters
    ----------
    amp, cosphi, sinphi, coeffs : np.ndarray
        See :func:`_sum_modes_numba`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(h_plus_real, h_plus_imag, h_cross_real, h_cross_imag)``,
        each of shape ``(n_freq,)``.
    """

    h_plus_real = (
        np.einsum('mf,mf,m->f', amp, cosphi, coeffs[:, 0])
        + np.einsum('mf,mf,m->f', amp, sinphi, coeffs[:, 1])
    )
    h_plus_imag = (
        np.einsum('mf,mf,m->f', amp, cosphi, coeffs[:, 2])
        + np.einsum('mf,mf,m->f', amp, sinphi, coeffs[:, 3])
    )
    h_cross_real = (
        np.einsum('mf,mf,m->f', amp, cosphi, coeffs[:, 4])
        + np.einsum('mf,mf,m->f', amp, sinphi, coeffs[:, 5])
    )
    h_cross_imag = (
        np.einsum('mf,mf,m->f', amp, cosphi, coeffs[:, 6])
        + np.einsum('mf,mf,m->f', amp, sinphi, coeffs[:, 7])
    )
    return h_plus_real, h_plus_imag, h_cross_real, h_cross_imag


def _build_mode_coeffs(
    modes: list[Mode],
    mode_indices: list[int],
    Ylm_real: dict[Mode, float],
    Ylm_imag: dict[Mode, float],
    Ylm_real_mneg: dict[Mode, float],
    Ylm_imag_mneg: dict[Mode, float],
) -> np.ndarray:
    r"""Build the spherical-harmonic coefficients required by the mode sum.

    For each requested mode :math:`(\ell, m)`, this function combines
    :math:`{}_{-2}Y_{\ell m}` and :math:`{}_{-2}Y_{\ell,-m}` according to
    the standard symmetry relation between positive and negative
    azimuthal modes (whose sign depends on the parity of :math:`\ell`),
    so that the contribution to :math:`h_+` and :math:`h_\times` can be
    written as a real linear combination of
    :math:`\cos\phi_{\ell m}` and :math:`\sin\phi_{\ell m}`.

    Parameters
    ----------
    modes : list[Mode]
        Full list of modes managed by the surrogate.
    mode_indices : list[int]
        Indices, into ``modes``, of the modes for which a coefficient row
        should be produced (``len(mode_indices) == n``).
    Ylm_real, Ylm_imag : dict[Mode, float]
        Real and imaginary parts of :math:`{}_{-2}Y_{\ell m}(\iota,\varphi)`
        for each mode in ``modes``.
    Ylm_real_mneg, Ylm_imag_mneg : dict[Mode, float]
        Same, but for :math:`{}_{-2}Y_{\ell,-m}(\iota,\varphi)`, keyed by
        the mode obtained via :meth:`Mode.opposite`.

    Returns
    -------
    np.ndarray
        Array of shape ``(n, 8)`` to be passed to the summation kernels.
        See :func:`_sum_modes_numba` for the column ordering.
    """

    n = len(mode_indices)
    coeffs = np.zeros((n, 8))
    for i, mode_idx in enumerate(mode_indices):
        mode = modes[mode_idx]
        mode_opp = mode.opposite()
        yr, yi = Ylm_real[mode], Ylm_imag[mode]
        yr_m, yi_m = Ylm_real_mneg[mode_opp], Ylm_imag_mneg[mode_opp]
        if mode.l % 2:
            # Odd l: Y_{l,-m} = -conj(Y_{l,m}) under the symmetry assumed here.
            coeffs[i, 0] = yr - yr_m
            coeffs[i, 1] = -(yi + yi_m)
            coeffs[i, 2] = yi + yi_m
            coeffs[i, 3] = yr - yr_m
            coeffs[i, 4] = -(yi - yi_m)
            coeffs[i, 5] = -(yr + yr_m)
            coeffs[i, 6] = yr + yr_m
            coeffs[i, 7] = -(yi - yi_m)
        else:
            # Even l: Y_{l,-m} = +conj(Y_{l,m}).
            coeffs[i, 0] = yr + yr_m
            coeffs[i, 1] = -(yi - yi_m)
            coeffs[i, 2] = yi - yi_m
            coeffs[i, 3] = yr + yr_m
            coeffs[i, 4] = -(yi + yi_m)
            coeffs[i, 5] = -(yr - yr_m)
            coeffs[i, 6] = yr - yr_m
            coeffs[i, 7] = -(yi + yi_m)
    return coeffs


class _LazyModelsDict(dict):
    """Dictionary that materializes :class:`Model` instances on demand.

    The per-mode :class:`Model` objects can be expensive to construct
    (they instantiate a full :class:`Dataset` with its waveform generator),
    so they are built lazily the first time the user accesses
    ``modes_model.models[mode]`` and cached for subsequent accesses.

    Parameters
    ----------
    modes_model : ModesModel
        Owning :class:`ModesModel`, used to look up the per-mode filename,
        generator factory and constructor keyword arguments.
    """

    def __init__(self, modes_model: "ModesModel"):
        super().__init__()
        self._modes_model = modes_model

    def __missing__(self, mode: Mode) -> Model:
        if mode not in self._modes_model.modes:
            raise KeyError(f"Mode {mode} not in {self._modes_model.modes}")
        model = Model(
            mode=mode,
            filename=self._modes_model.mode_filename(mode),
            waveform_generator=self._modes_model._generator_factory(mode),
            **self._modes_model._model_kwargs,
        )
        self[mode] = model
        return model


class ModesModel:
    r"""Higher-order-modes surrogate.

    A :class:`ModesModel` orchestrates one :class:`Model` per spherical
    harmonic mode :math:`(\ell, m)` and combines their predictions to
    produce the observer-frame polarizations :math:`h_+, h_\times` via

    .. math::
        h_+ - i\, h_\times = \sum_{\ell m} A_{\ell m}\, e^{-i\phi_{\ell m}}
                              \; {}_{-2}Y_{\ell m}(\iota, \varphi).

    Each per-mode model is instantiated lazily on first access through
    :attr:`models`, which means that constructing a :class:`ModesModel`
    is cheap until predictions are actually requested.

    Parameters
    ----------
    modes : list[Mode]
        Modes to include in the surrogate. Must be non-empty.
    generator_factory : ModeGeneratorFactory, optional
        Callable that, given a mode, returns the appropriate
        :class:`~mlgw_bns.higher_order_modes.ModeGenerator` to use during
        training. Defaults to :func:`teob_mode_generator_factory`.
    time_shifts_predictor : TimeshiftsGPR or TimeshiftsNN, optional
        Predictor for the mode-relative time shifts used to align the
        mergers of different modes. If ``None``, the constructor tries to
        load a default :class:`TimeshiftsNN` checkpoint, falling back to a
        :class:`TimeshiftsGPR` checkpoint, and finally storing ``None`` if
        neither is available (in which case the user must supply
        ``time_shifts`` explicitly to :meth:`predict`).
    **model_kwargs
        Extra keyword arguments forwarded to each :class:`Model`. The
        special key ``filename`` is consumed here and used as the *base*
        filename; each per-mode model file is named
        ``"{base_filename}_l{l}_m{m}"``.

    Attributes
    ----------
    modes : list[Mode]
        Modes included in this model.
    models : dict[Mode, Model]
        Lazy mapping ``mode -> Model``. The :class:`Model` instance is
        created on first access.
    base_filename : str
        Base name used to derive each per-mode filename.
    time_shifts_predictor : TimeshiftsGPR or TimeshiftsNN or None
        Predictor used to compute the per-mode time shifts, when not
        supplied externally.

    Raises
    ------
    ValueError
        If ``modes`` is empty.

    References
    ----------
    See Appendix E of `arXiv:2004.06503
    <https://arxiv.org/pdf/2004.06503.pdf>`_ for the mode decomposition.
    """

    def __init__(
        self,
        modes: list[Mode],
        generator_factory: ModeGeneratorFactory = teob_mode_generator_factory,
        time_shifts_predictor: Optional[Union[TimeshiftsGPR, TimeshiftsNN]] = None,
        **model_kwargs,
    ):
        if not modes:
            raise ValueError("At least one mode must be provided")

        self.modes = modes
        self._base_filename = model_kwargs.pop("filename", "")

        if time_shifts_predictor is None:
            self.time_shifts_predictor = self._load_default_time_shifts_predictor()
        else:
            self.time_shifts_predictor = time_shifts_predictor

        # Stored for lazy construction of the per-mode `Model` objects.
        self._generator_factory = generator_factory
        self._model_kwargs = model_kwargs

        self.models: dict[Mode, Model] = _LazyModelsDict(self)

    @staticmethod
    def _load_default_time_shifts_predictor() -> Optional[Union[TimeshiftsNN, TimeshiftsGPR]]:
        """Try to load the default time-shift predictor.

        Attempts the lightweight :class:`TimeshiftsNN` (RFF+Ridge) first,
        falling back to :class:`TimeshiftsGPR`. Returns ``None`` if neither
        can be loaded.
        """
        try:
            return load_timeshifts_predictor(
                _DEFAULT_TIMESHIFTS_NN_PATH,
                _DEFAULT_TIMESHIFTS_GPR_PATH,
            )
        except ValueError as e:
            logging.warning(
                "Could not load default time-shift predictor (%s). "
                "`time_shifts` must be provided explicitly to `predict`.",
                e,
            )
            return None

    def mode_filename(self, mode: Mode) -> str:
        """Return the on-disk filename for a single mode.

        Parameters
        ----------
        mode : Mode
            Mode whose filename should be returned.

        Returns
        -------
        str
            Filename of the form ``"{base_filename}_l{l}_m{m}"``.
        """
        return f"{self.base_filename}_l{mode.l}_m{mode.m}"

    @property
    def base_filename(self) -> str:
        """Base filename used to derive each per-mode model filename."""
        return self._base_filename

    @base_filename.setter
    def base_filename(self, value: str) -> None:
        """Set the base filename and propagate it to already-built mode models."""
        self._base_filename = value
        for mode in self.modes:
            # Only update modes whose Model has already been materialized,
            # to avoid eagerly building all of them through __missing__.
            if mode in self.models:
                self.models[mode].filename = self.mode_filename(mode)

    @property
    def dataset(self) -> Dataset:
        """Dataset of the first mode model.

        All per-mode models share the same dataset configuration, so the
        first one is returned for convenience (e.g. for accessing the
        frequency grid or the reference total mass).

        Raises
        ------
        ValueError
            If this :class:`ModesModel` was somehow built with no modes.
        """
        if not self.modes:
            raise ValueError("No models available")
        return self.models[self.modes[0]].dataset

    @property
    def auxiliary_data_available(self) -> bool:
        """``True`` iff every per-mode model has PCA + downsampling data loaded."""
        return all(self.models[mode].auxiliary_data_available for mode in self.modes)

    @property
    def nn_available(self) -> bool:
        """``True`` iff every per-mode model has a trained neural network loaded."""
        return all(self.models[mode].nn_available for mode in self.modes)

    @property
    def training_dataset_available(self) -> bool:
        """``True`` iff every per-mode model has its training dataset available."""
        return all(self.models[mode].training_dataset_available for mode in self.modes)

    def __str__(self) -> str:
        n_modes = len(self.modes)
        modes_str = ", ".join(f"({m.l},{m.m})" for m in self.modes)

        return (
            "ModesModel("
            f"modes=[{modes_str}], "
            f"n_modes={n_modes}, "
            f"base_filename={self.base_filename}, "
            f"auxiliary_data_available={self.auxiliary_data_available}, "
            f"nn_available={self.nn_available}, "
            f"training_dataset_available={self.training_dataset_available})"
        )

    @classmethod
    def default_for_testing(
        cls,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> "ModesModel":
        """Load a pretrained :class:`ModesModel` shipped with the package.

        Convenience constructor that mirrors :meth:`Model.default_for_testing`
        for the multi-mode case. The arrays/metadata/nn streams of every
        mode are loaded from the package resources (``data/HOM/``).

        Parameters
        ----------
        model_name : str, optional
            Name of the model to load. Must be one of
            :data:`MODES_MODELS_AVAILABLE`. Defaults to the first entry.
        **kwargs
            Extra keyword arguments forwarded to the :class:`ModesModel`
            constructor. The reserved keys ``filename`` and ``modes`` are
            consumed here:

            * ``filename`` overrides the base filename after loading, so
              that subsequent saves write to a user-chosen location.
            * ``modes`` overrides the default list of modes (which is
              ``[(2,2), (2,1), (3,3), (4,4)]``).

        Returns
        -------
        ModesModel
            Loaded :class:`ModesModel` instance.

        Raises
        ------
        ValueError
            If ``model_name`` is not in :data:`MODES_MODELS_AVAILABLE`.
        """
        if model_name is None:
            model_name = MODES_MODELS_AVAILABLE[0]

        if model_name not in MODES_MODELS_AVAILABLE:
            raise ValueError(f"Model {model_name} not available!")

        given_filename = kwargs.pop("filename", None)

        modes = kwargs.pop("modes", None)
        if modes is None:
            # Default set of modes loaded by the pretrained checkpoints.
            modes = [Mode(2, 2), Mode(2, 1), Mode(3, 3), Mode(4, 4)]

        model = cls(
            modes=modes,
            filename=PRETRAINED_MODES_MODEL_FOLDER + model_name,
            **kwargs,
        )

        for mode in model.modes:
            mode_model = model.models[mode]
            try:
                stream_meta = pkg_resources.resource_stream(
                    __name__, mode_model.filename_metadata
                )
                stream_arrays = pkg_resources.resource_stream(
                    __name__, mode_model.filename_arrays
                )
                stream_nn = pkg_resources.resource_stream(
                    __name__, mode_model.filename_nn
                )
                mode_model.load(streams=(stream_meta, stream_arrays, stream_nn))
            except Exception as e:  # noqa: BLE001
                logging.warning("Could not load model for mode %s: %s", mode, e)

        if given_filename is not None:
            model.base_filename = given_filename

        return model

    def generate(
        self,
        training_downsampling_dataset_size: Optional[int] = 64,
        training_pca_dataset_size: Optional[int] = 256,
        training_nn_dataset_size: Optional[int] = 256,
    ) -> None:
        """Run :meth:`Model.generate` for every mode.

        Builds the downsampling indices, PCA data and training residuals
        for each per-mode :class:`Model`. The three dataset sizes have the
        same meaning as in :meth:`Model.generate`; setting one of them to
        ``None`` reuses pre-existing data for that step.

        Parameters
        ----------
        training_downsampling_dataset_size : int, optional
            Size of the dataset used to fit the downsampling indices.
            Defaults to 64.
        training_pca_dataset_size : int, optional
            Size of the dataset used to fit the PCA components.
            Defaults to 256.
        training_nn_dataset_size : int, optional
            Size of the dataset used to train the neural network on the
            PCA residuals. Defaults to 256.
        """
        for mode in self.modes:
            self.models[mode].generate(
                training_downsampling_dataset_size=training_downsampling_dataset_size,
                training_pca_dataset_size=training_pca_dataset_size,
                training_nn_dataset_size=training_nn_dataset_size,
            )

    def set_hyper_and_train_nn(
        self,
        hyper: Optional[Hyperparameters] = None,
        idxs: Union[list[int], slice] = slice(None),
    ) -> None:
        """Train the neural network of each per-mode model.

        Parameters
        ----------
        hyper : Hyperparameters, optional
            Hyperparameters for the neural network. If ``None``, every
            per-mode model uses its own defaults.
        idxs : list[int] or slice, optional
            Selection over the training dataset, forwarded to
            :meth:`Model.set_hyper_and_train_nn`. Defaults to all data.
        """
        for mode in self.modes:
            self.models[mode].set_hyper_and_train_nn(hyper=hyper, idxs=idxs)

    def save(self, include_training_data: bool = True) -> None:
        """Save every per-mode model to disk.

        Because the files of different modes are independent, the writes
        are dispatched to a thread pool when there is more than one mode,
        which is typically faster on slow filesystems.

        Parameters
        ----------
        include_training_data : bool, optional
            Whether to also persist the per-mode training residuals and
            parameters. Defaults to ``True``.
        """
        if len(self.modes) > 1:
            with ThreadPoolExecutor(max_workers=len(self.modes)) as executor:
                # Drain the iterator so that any exceptions are propagated.
                list(
                    executor.map(
                        lambda m: self.models[m].save(
                            include_training_data=include_training_data
                        ),
                        self.modes,
                    )
                )
        else:
            for mode in self.modes:
                self.models[mode].save(include_training_data=include_training_data)

    def load(
        self,
        streams: Optional[tuple[IO[bytes], IO[bytes], IO[bytes]]] = None,
    ) -> None:
        """Load every per-mode model from disk.

        Parameters
        ----------
        streams : tuple[IO[bytes], IO[bytes], IO[bytes]], optional
            Pre-opened streams ``(metadata, arrays, nn)`` to load from,
            forwarded as-is to every per-mode :meth:`Model.load`. When
            ``None`` (the default), each model opens its own files from
            the path implied by :meth:`mode_filename`.
        """
        for mode in self.modes:
            self.models[mode].load(streams=streams)

    def predict_amplitude_phase_mode(
        self,
        mode: Mode,
        frequencies: np.ndarray,
        params: ParametersWithExtrinsic,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the amplitude and phase of a single mode.

        Parameters
        ----------
        mode : Mode
            Mode to predict. Must be in :attr:`modes`.
        frequencies : np.ndarray
            Frequencies at which to evaluate the mode, in Hz.
        params : ParametersWithExtrinsic
            Parameters of the source.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(amplitude, phase)`` arrays for the requested mode.

        Raises
        ------
        ValueError
            If ``mode`` is not among :attr:`modes`.
        """
        if mode not in self.models:
            raise ValueError(f"Mode {mode} is not included in this model")

        return self.models[mode].predict_amplitude_phase_optimized(frequencies, params)

    def predict(
        self,
        frequencies: np.ndarray,
        params: ParametersWithExtrinsic,
        time_shifts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Predict the full frequency-domain waveform from all modes.

        Combines the predictions of every per-mode :class:`Model` into the
        two observer-frame polarizations :math:`h_+, h_\times`, using the
        provided mode-relative time shifts and the inclination contained
        in ``params``.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies at which to evaluate the waveform, in Hz.
        params : ParametersWithExtrinsic
            Source parameters (intrinsic + extrinsic).
        time_shifts : np.ndarray
            Time shifts, one per mode, that align the per-mode mergers
            in the time domain. Typically produced by
            :attr:`time_shifts_predictor`. A scalar is broadcast to every
            mode.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(h, h_plus, h_cross)``, where ``h = h_plus - i * h_cross``.

        References
        ----------
        See Appendix E of `arXiv:2004.06503
        <https://arxiv.org/pdf/2004.06503.pdf>`_.
        """
        return self._compute_polarizations_from_modes(
            frequencies=frequencies,
            params=params,
            time_shifts=time_shifts,
            inclination=params.inclination,
        )

    def _compute_polarizations_from_modes(
        self,
        frequencies: np.ndarray,
        params: ParametersWithExtrinsic,
        time_shifts: np.ndarray,
        inclination: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Internal driver for :meth:`predict`.

        Computes the four Cartesian components ``(h_+,r), (h_+,i),
        (h_x,r), (h_x,i)`` via :meth:`_hpc_waveform`, then assembles the
        complex polarizations and applies the conventional
        :math:`1/(2\eta)` normalization.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies at which to evaluate the waveform, in Hz.
        params : ParametersWithExtrinsic
            Source parameters.
        time_shifts : np.ndarray
            Per-mode time shifts.
        inclination : float
            Inclination angle, in radians.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(h, h_plus, h_cross)``.
        """
        h_plus_real, h_plus_imag, h_cross_real, h_cross_imag = self._hpc_waveform(
            frequencies=frequencies,
            params=params,
            time_shifts=time_shifts,
            inclination=inclination,
            use_pn=False,
        )

        eta = params.intrinsic(self.dataset).eta

        hp_pred = (h_plus_real + 1j * h_plus_imag) / eta / 2
        hc_pred = (h_cross_real + 1j * h_cross_imag) / eta / 2
        h_pred = hp_pred - 1j * hc_pred

        return h_pred, hp_pred, hc_pred

    def predict_modes_dict(
        self,
        frequencies: np.ndarray,
        params: ParametersWithExtrinsic,
        time_shifts: np.ndarray,
    ) -> dict[tuple[int, int], np.ndarray]:
        r"""Return the per-mode complex Cartesian contributions.

        Each entry is the contribution of the corresponding mode to the
        observer-frame combination :math:`h_+ - i\, h_\times`, already
        weighted by :math:`{}_{-2}Y_{\ell m}(\iota, \varphi)`. Summing the
        returned arrays reproduces the output of :meth:`predict`.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies at which to evaluate the waveform, in Hz.
        params : ParametersWithExtrinsic
            Source parameters.
        time_shifts : np.ndarray
            Per-mode time shifts (see :meth:`predict`).

        Returns
        -------
        dict[tuple[int, int], np.ndarray]
            Mapping ``(l, m) -> h_lm = h_+ - i h_x`` (one complex array
            per mode).
        """
        modes_dict = self._hpc_waveform_per_mode(
            frequencies=frequencies,
            params=params,
            time_shifts=time_shifts,
            inclination=params.inclination,
            use_pn=False,
        )
        eta = params.intrinsic(self.dataset).eta
        result: dict[tuple[int, int], np.ndarray] = {}
        for (l, m), (hp_real, hp_imag, hc_real, hc_imag) in modes_dict.items():
            hp = (hp_real + 1j * hp_imag) / eta / 2
            hc = (hc_real + 1j * hc_imag) / eta / 2
            result[(l, m)] = hp - 1j * hc
        return result

    def _hpc_waveform_per_mode(
        self,
        frequencies: np.ndarray,
        params: ParametersWithExtrinsic,
        time_shifts: np.ndarray,
        inclination: float,
        use_pn: bool,
    ) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        r"""Per-mode Cartesian components of :math:`h_+` and :math:`h_\times`.

        Same machinery as :meth:`_hpc_waveform`, but instead of summing
        over the modes it returns one tuple of components per mode.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies at which to evaluate, in Hz.
        params : ParametersWithExtrinsic
            Source parameters.
        time_shifts : np.ndarray
            Per-mode time shifts.
        inclination : float
            Inclination angle, in radians.
        use_pn : bool
            If ``True``, take the per-mode amplitude/phase from the
            Post-Newtonian (TaylorF2-style) expressions in
            :mod:`~mlgw_bns.pn_modes` instead of from the trained
            :class:`Model` instances. Used by :meth:`get_taylorf2_modes_dict`.

        Returns
        -------
        dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
            Mapping ``(l, m) -> (h_plus_real, h_plus_imag,
            h_cross_real, h_cross_imag)`` for every mode in :attr:`modes`.
        """
        assert use_pn is not None
        Ylm_real, Ylm_imag, Ylm_real_mneg, Ylm_imag_mneg = self._compute_Ylm_modes(
            modes=self.modes,
            phi=0.0,
            iota=inclination,
        )

        active_indices: list[int] = []
        amps_list: list[np.ndarray] = []
        phases_list: list[np.ndarray] = []
        dataset = self.dataset

        for idx, mode in enumerate(self.modes):
            if use_pn:
                parameters_intrinsic = params.intrinsic(dataset)
                amp = _post_newtonian_amplitudes_by_mode[mode](
                    parameters_intrinsic,
                    frequencies * params.mass_sum_seconds,
                )
                phase = _post_newtonian_phases_by_mode[mode](
                    parameters_intrinsic,
                    frequencies * params.mass_sum_seconds,
                )
            else:
                amp, phase = self.models[mode].predict_amplitude_phase_optimized(
                    frequencies, params
                )
                ts = time_shifts if np.isscalar(time_shifts) else time_shifts[idx]
                # Time shifts are stored in units of the reference total mass
                # of the dataset, so we rescale to the requested total mass.
                ts_scaled = ts * (params.total_mass / self.dataset.total_mass)
                phase += 2 * np.pi * frequencies * ts_scaled
            active_indices.append(idx)
            amps_list.append(amp)
            phases_list.append(phase)

        if not active_indices:
            return {}

        amp_arr = np.stack(amps_list)
        cosphi_arr = np.cos(np.stack(phases_list))
        sinphi_arr = np.sin(np.stack(phases_list))
        coeffs = _build_mode_coeffs(
            self.modes,
            active_indices,
            Ylm_real,
            Ylm_imag,
            Ylm_real_mneg,
            Ylm_imag_mneg,
        )
        result: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for i, mode in enumerate(self.modes):
            c = coeffs[i]
            h_plus_real = amp_arr[i] * (cosphi_arr[i] * c[0] + sinphi_arr[i] * c[1])
            h_plus_imag = amp_arr[i] * (cosphi_arr[i] * c[2] + sinphi_arr[i] * c[3])
            h_cross_real = amp_arr[i] * (cosphi_arr[i] * c[4] + sinphi_arr[i] * c[5])
            h_cross_imag = amp_arr[i] * (cosphi_arr[i] * c[6] + sinphi_arr[i] * c[7])
            result[(mode.l, mode.m)] = (h_plus_real, h_plus_imag, h_cross_real, h_cross_imag)
        return result

    def get_taylorf2_modes_dict(
        self,
        frequencies: np.ndarray,
        params: ParametersWithExtrinsic,
        inclination: Optional[float] = None,
    ) -> dict[tuple[int, int], np.ndarray]:
        r"""Per-mode complex contributions using TaylorF2 (post-Newtonian).

        Same output format as :meth:`predict_modes_dict`, but the
        amplitude and phase of every mode are taken from the
        Post-Newtonian (TaylorF2-style) expressions in
        :mod:`~mlgw_bns.pn_modes`. No time shifts are applied
        (``time_shifts=0``), since the PN expressions are already aligned
        across modes.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies at which to evaluate the waveform, in Hz.
        params : ParametersWithExtrinsic
            Source parameters.
        inclination : float, optional
            Inclination angle, in radians. Defaults to ``params.inclination``.

        Returns
        -------
        dict[tuple[int, int], np.ndarray]
            Mapping ``(l, m) -> h_lm = h_+ - i h_x``.
        """
        if inclination is None:
            inclination = params.inclination

        dataset = self.dataset
        modes_dict = self._hpc_waveform_per_mode(
            frequencies=frequencies,
            params=params,
            time_shifts=0.0,
            inclination=inclination,
            use_pn=True,
        )
        eta = params.intrinsic(dataset).eta
        result: dict[tuple[int, int], np.ndarray] = {}
        for (l, m), (hp_real, hp_imag, hc_real, hc_imag) in modes_dict.items():
            hp = (hp_real + 1j * hp_imag) / eta / 2
            hc = (hc_real + 1j * hc_imag) / eta / 2
            result[(l, m)] = hp - 1j * hc
        return result

    def get_teob_modes_dict(
        self,
        frequencies: np.ndarray,
        params: ParametersWithExtrinsic,
        inclination: Optional[float] = None,
    ) -> dict[tuple[int, int], np.ndarray]:
        r"""Per-mode complex contributions from the underlying EOB code.

        Calls each mode's underlying TEOBResumS-based waveform generator
        directly (via :meth:`get_amplitude_phase_at_inclination`) rather
        than going through the surrogate's neural network, then assembles
        the observer-frame combination :math:`h_+ - i\, h_\times`. Useful
        as a ground-truth reference when validating the surrogate.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies at which to evaluate the waveform, in Hz.
        params : ParametersWithExtrinsic
            Source parameters.
        inclination : float, optional
            Inclination angle, in radians. Defaults to ``params.inclination``.

        Returns
        -------
        dict[tuple[int, int], np.ndarray]
            Mapping ``(l, m) -> h_lm = h_+ - i h_x``.
        """
        if inclination is None:
            inclination = params.inclination

        dataset = self.dataset
        # Use a shallow copy of the dataset whose total_mass is set to the
        # requested total mass, so that the EOB generator interprets the
        # natural-unit frequencies consistently.
        dataset_for_teob = copy.copy(dataset)
        dataset_for_teob.total_mass = params.total_mass
        params_teob = params.intrinsic(dataset_for_teob)

        f_natural = frequencies * params.mass_sum_seconds
        Ylm_real, Ylm_imag, Ylm_real_mneg, Ylm_imag_mneg = self._compute_Ylm_modes(
            modes=self.modes,
            phi=0.0,
            iota=inclination,
        )

        amps_list: list[np.ndarray] = []
        phases_list: list[np.ndarray] = []
        for mode in self.modes:
            _f_spa, amp, phase = self.models[mode].waveform_generator.get_amplitude_phase_at_inclination(
                params_teob, f_natural, inclination=inclination
            )
            amps_list.append(amp)
            phases_list.append(phase)

        amp_arr = np.stack(amps_list)
        cosphi_arr = np.cos(np.stack(phases_list))
        sinphi_arr = np.sin(np.stack(phases_list))
        coeffs = _build_mode_coeffs(
            self.modes,
            list(range(len(self.modes))),
            Ylm_real,
            Ylm_imag,
            Ylm_real_mneg,
            Ylm_imag_mneg,
        )

        eta = params.intrinsic(dataset).eta
        result: dict[tuple[int, int], np.ndarray] = {}
        for i, mode in enumerate(self.modes):
            c = coeffs[i]
            h_plus_real = amp_arr[i] * (cosphi_arr[i] * c[0] + sinphi_arr[i] * c[1])
            h_plus_imag = amp_arr[i] * (cosphi_arr[i] * c[2] + sinphi_arr[i] * c[3])
            h_cross_real = amp_arr[i] * (cosphi_arr[i] * c[4] + sinphi_arr[i] * c[5])
            h_cross_imag = amp_arr[i] * (cosphi_arr[i] * c[6] + sinphi_arr[i] * c[7])
            hp = (h_plus_real + 1j * h_plus_imag) / eta / 2
            hc = (h_cross_real + 1j * h_cross_imag) / eta / 2
            result[(mode.l, mode.m)] = hp - 1j * hc
        return result

    def _hpc_waveform(
        self,
        frequencies: np.ndarray,
        params: ParametersWithExtrinsic,
        time_shifts: np.ndarray,
        inclination: float,
        use_pn: Optional[bool] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""Cartesian components of :math:`h_+` and :math:`h_\times` summed over modes.

        Collects the amplitude and phase of every mode (either from the
        trained surrogate or from the PN expressions, depending on
        ``use_pn``), packs them into ``(n_modes, n_freq)`` arrays,
        and delegates the actual mode sum to :func:`_sum_modes_einsum`.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies at which to evaluate the waveform, in Hz.
        params : ParametersWithExtrinsic
            Source parameters.
        time_shifts : np.ndarray or float
            Per-mode time shifts, in seconds, in the reference total-mass
            units. A scalar value is broadcast to every mode. Ignored
            when ``use_pn`` is ``True``.
        inclination : float
            Inclination angle, in radians.
        use_pn : bool
            ``True`` for the PN per-mode expressions, ``False`` for the
            surrogate. Must be explicitly provided; passing ``None``
            triggers an assertion error.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            ``(h_plus_real, h_plus_imag, h_cross_real, h_cross_imag)``,
            each of shape ``(n_freq,)``.
        """
        assert use_pn is not None, "use_pn must be provided"

        Ylm_real, Ylm_imag, Ylm_real_mneg, Ylm_imag_mneg = self._compute_Ylm_modes(
            modes=self.modes,
            phi=0.0,
            iota=inclination,
        )

        active_indices: list[int] = []
        amps_list: list[np.ndarray] = []
        phases_list: list[np.ndarray] = []

        dataset = self.dataset
        for idx, mode in enumerate(self.modes):
            if use_pn:
                parameters_intrinsic = params.intrinsic(dataset)
                amp = _post_newtonian_amplitudes_by_mode[mode](
                    parameters_intrinsic,
                    frequencies * params.mass_sum_seconds,
                )
                phase = _post_newtonian_phases_by_mode[mode](
                    parameters_intrinsic,
                    frequencies * params.mass_sum_seconds,
                )
            else:
                amp, phase = self.models[mode].predict_amplitude_phase_optimized(
                    frequencies, params
                )
                ts = time_shifts if np.isscalar(time_shifts) else time_shifts[idx]
                phase += 2 * np.pi * frequencies * ts

            active_indices.append(idx)
            amps_list.append(amp)
            phases_list.append(phase)

        if not active_indices:
            zeros = np.zeros_like(frequencies)
            return zeros, zeros.copy(), zeros.copy(), zeros.copy()

        amp_arr = np.stack(amps_list)
        cosphi_arr = np.cos(np.stack(phases_list))
        sinphi_arr = np.sin(np.stack(phases_list))
        coeffs = _build_mode_coeffs(
            self.modes,
            active_indices,
            Ylm_real,
            Ylm_imag,
            Ylm_real_mneg,
            Ylm_imag_mneg,
        )
        return _sum_modes_einsum(amp_arr, cosphi_arr, sinphi_arr, coeffs)

    def _compute_Ylm_modes(
        self,
        modes: list[Mode],
        phi: float,
        iota: float,
    ) -> tuple[
        dict[Mode, float],
        dict[Mode, float],
        dict[Mode, float],
        dict[Mode, float],
    ]:
        r"""Evaluate the spin-weighted spherical harmonics for the given modes.

        For each mode :math:`(\ell, m)`, computes both
        :math:`{}_{-2}Y_{\ell m}(\iota, \varphi)` and the "opposite"
        :math:`{}_{-2}Y_{\ell,-m}(\iota, \varphi)`, splitting them into
        real and imaginary parts. These are the building blocks consumed
        by :func:`_build_mode_coeffs`.

        Parameters
        ----------
        modes : list[Mode]
            Modes for which to evaluate the harmonics.
        phi : float
            Azimuthal angle :math:`\varphi`, in radians.
        iota : float
            Polar (inclination) angle :math:`\iota`, in radians.

        Returns
        -------
        tuple[dict, dict, dict, dict]
            Four dictionaries, in order:

            * ``Ylm_real[(l, m)]`` and ``Ylm_imag[(l, m)]`` are the real
              and imaginary parts of :math:`{}_{-2}Y_{\ell m}`,
            * ``Ylm_real_mneg[(l, -m)]`` and ``Ylm_imag_mneg[(l, -m)]``
              are the same quantities for the opposite mode
              :math:`{}_{-2}Y_{\ell,-m}`, keyed by ``mode.opposite()``.
        """
        Ylm_real: dict[Mode, float] = {}
        Ylm_imag: dict[Mode, float] = {}
        Ylm_real_mneg: dict[Mode, float] = {}
        Ylm_imag_mneg: dict[Mode, float] = {}

        for mode in modes:
            Ylm_real[mode], Ylm_imag[mode] = spinsphericalharm(
                -2, mode.l, mode.m, phi, iota
            )
            mode_opposite = mode.opposite()
            Ylm_real_mneg[mode_opposite], Ylm_imag_mneg[mode_opposite] = spinsphericalharm(
                -2, mode.l, -mode.m, phi, iota
            )

        return Ylm_real, Ylm_imag, Ylm_real_mneg, Ylm_imag_mneg
