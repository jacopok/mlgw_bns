r"""Higher-order-mode surrogate model.

This module defines :class:`Model`, a thin orchestrator that owns one
:class:`ModeModel` instance per spherical-harmonic mode :math:`(\ell, m)` and
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
and phase is reconstructed by a :class:`ModeModel`, while the mode-relative
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
from importlib.resources import files
from joblib import Parallel, delayed
from numba import njit, prange  # type: ignore

from .data_management import Residuals
from .dataset_generation import Dataset
from .progress import joblib_progress
from .higher_order_modes import (
    Mode,
    ModeGeneratorFactory,
    teob_mode_generator_factory,
    _post_newtonian_amplitudes_by_mode,
    _post_newtonian_phases_by_mode,
)
from .mode_model import ModeModel, ParametersWithExtrinsic
from .neural_network import (
    Hyperparameters,
    ModePhasesNN,
    TimeshiftsGPR,
    TimeshiftsNN,
    load_mode_phases_predictor_from_file,
    load_timeshifts_predictor_from_file,
)
from .special_func import spinsphericalharm

#: Subfolder, relative to the package, holding the pretrained models.
PRETRAINED_MODEL_FOLDER = "data/"

#: Names of the pretrained models shipped with the package.
MODELS_AVAILABLE = ["default_hom"]

#: Modes covered by the pretrained models.
DEFAULT_MODES = [Mode(2, 2), Mode(2, 1), Mode(3, 3), Mode(4, 4)]


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


def _broadcast_time_shifts(
    time_shifts: Union[float, np.ndarray], n_modes: int
) -> np.ndarray:
    """Normalize ``time_shifts`` to one float per mode.

    Parameters
    ----------
    time_shifts : np.ndarray or float
        Either one time shift per mode, or a single one to be used for
        all of them.
    n_modes : int
        Number of modes in the model.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_modes,)``.
    """
    return np.broadcast_to(np.asarray(time_shifts, dtype=float), (n_modes,))


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


class _LazyModeModelsDict(dict):
    """Dictionary that materializes :class:`ModeModel` instances on demand.

    The per-mode :class:`ModeModel` objects can be expensive to construct
    (they instantiate a full :class:`Dataset` with its waveform generator),
    so they are built lazily the first time the user accesses
    ``model.mode_models[mode]`` and cached for subsequent accesses.

    Parameters
    ----------
    model : Model
        Owning :class:`Model`, used to look up the per-mode filename,
        generator factory and constructor keyword arguments.
    """

    def __init__(self, model: "Model"):
        super().__init__()
        self._model = model

    def __missing__(self, mode: Mode) -> ModeModel:
        if mode not in self._model.modes:
            raise KeyError(f"Mode {mode} not in {self._model.modes}")
        mode_model = ModeModel(
            mode=mode,
            filename=self._model.mode_filename(mode),
            waveform_generator=self._model._generator_factory(mode),
            **self._model._mode_model_kwargs,
        )
        # Every mode uses the one shared predictor; see
        # `Model._propagate_time_shifts_predictor`. It may still be
        # None here, if this model has not been trained or loaded yet.
        mode_model.timeshifts_predictor = self._model.time_shifts_predictor
        mode_model.mode_phases_predictor = self._model.mode_phases_predictor
        mode_model.mode_phases_index = self._model.modes.index(mode)
        self[mode] = mode_model
        return mode_model


class Model:
    r"""Higher-order-modes surrogate.

    A :class:`Model` orchestrates one :class:`ModeModel` per spherical
    harmonic mode :math:`(\ell, m)` and combines their predictions to
    produce the observer-frame polarizations :math:`h_+, h_\times` via

    .. math::
        h_+ - i\, h_\times = \sum_{\ell m} A_{\ell m}\, e^{-i\phi_{\ell m}}
                              \; {}_{-2}Y_{\ell m}(\iota, \varphi).

    Each per-mode model is instantiated lazily on first access through
    :attr:`mode_models`, which means that constructing a :class:`Model`
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
        Extra keyword arguments forwarded to each :class:`ModeModel`. The
        special key ``filename`` is consumed here and used as the *base*
        filename; each per-mode model file is named
        ``"{base_filename}_l{l}_m{m}"``.

    Attributes
    ----------
    modes : list[Mode]
        Modes included in this model.
    mode_models : dict[Mode, ModeModel]
        Lazy mapping ``mode -> ModeModel``. The :class:`ModeModel` instance is
        created on first access.
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

        # Shared, cross-mode predictor of the per-mode reference phases
        # ``[phi_lm[f0] for lm in modes]``. Trained by
        # :meth:`_train_reference_predictors` alongside the time-shift
        # predictor, on a small dedicated pre-pass dataset.
        self.mode_phases_predictor: Optional[ModePhasesNN] = (
            self._load_default_mode_phases_predictor()
        )

        # Stored for lazy construction of the per-mode `ModeModel` objects.
        self._generator_factory = generator_factory
        self._mode_model_kwargs = model_kwargs

        self.mode_models: dict[Mode, ModeModel] = _LazyModeModelsDict(self)

    def _load_default_time_shifts_predictor(
        self,
    ) -> Optional[Union[TimeshiftsNN, TimeshiftsGPR]]:
        """Try to load the time-shift predictor saved alongside this model.

        Looks for the checkpoint at :attr:`filename_timeshifts`, i.e.
        ``"{base_filename}_timeshifts.pkl"``. Returns ``None`` if it is
        not available (e.g. no ``base_filename`` was set yet, or the
        model has not been trained/saved).
        """
        if not self.base_filename:
            return None
        try:
            return load_timeshifts_predictor_from_file(self.filename_timeshifts)
        except (FileNotFoundError, ValueError) as e:
            logging.warning(
                "Could not load default time-shift predictor (%s). "
                "`time_shifts` must be provided explicitly to `predict`.",
                e,
            )
            return None

    @property
    def filename_timeshifts(self) -> str:
        """File name in which to save the shared mode time-shifts predictor."""
        return f"{self.base_filename}_timeshifts.pkl"

    def _load_default_mode_phases_predictor(self) -> Optional[ModePhasesNN]:
        """Try to load the mode-phases predictor saved alongside this model."""
        if not self.base_filename:
            return None
        try:
            return load_mode_phases_predictor_from_file(self.filename_mode_phases)
        except (FileNotFoundError, ValueError) as e:
            logging.warning("Could not load default mode-phases predictor (%s).", e)
            return None

    @property
    def filename_mode_phases(self) -> str:
        """File name in which to save the shared per-mode reference-phase predictor."""
        return f"{self.base_filename}_mode_phases.pkl"

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
            # Only update modes whose ModeModel has already been materialized,
            # to avoid eagerly building all of them through __missing__.
            if mode in self.mode_models:
                self.mode_models[mode].filename = self.mode_filename(mode)

    @property
    def dataset(self) -> Dataset:
        """Dataset of the first mode model.

        All per-mode models share the same dataset configuration, so the
        first one is returned for convenience (e.g. for accessing the
        frequency grid or the reference total mass).

        Raises
        ------
        ValueError
            If this :class:`Model` was somehow built with no modes.
        """
        if not self.modes:
            raise ValueError("No models available")
        return self.mode_models[self.modes[0]].dataset

    @property
    def auxiliary_data_available(self) -> bool:
        """``True`` iff every per-mode model has PCA + downsampling data loaded."""
        return all(self.mode_models[mode].auxiliary_data_available for mode in self.modes)

    @property
    def nn_available(self) -> bool:
        """``True`` iff every per-mode model has a trained neural network loaded."""
        return all(self.mode_models[mode].nn_available for mode in self.modes)

    @property
    def training_dataset_available(self) -> bool:
        """``True`` iff every per-mode model has its training dataset available."""
        return all(self.mode_models[mode].training_dataset_available for mode in self.modes)

    def __str__(self) -> str:
        n_modes = len(self.modes)
        modes_str = ", ".join(f"({m.l},{m.m})" for m in self.modes)

        return (
            "Model("
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
    ) -> "Model":
        """Load a pretrained :class:`Model` shipped with the package.

        The metadata/arrays/nn streams of every mode, plus the single
        shared time-shift predictor, are read from the package resources
        (:data:`PRETRAINED_MODEL_FOLDER`). This is the quickest way to get
        a usable model without training one.

        Parameters
        ----------
        model_name : str, optional
            Name of the model to load. Must be one of
            :data:`MODELS_AVAILABLE`. Defaults to the first entry.
        **kwargs
            Extra keyword arguments forwarded to the :class:`Model`
            constructor. The reserved keys ``filename`` and ``modes`` are
            consumed here:

            * ``filename`` overrides the base filename after loading, so
              that subsequent saves write to a user-chosen location.
            * ``modes`` overrides the list of modes to load, which
              defaults to :data:`DEFAULT_MODES`.

        Returns
        -------
        Model
            Loaded :class:`Model` instance.

        Raises
        ------
        ValueError
            If ``model_name`` is not in :data:`MODELS_AVAILABLE`.
        """
        if model_name is None:
            model_name = MODELS_AVAILABLE[0]

        if model_name not in MODELS_AVAILABLE:
            raise ValueError(f"Model {model_name} not available!")

        given_filename = kwargs.pop("filename", None)

        modes = kwargs.pop("modes", None)
        if modes is None:
            modes = list(DEFAULT_MODES)

        base_filename = PRETRAINED_MODEL_FOLDER + model_name

        # Load the shared predictor up front and hand it to the constructor:
        # letting the constructor look for it on disk would only find it if
        # the cwd happened to mirror the package layout.
        kwargs.setdefault(
            "time_shifts_predictor",
            load_timeshifts_predictor_from_file(
                files(__name__).joinpath(f"{base_filename}_timeshifts.pkl").open("rb")
            ),
        )

        model = cls(modes=modes, filename=base_filename, **kwargs)

        try:
            model.mode_phases_predictor = load_mode_phases_predictor_from_file(
                files(__name__).joinpath(f"{base_filename}_mode_phases.pkl").open("rb")
            )
            model._propagate_mode_phases_predictor()
        except (FileNotFoundError, ValueError):
            logging.warning(
                "Pretrained model %s has no mode-phases predictor.", model_name
            )

        for mode in model.modes:
            mode_model = model.mode_models[mode]
            # The per-mode timeshift stream is None: every mode shares the
            # single predictor loaded above.
            mode_model.load(
                streams=(
                    files(__name__).joinpath(mode_model.filename_metadata).open("rb"),
                    files(__name__).joinpath(mode_model.filename_arrays).open("rb"),
                    files(__name__).joinpath(mode_model.filename_nn).open("rb"),
                    None,
                )
            )

        if given_filename is not None:
            model.base_filename = given_filename

        return model

    def generate(
        self,
        training_downsampling_dataset_size: Optional[int] = 64,
        training_pca_dataset_size: Optional[int] = 256,
        training_nn_dataset_size: Optional[int] = 256,
        reference_dataset_size: int = 2000,
        reference_grid_points: int = 64,
        reference_fmax_hz: float = 512.0,
        seed: int = 0,
    ) -> None:
        """Run :meth:`ModeModel.generate` for every mode.

        Builds the downsampling indices, PCA data and training residuals
        for each per-mode :class:`ModeModel`. The three ``training_*``
        dataset sizes have the same meaning as in :meth:`ModeModel.generate`;
        setting one of them to ``None`` reuses pre-existing data for that step.

        When ``training_nn_dataset_size`` is not ``None``, a reference
        pre-pass (:meth:`_train_reference_predictors`) runs *first*: it fits
        the shared cross-mode time-shift predictor :math:`\\Delta t(\\theta)`
        and the shared per-mode reference-phase predictor
        :math:`\\phi_{\\ell m}(f_0)` on ``reference_dataset_size`` waveforms
        sampled on a coarse ``reference_grid_points``-node geometric grid
        (``f_0 -> reference_fmax_hz``). Both predictions are then subtracted
        from every mode's training residuals before the downsampling, PCA
        and NN steps see them, and added back at predict time.

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
        reference_dataset_size : int, optional
            Number of waveforms for the shared time-shift / reference-phase
            pre-pass. Defaults to 2000.
        reference_grid_points : int, optional
            Number of geometric frequency nodes for the pre-pass. Defaults
            to 64.
        reference_fmax_hz : float, optional
            Upper frequency of the pre-pass grid, in Hz. Defaults to 512.
        seed : int, optional
            Seed for the pre-pass parameter generator. Defaults to 0.

        Raises
        ------
        ValueError
            If ``Mode(2, 2)`` is not among :attr:`modes`.
        """
        reference_mode = Mode(2, 2)
        if reference_mode not in self.modes:
            raise ValueError(
                "Model.generate() requires Mode(2, 2) to be among "
                "`self.modes`, since the shared predictors are trained from it."
            )

        if training_nn_dataset_size is not None:
            self._train_reference_predictors(
                reference_dataset_size,
                reference_grid_points,
                reference_fmax_hz,
                seed,
            )

        # Per-mode downsampling indices first: each still trains on its own
        # (small) EOB waveform sweep -- see the plan's Step C.
        if training_downsampling_dataset_size is not None:
            for mode in self.modes:
                mode_model = self.mode_models[mode]
                logging.info("Training the downsampling for mode %s", mode)
                mode_model.downsampling_indices = mode_model.downsampling_training.train(
                    training_downsampling_dataset_size
                )

        # One shared multi-mode EOB sweep feeds the PCA and NN training of
        # every mode, instead of one sweep per (mode, stage).
        precomputed_by_mode: Optional[dict] = None
        if training_pca_dataset_size is not None or training_nn_dataset_size is not None:
            precomputed_by_mode = self._multimode_training_residuals(
                training_pca_dataset_size, training_nn_dataset_size
            )

        for mode in self.modes:
            self.mode_models[mode].generate(
                training_downsampling_dataset_size=None,
                training_pca_dataset_size=training_pca_dataset_size,
                training_nn_dataset_size=training_nn_dataset_size,
                timeshifts_predictor=self.time_shifts_predictor,
                precomputed_residuals=(
                    None if precomputed_by_mode is None
                    else precomputed_by_mode[mode]
                ),
            )

    def _multimode_training_residuals(
        self,
        training_pca_dataset_size: Optional[int],
        training_nn_dataset_size: Optional[int],
    ) -> dict:
        """One multi-mode EOB sweep for the PCA + NN training sets.

        Draws ``max(pca_size, nn_size)`` parameters from the same
        ``seed=2`` generator that ``Dataset.generate_residuals`` uses, runs
        one EOB call per point via :meth:`_multimode_mode_residuals`, and
        returns ``mode -> (freq_downsampled_natural, ParameterSet,
        Residuals)`` shaped exactly like ``Dataset.generate_residuals``'s
        return so :meth:`ModeModel.generate` can consume it directly.
        """
        size = max(
            s for s in (training_pca_dataset_size, training_nn_dataset_size)
            if s is not None
        )
        dataset = self.mode_models[Mode(2, 2)].dataset
        parameter_generator = dataset.make_parameter_generator(seed=2)
        params_list = [next(parameter_generator) for _ in range(size)]

        downsampling_indices_by_mode = {
            mode: self.mode_models[mode].downsampling_indices for mode in self.modes
        }
        amplitude_reference_by_mode = {
            mode: self.mode_models[mode].dataset.amplitude_reference_parameters
            for mode in self.modes
        }

        parameter_array, amp_residuals, phase_residuals = self._multimode_mode_residuals(
            params_list,
            dataset.frequencies,
            downsampling_indices_by_mode,
            amplitude_reference_by_mode,
            progress_desc="PCA/NN training sweep",
        )
        if len(parameter_array) < size:
            logging.warning(
                "Multi-mode training sweep: only %d/%d valid waveforms",
                len(parameter_array), size,
            )

        precomputed = {}
        for mode in self.modes:
            phase_indices = self.mode_models[mode].downsampling_indices.phase_indices
            precomputed[mode] = (
                dataset.frequencies[phase_indices],
                # float64 params + phase residual: see the dtype note in
                # Dataset.generate_residuals. Amplitude residual is O(1).
                dataset.parameter_set_cls(np.asarray(parameter_array, dtype=np.float64)),
                Residuals(
                    amp_residuals[mode].astype(np.float32),
                    np.asarray(phase_residuals[mode], dtype=np.float64),
                ),
            )
        return precomputed

    def _train_reference_predictors(
        self,
        reference_dataset_size: int,
        reference_grid_points: int,
        reference_fmax_hz: float,
        seed: int,
    ) -> None:
        r"""Fit the shared time-shift and per-mode reference-phase predictors.

        Runs *before* any per-mode :meth:`ModeModel.generate`, on its own
        small parameter sample and a coarse geometric frequency grid
        (``f_0 -> reference_fmax_hz``). One EOB residual draw per mode
        feeds both:

        * :class:`~mlgw_bns.neural_network.TimeshiftsNN` --- the
          least-squares low-frequency slope of the (2,2) phase residual
          (``Residuals.phase_timeshifts``), the shared cross-mode
          :math:`\Delta t(\theta)`;
        * :class:`~mlgw_bns.neural_network.ModePhasesNN` --- the raw
          per-mode reference phase :math:`\phi_{\ell m}(f_0)`, which it
          models as an analytic stationary-phase backbone plus a smooth
          ridge-fit leftover (see
          :func:`~mlgw_bns.pn_modes.reference_phase_backbone`).

        Both predictions are subtracted from the per-mode training
        residuals by ``remove_linear_trend`` and added back at predict
        time, so the downstream PCA/NN only ever see small residuals.
        """
        reference_mode = Mode(2, 2)
        if reference_mode not in self.modes:
            raise ValueError(
                "Model.generate() requires Mode(2, 2) to be among "
                "`self.modes`, since the shared predictors are trained from it."
            )

        dataset = self.mode_models[reference_mode].dataset
        f0_natural = float(dataset.frequencies[0])
        grid_hz = np.geomspace(
            dataset.natural_units_to_hz(f0_natural),
            min(reference_fmax_hz, dataset.effective_srate_hz / 2),
            reference_grid_points,
        )
        f_ref_natural = dataset.hz_to_natural_units(grid_hz)
        f_ref_natural[0] = f0_natural
        grid_hz = dataset.natural_units_to_hz(f_ref_natural)

        parameter_generator = dataset.make_parameter_generator(seed=seed)
        params_list = [next(parameter_generator) for _ in range(reference_dataset_size)]
        parameter_array = np.array([p.array for p in params_list], dtype=float)

        parameter_array, _, phase_residuals = self._multimode_mode_residuals(
            params_list, f_ref_natural, progress_desc="Reference pre-pass sweep"
        )

        if len(parameter_array) < 2:
            raise RuntimeError(
                "The reference pre-pass produced fewer than 2 valid waveforms."
            )
        logging.info(
            "Reference pre-pass: %d/%d valid waveforms on a %d-point grid "
            "[%.1f, %.1f] Hz",
            len(parameter_array), reference_dataset_size, len(grid_hz),
            grid_hz[0], grid_hz[-1],
        )

        reference_phase_residuals = phase_residuals[reference_mode]
        timeshifts = Residuals(
            np.zeros_like(reference_phase_residuals), reference_phase_residuals
        ).phase_timeshifts(frequencies=grid_hz)
        self.time_shifts_predictor = TimeshiftsNN(
            training_params=parameter_array,
            training_timeshifts=timeshifts,
        ).fit()
        self._propagate_time_shifts_predictor()

        reference_phases = np.stack(
            [phase_residuals[mode][:, 0] for mode in self.modes], axis=1
        )
        self.mode_phases_predictor = ModePhasesNN(
            modes=[(m.l, m.m) for m in self.modes],
            f0_natural=f0_natural,
            training_params=parameter_array,
            training_mode_phases=reference_phases,
        ).fit()
        self._propagate_mode_phases_predictor()

    def _multimode_mode_residuals(
        self,
        params_list: list,
        frequencies_natural: np.ndarray,
        downsampling_indices_by_mode: Optional[dict] = None,
        amplitude_reference_by_mode: Optional[dict] = None,
        progress_desc: str = "Multi-mode EOB sweep",
    ):
        r"""One EOB call per parameter point, residuals for every mode.

        For each :class:`~mlgw_bns.dataset_generation.WaveformParameters` in
        ``params_list`` a single
        :meth:`~mlgw_bns.higher_order_modes.TEOBResumSModeGenerator.all_modes_amplitude_phase`
        call produces all of :attr:`modes`; the per-mode Post-Newtonian
        amplitude/phase are then divided/subtracted and the result optionally
        cropped to that mode's downsampling indices. This replaces the
        one-EOB-call-per-(mode, parameter) pattern in the training path.

        Parameters
        ----------
        params_list
            Shared parameter sample; every mode is evaluated at the same points.
        frequencies_natural
            Grid (natural units) handed to the EOB call.
        downsampling_indices_by_mode
            Optional ``mode -> DownsamplingIndices``; when given the returned
            residuals are already restricted to those indices (per mode).
        amplitude_reference_by_mode
            Optional ``mode -> WaveformParameters`` for the fixed-reference
            amplitude normalisation (see
            :meth:`WaveformGenerator.generate_residuals`).
        progress_desc
            Label for the progress reporting of the parallel EOB sweep.

        Returns
        -------
        tuple
            ``(parameter_array, amp_residuals, phase_residuals)`` where the
            two dicts map ``mode -> np.ndarray`` of shape
            ``(n_valid, n_points_for_that_mode)``; a parameter is dropped from
            *all* modes if the EOB call fails or returns non-finite / wrong-shape
            output for any of them. ``parameter_array`` has shape
            ``(n_valid, 5)``.
        """
        modes = list(self.modes)
        generator = self.mode_models[modes[0]].waveform_generator
        frequencies_natural = np.asarray(frequencies_natural, dtype=float)
        n_points = len(frequencies_natural)

        pn_generators = {m: self.mode_models[m].waveform_generator for m in modes}
        ds_idx = downsampling_indices_by_mode or {}
        amp_ref = amplitude_reference_by_mode or {}

        def _one(params):
            try:
                waveforms = generator.all_modes_amplitude_phase(
                    params, modes, frequencies_natural
                )
            except Exception:  # pragma: no cover - EOB blowups
                return None
            out = {}
            for mode in modes:
                f_eob, amp_eob, phi_eob = waveforms[mode]
                if (
                    len(amp_eob) != n_points
                    or not np.all(np.isfinite(amp_eob))
                    or not np.all(np.isfinite(phi_eob))
                ):
                    return None
                pn_gen = pn_generators[mode]
                reference = amp_ref.get(mode)
                amp_pn = pn_gen.post_newtonian_amplitude(
                    params if reference is None else reference, f_eob
                )
                phi_pn = pn_gen.post_newtonian_phase(params, f_eob)
                amp_res = amp_eob / amp_pn
                phi_res = phi_eob - phi_pn
                if mode in ds_idx:
                    amp_indices, phi_indices = ds_idx[mode]
                    amp_res = amp_res[amp_indices]
                    phi_res = phi_res[phi_indices]
                out[mode] = (
                    np.asarray(amp_res, dtype=float),
                    np.asarray(phi_res, dtype=float),
                )
            return out

        with joblib_progress(progress_desc, len(params_list)):
            results = Parallel(n_jobs=16)(delayed(_one)(p) for p in params_list)

        keep = [i for i, r in enumerate(results) if r is not None]
        parameter_array = np.array(
            [params_list[i].array for i in keep], dtype=float
        )
        amp_residuals = {
            mode: np.stack([results[i][mode][0] for i in keep])
            if keep
            else np.empty((0, 0))
            for mode in modes
        }
        phase_residuals = {
            mode: np.stack([results[i][mode][1] for i in keep])
            if keep
            else np.empty((0, 0))
            for mode in modes
        }
        return parameter_array, amp_residuals, phase_residuals

    def _propagate_mode_phases_predictor(self) -> None:
        """Point every already-built per-mode model at the shared
        mode-phases predictor, tagging each with its output column."""
        for idx, mode in enumerate(self.modes):
            if mode in self.mode_models:
                mm = self.mode_models[mode]
                mm.mode_phases_predictor = self.mode_phases_predictor
                mm.mode_phases_index = idx

    def _propagate_time_shifts_predictor(self) -> None:
        """Point every already-built per-mode model at the shared predictor.

        The models built later pick it up in
        :meth:`_LazyModeModelsDict.__missing__`; this covers the ones which
        already exist by the time the predictor becomes available.
        """
        for model in self.mode_models.values():
            model.timeshifts_predictor = self.time_shifts_predictor

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
            :meth:`ModeModel.set_hyper_and_train_nn`. Defaults to all data.
        """
        for mode in self.modes:
            self.mode_models[mode].set_hyper_and_train_nn(hyper=hyper, idxs=idxs)

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
        # `include_timeshifts_predictor=False`: there is one predictor for
        # all the modes, written once below under this model's own base
        # filename, rather than a redundant copy next to every mode.
        def save_mode(mode: Mode) -> None:
            self.mode_models[mode].save(
                include_training_data=include_training_data,
                include_timeshifts_predictor=False,
            )

        if len(self.modes) > 1:
            with ThreadPoolExecutor(max_workers=len(self.modes)) as executor:
                # Drain the iterator so that any exceptions are propagated.
                list(executor.map(save_mode, self.modes))
        else:
            for mode in self.modes:
                save_mode(mode)

        if self.time_shifts_predictor is not None:
            self.time_shifts_predictor.save_model(self.filename_timeshifts)

        if self.mode_phases_predictor is not None:
            self.mode_phases_predictor.save_model(self.filename_mode_phases)

    def load(
        self,
        streams: Optional[tuple[IO[bytes], IO[bytes], IO[bytes]]] = None,
    ) -> None:
        """Load every per-mode model from disk.

        Parameters
        ----------
        streams : tuple[IO[bytes], IO[bytes], IO[bytes]], optional
            Pre-opened streams ``(metadata, arrays, nn)`` to load from,
            forwarded as-is to every per-mode :meth:`ModeModel.load`. When
            ``None`` (the default), each model opens its own files from
            the path implied by :meth:`mode_filename`.
        """
        for mode in self.modes:
            self.mode_models[mode].load(streams=streams)

        # Per-mode checkpoints carry no predictor of their own (older ones
        # may, in which case this overwrites the redundant copy with the
        # shared one they were all identical to anyway).
        if self.time_shifts_predictor is None:
            self.time_shifts_predictor = self._load_default_time_shifts_predictor()
        self._propagate_time_shifts_predictor()

        if self.mode_phases_predictor is None:
            self.mode_phases_predictor = self._load_default_mode_phases_predictor()
        self._propagate_mode_phases_predictor()

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
        if mode not in self.modes:
            raise ValueError(f"Mode {mode} is not included in this model")

        return self.mode_models[mode].predict_amplitude_phase_optimized(frequencies, params)

    def _resolve_time_shifts(
        self,
        params: ParametersWithExtrinsic,
        time_shifts: Optional[Union[float, np.ndarray]],
    ) -> Union[float, np.ndarray]:
        """Return the time shifts to use, predicting them if needed.

        Parameters
        ----------
        params : ParametersWithExtrinsic
            Source parameters, used to query :attr:`time_shifts_predictor`.
        time_shifts : np.ndarray or float or None
            Explicitly provided time shifts, returned unchanged if not
            ``None``.

        Returns
        -------
        np.ndarray or float
            The given ``time_shifts``, or the prediction of
            :attr:`time_shifts_predictor` for ``params``.

        Raises
        ------
        ValueError
            If ``time_shifts`` is ``None`` and this model has no
            time-shift predictor available.
        """
        if time_shifts is not None:
            return time_shifts

        if self.time_shifts_predictor is None:
            raise ValueError(
                "This model has no time-shift predictor available, so the "
                "`time_shifts` aligning the mode mergers cannot be computed "
                "automatically: please provide them explicitly."
            )

        # One row in, one row out: a scalar if the predictor was trained on
        # a single shared time shift, one value per mode otherwise.
        prediction = self.time_shifts_predictor.predict(
            np.array([params.intrinsic(self.dataset).array])
        )
        return np.asarray(prediction)[0]

    def predict(
        self,
        frequencies: np.ndarray,
        params: ParametersWithExtrinsic,
        time_shifts: Optional[Union[float, np.ndarray]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Predict the full frequency-domain waveform from all modes.

        Combines the predictions of every per-mode :class:`ModeModel` into the
        two observer-frame polarizations :math:`h_+, h_\times`, using the
        provided mode-relative time shifts and the inclination contained
        in ``params``.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies at which to evaluate the waveform, in Hz.
        params : ParametersWithExtrinsic
            Source parameters (intrinsic + extrinsic).
        time_shifts : np.ndarray or float, optional
            Time shifts, one per mode, that align the per-mode mergers
            in the time domain; a scalar is broadcast to every mode.
            If ``None`` (the default) they are predicted from ``params``
            with :attr:`time_shifts_predictor`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The complex polarizations ``(h_plus, h_cross)``, in the same
            convention as :meth:`ModeModel.predict`. The combination
            appearing in the mode decomposition is ``h_plus - 1j * h_cross``.

        Raises
        ------
        ValueError
            If ``time_shifts`` is ``None`` and no predictor is available.

        References
        ----------
        See Appendix E of `arXiv:2004.06503
        <https://arxiv.org/pdf/2004.06503.pdf>`_.
        """
        return self._compute_polarizations_from_modes(
            frequencies=frequencies,
            params=params,
            time_shifts=self._resolve_time_shifts(params, time_shifts),
            inclination=params.inclination,
        )

    def _compute_polarizations_from_modes(
        self,
        frequencies: np.ndarray,
        params: ParametersWithExtrinsic,
        time_shifts: Union[float, np.ndarray],
        inclination: float,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        time_shifts : np.ndarray or float
            Per-mode time shifts; a scalar is broadcast to every mode.
        inclination : float
            Inclination angle, in radians.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(h_plus, h_cross)``.
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

        return hp_pred, hc_pred

    def predict_modes_dict(
        self,
        frequencies: np.ndarray,
        params: ParametersWithExtrinsic,
        time_shifts: Optional[Union[float, np.ndarray]] = None,
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
        time_shifts : np.ndarray or float, optional
            Per-mode time shifts (see :meth:`predict`). Predicted from
            ``params`` if ``None``.

        Returns
        -------
        dict[tuple[int, int], np.ndarray]
            Mapping ``(l, m) -> h_lm = h_+ - i h_x`` (one complex array
            per mode).
        """
        modes_dict = self._hpc_waveform_per_mode(
            frequencies=frequencies,
            params=params,
            time_shifts=self._resolve_time_shifts(params, time_shifts),
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
        time_shifts: Union[float, np.ndarray],
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
        time_shifts : np.ndarray or float
            Per-mode time shifts; a scalar is broadcast to every mode.
        inclination : float
            Inclination angle, in radians.
        use_pn : bool
            If ``True``, take the per-mode amplitude/phase from the
            Post-Newtonian (TaylorF2-style) expressions in
            :mod:`~mlgw_bns.pn_modes` instead of from the trained
            :class:`ModeModel` instances. Used by :meth:`get_taylorf2_modes_dict`.

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

        time_shifts_per_mode = _broadcast_time_shifts(time_shifts, len(self.modes))

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
                amp, phase = self.mode_models[mode].predict_amplitude_phase_optimized(
                    frequencies, params
                )
                ts = time_shifts_per_mode[idx]
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
            _f_spa, amp, phase = self.mode_models[mode].waveform_generator.get_amplitude_phase_at_inclination(
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
        time_shifts: Union[float, np.ndarray],
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

        time_shifts_per_mode = _broadcast_time_shifts(time_shifts, len(self.modes))

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
                amp, phase = self.mode_models[mode].predict_amplitude_phase_optimized(
                    frequencies, params
                )
                ts = time_shifts_per_mode[idx]
                # Time shifts are stored in units of the reference total mass
                # of the dataset, so we rescale to the requested total mass.
                ts_scaled = ts * (params.total_mass / self.dataset.total_mass)
                phase += 2 * np.pi * frequencies * ts_scaled

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
