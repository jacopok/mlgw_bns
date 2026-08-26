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
and phase is reconstructed by a :class:`ModeModel`, from residuals which are
referenced, waveform by waveform, to the (2,2) mode at the lowest frequency
of the training band; at prediction time the (2,2) mode itself gives the
merger time and coalescence phase to which all of them are referenced.

The module also exposes two summation kernels --- a Numba parallel kernel
and a NumPy ``einsum`` kernel --- that perform the per-frequency sum over
modes, weighted by the appropriate combinations of the spin-weighted
spherical harmonics.
"""

from __future__ import annotations

import copy
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import IO, TYPE_CHECKING, Callable, Optional, Sequence, Union

import numpy as np
from importlib.resources import files
from joblib import Parallel, delayed
from numba import njit, prange  # type: ignore

from .data_management import (
    ParameterRanges,
    Residuals,
    re_reference,
    reference_gauge,
)
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
from .neural_network import Hyperparameters
from .special_func import spinsphericalharm

if TYPE_CHECKING:
    from .batched import BatchedSurrogate

#: Subfolder, relative to the package, holding the pretrained models.
PRETRAINED_MODEL_FOLDER = "data/"

#: Names of the pretrained models shipped with the package.
MODELS_AVAILABLE = ["default_hom"]

#: Modes covered by the pretrained models.
DEFAULT_MODES = [
    Mode(2, 2),
    Mode(2, 1),
    Mode(3, 1),
    Mode(3, 2),
    Mode(3, 3),
    Mode(4, 3),
    Mode(4, 4),
]


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

    Raises
    ------
    ValueError
        If ``modes`` does not include the (2,2) mode, which sets the merger
        time and coalescence phase of all the others.

    References
    ----------
    See Appendix E of `arXiv:2004.06503
    <https://arxiv.org/pdf/2004.06503.pdf>`_ for the mode decomposition.
    """

    def __init__(
        self,
        modes: list[Mode],
        generator_factory: ModeGeneratorFactory = teob_mode_generator_factory,
        **model_kwargs,
    ):
        modes = [Mode(int(lm[0]), int(lm[1])) for lm in modes]
        if Mode(2, 2) not in modes:
            raise ValueError(
                "The modes of a Model must include (2, 2): it sets the merger "
                "time and coalescence phase of all the others."
            )

        self.modes = modes
        self._base_filename = model_kwargs.pop("filename", "")

        # Stored for lazy construction of the per-mode `ModeModel` objects.
        self._generator_factory = generator_factory
        self._mode_model_kwargs = model_kwargs

        self.mode_models: dict[Mode, ModeModel] = _LazyModeModelsDict(self)

        # `(modes, backend) -> BatchedSurrogate`, see `batched_surrogate`;
        # emptied whenever the underlying model changes.
        self._batched_cache: dict = {}

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
    def parameter_ranges(self) -> ParameterRanges:
        """Parameter ranges within which predictions are accepted.

        Those of the first mode model; :meth:`predict` and
        :meth:`predict_modes_dict` raise outside them, and the batched
        :meth:`predict_modes_amp_phase` returns NaN rows.

        Assigning a new :class:`~mlgw_bns.data_management.ParameterRanges`
        applies it to every mode, e.g. to accept tidal deformabilities
        below the lower bound that was only a guard for the EOB code at
        training time::

            model.parameter_ranges = dataclasses.replace(
                model.parameter_ranges, lambda1_range=(0.0, 5000.0),
                lambda2_range=(0.0, 5000.0))

        Only the checks change: each mode's :class:`Dataset` keeps the
        ranges it was trained with, from which it derives its frequency
        band and its reference amplitude. Mutating the returned object in
        place is *not* the same (it is shared with the first mode's
        dataset, and not seen by the other modes).
        """
        return self.mode_models[self.modes[0]].parameter_ranges

    @parameter_ranges.setter
    def parameter_ranges(self, value: ParameterRanges) -> None:
        for mode in self.modes:
            self.mode_models[mode].parameter_ranges = value
        self._batched_cache.clear()

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

        The metadata/arrays/nn streams of every mode are read from the
        package resources
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

        model = cls(modes=modes, filename=base_filename, **kwargs)

        for mode in model.modes:
            mode_model = model.mode_models[mode]
            mode_model.load(
                streams=(
                    files(__name__).joinpath(mode_model.filename_metadata).open("rb"),
                    files(__name__).joinpath(mode_model.filename_arrays).open("rb"),
                    files(__name__).joinpath(mode_model.filename_nn).open("rb"),
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
        n_jobs: int = 1,
    ) -> None:
        """Run :meth:`ModeModel.generate` for every mode.

        Builds the downsampling indices, PCA data and training residuals
        for each per-mode :class:`ModeModel`. The three ``training_*``
        dataset sizes have the same meaning as in :meth:`ModeModel.generate`;
        setting one of them to ``None`` reuses pre-existing data for that step.

        One EOB call per parameter point produces every mode; the phase
        residuals of each waveform are referenced to its (2,2) mode at the
        lowest frequency of the band (see
        :func:`~mlgw_bns.data_management.re_reference`), so that no
        regressor for time shifts or mode phases is needed.

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
        n_jobs : int, optional
            Number of parallel worker processes used for every EOB sweep in
            this call (the per-mode downsampling training, and the shared
            PCA/NN sweep). Sequential (``1``) by
            default -- parallelism is opt-in; pass a higher value
            explicitly to use multiple workers.

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
        self._batched_cache.clear()

        # Per-mode downsampling indices first: each still trains on its own
        # (small) EOB waveform sweep -- see the plan's Step C.
        if training_downsampling_dataset_size is not None:
            for mode in self.modes:
                mode_model = self.mode_models[mode]
                logging.info("Training the downsampling for mode %s", mode)
                mode_model.downsampling_training.n_jobs = n_jobs
                mode_model.downsampling_indices = mode_model.downsampling_training.train(
                    training_downsampling_dataset_size
                )

        # One shared multi-mode EOB sweep feeds the PCA and NN training of
        # every mode, instead of one sweep per (mode, stage).
        precomputed_by_mode: Optional[dict] = None
        if training_pca_dataset_size is not None or training_nn_dataset_size is not None:
            precomputed_by_mode = self._multimode_training_residuals(
                training_pca_dataset_size, training_nn_dataset_size, n_jobs=n_jobs
            )

        for mode in self.modes:
            self.mode_models[mode].generate(
                training_downsampling_dataset_size=None,
                training_pca_dataset_size=training_pca_dataset_size,
                training_nn_dataset_size=training_nn_dataset_size,
                precomputed_residuals=(
                    None if precomputed_by_mode is None
                    else precomputed_by_mode[mode]
                ),
                n_jobs=n_jobs,
            )

    def _multimode_training_residuals(
        self,
        training_pca_dataset_size: Optional[int],
        training_nn_dataset_size: Optional[int],
        n_jobs: int = 1,
    ) -> dict:
        """One multi-mode EOB sweep for the PCA + NN training sets.

        Draws ``max(pca_size, nn_size)`` parameters from the same
        ``seed=2`` generator that ``Dataset.generate_residuals`` uses, runs
        one EOB call per point via :meth:`_multimode_mode_residuals`, and
        returns ``mode -> (freq_downsampled_natural, ParameterSet,
        Residuals)`` shaped exactly like ``Dataset.generate_residuals``'s
        return so :meth:`ModeModel.generate` can consume it directly.

        ``n_jobs`` is sequential (``1``) by default -- parallelism is
        opt-in; pass a higher value explicitly to use multiple workers.
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
            n_jobs=n_jobs,
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

    def _multimode_mode_residuals(
        self,
        params_list: list,
        frequencies_natural: np.ndarray,
        downsampling_indices_by_mode: Optional[dict] = None,
        amplitude_reference_by_mode: Optional[dict] = None,
        progress_desc: str = "Multi-mode EOB sweep",
        n_jobs: int = 1,
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
        n_jobs
            Number of parallel worker processes for the EOB sweep.
            Sequential (``1``) by default -- parallelism is opt-in; pass a
            higher value explicitly to use multiple workers.

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
            raw = {}
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
                raw[mode] = (f_eob, amp_eob / amp_pn, phi_eob - phi_pn)
            # Every mode referenced to the (2,2) of the same waveform at f0,
            # on the full grid; see `reference_gauge` and `re_reference`.
            f_22, _, phi_22 = raw[Mode(2, 2)]
            value, slope = reference_gauge(f_22, phi_22)
            out = {}
            for mode in modes:
                f_eob, amp_res, phi_res = raw[mode]
                phi_res = re_reference(f_eob, phi_res, mode.m, value, slope)
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
            results = Parallel(n_jobs=n_jobs)(delayed(_one)(p) for p in params_list)

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
        self._batched_cache.clear()

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
        def save_mode(mode: Mode) -> None:
            self.mode_models[mode].save(include_training_data=include_training_data)

        if len(self.modes) > 1:
            with ThreadPoolExecutor(max_workers=len(self.modes)) as executor:
                # Drain the iterator so that any exceptions are propagated.
                list(executor.map(save_mode, self.modes))
        else:
            for mode in self.modes:
                save_mode(mode)

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

        self._batched_cache.clear()

    def merger_reference(self, params: ParametersWithExtrinsic) -> tuple[float, float]:
        """The (2,2) reference line of ``params``, see
        :meth:`ModeModel.merger_reference`."""
        mode_model = self.mode_models[Mode(2, 2)]
        return mode_model.merger_reference(params.intrinsic(mode_model.dataset))

    def _mode_amplitudes_and_phases(
        self,
        frequencies: np.ndarray,
        params: ParametersWithExtrinsic,
        source: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Amplitude and phase of every mode, from the requested source.

        This is the one place where the per-mode
        :math:`A_{\ell m}(f), \phi_{\ell m}(f)` are gathered; the
        polarization builders (:meth:`_hpc_waveform`,
        :meth:`_hpc_waveform_per_mode`, :meth:`get_teob_modes_dict`,
        :meth:`coprecessing_modes_dict`) all go through it, so that they
        cannot drift apart in their conventions.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies at which to evaluate, in Hz.
        params : ParametersWithExtrinsic
            Source parameters.
        source : str
            Where the amplitude and phase come from:

            * ``"surrogate"``: the trained :class:`ModeModel` networks,
              referenced to the merger (see :meth:`merger_reference`);
            * ``"post_newtonian"``: the TaylorF2-style expressions of
              :mod:`~mlgw_bns.pn_modes`, in their own reference;
            * ``"eob"``: TEOBResumS, through :meth:`teob_modes_amp_phase`,
              referenced to the merger as the surrogate is.

          Note that none of the three depends on the inclination: these
          are the multipoles, which the caller is expected to weight by
          the spin-weighted spherical harmonics itself.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(amplitudes, phases)``, each of shape
            ``(len(self.modes), len(frequencies))`` and ordered like
            :attr:`modes`.

        Raises
        ------
        ValueError
            If ``source`` is not one of the three accepted values.
        """
        if source == "eob":
            modes_amp_phase = self.teob_modes_amp_phase(frequencies, params)
            return (
                np.stack([modes_amp_phase[mode][0] for mode in self.modes]),
                np.stack([modes_amp_phase[mode][1] for mode in self.modes]),
            )
        if source not in ("surrogate", "post_newtonian"):
            raise ValueError(
                f"Unknown amplitude/phase source {source!r}: expected one of "
                "'surrogate', 'post_newtonian', 'eob'."
            )

        amps_list: list[np.ndarray] = []
        phases_list: list[np.ndarray] = []
        if source == "post_newtonian":
            parameters_intrinsic = params.intrinsic(self.dataset)
        else:
            merger_reference = self.merger_reference(params)

        for mode in self.modes:
            if source == "post_newtonian":
                amp = _post_newtonian_amplitudes_by_mode[mode](
                    parameters_intrinsic,
                    frequencies * params.mass_sum_seconds,
                )
                phase = _post_newtonian_phases_by_mode[mode](
                    parameters_intrinsic,
                    frequencies * params.mass_sum_seconds,
                )
            else:
                amp, phase = self.mode_models[mode].predict_amplitude_phase(
                    frequencies, params, merger_reference=merger_reference
                )
            amps_list.append(amp)
            phases_list.append(phase)

        return np.stack(amps_list), np.stack(phases_list)

    def coprecessing_modes_dict(
        self,
        frequencies: np.ndarray,
        params: ParametersWithExtrinsic,
        source: str = "surrogate",
    ) -> dict[tuple[int, int], np.ndarray]:
        r"""Per-mode frequency-domain multipoles, with no sky projection.

        Returns the multipoles :math:`\tilde{h}_{\ell m}(f)` themselves
        --- what :meth:`predict_modes_dict` returns *before* it is
        weighted by :math:`{}_{-2}Y_{\ell m}(\iota, \varphi)`:

        .. math::
            \tilde{h}_{\ell m}(f) = \frac{1}{\eta}
                A_{\ell m}(f) e^{i \phi_{\ell m}(f)} \,.

        For an aligned-spin binary --- which is all this surrogate models
        --- these *are* the co-precessing-frame multipoles, which is what
        makes them the natural input to the precessing twist of
        :mod:`~mlgw_bns.precessing_model`.

        Only :math:`m > 0` multipoles are returned: with the
        :math:`\tilde{h}(f) = \int h(t) e^{2 \pi i f t} \mathrm{d}t`
        convention used throughout, the :math:`m < 0` multipoles of an
        inspiralling binary have their support at :math:`f < 0` and so
        vanish on the positive-frequency grid, being recoverable there
        from :math:`h_{\ell, -m} = (-1)^\ell h_{\ell m}^*`.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies at which to evaluate the multipoles, in Hz.
        params : ParametersWithExtrinsic
            Source parameters. The inclination it carries is *not* used.
        source : str
            ``"surrogate"`` (default), ``"post_newtonian"`` or ``"eob"``;
            see :meth:`_mode_amplitudes_and_phases`.

        Returns
        -------
        dict[tuple[int, int], np.ndarray]
            Mapping ``(l, m) -> h_lm(f)``, one complex array per mode.
        """
        amp_arr, phase_arr = self._mode_amplitudes_and_phases(
            frequencies=frequencies,
            params=params,
            source=source,
        )
        eta = params.intrinsic(self.dataset).eta
        return {
            (mode.l, mode.m): amp_arr[i] * np.exp(1j * phase_arr[i]) / eta
            for i, mode in enumerate(self.modes)
        }

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
            ``(amplitude, phase)`` arrays for the requested mode, referenced
            to the merger as in :meth:`ModeModel.predict_amplitude_phase`.

        Raises
        ------
        ValueError
            If ``mode`` is not among :attr:`modes`.
        """
        if mode not in self.modes:
            raise ValueError(f"Mode {mode} is not included in this model")

        return self.mode_models[mode].predict_amplitude_phase(
            frequencies, params, merger_reference=self.merger_reference(params)
        )

    def predict(
        self,
        frequencies: np.ndarray,
        params: ParametersWithExtrinsic,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Predict the full frequency-domain waveform from all modes.

        Combines the predictions of every per-mode :class:`ModeModel` into the
        two observer-frame polarizations :math:`h_+, h_\times`, using the
        inclination contained in ``params``. The merger is at
        ``params.merger_time``, with orbital phase ``params.coalescence_phase``.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies at which to evaluate the waveform, in Hz.
        params : ParametersWithExtrinsic
            Source parameters (intrinsic + extrinsic).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The complex polarizations ``(h_plus, h_cross)``, in the same
            convention as :meth:`ModeModel.predict`. The combination
            appearing in the mode decomposition is ``h_plus - 1j * h_cross``.

        References
        ----------
        See Appendix E of `arXiv:2004.06503
        <https://arxiv.org/pdf/2004.06503.pdf>`_.
        """
        h_plus_real, h_plus_imag, h_cross_real, h_cross_imag = self._hpc_waveform(
            frequencies=frequencies,
            params=params,
            inclination=params.inclination,
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

        Returns
        -------
        dict[tuple[int, int], np.ndarray]
            Mapping ``(l, m) -> h_lm = h_+ - i h_x`` (one complex array
            per mode).
        """
        modes_dict = self._hpc_waveform_per_mode(
            frequencies=frequencies,
            params=params,
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

    def batched_surrogate(
        self, modes: Optional[Sequence] = None, backend: str = "numpy"
    ) -> "BatchedSurrogate":
        """The model frozen for batched evaluation of ``modes``.

        Built on first use and cached per ``(modes, backend)``; see
        :class:`~mlgw_bns.batched.BatchedSurrogate`. The cache is emptied
        when the model is loaded, (re)trained or given new
        :attr:`parameter_ranges`.

        Parameters
        ----------
        modes : sequence of (l, m), optional
            Defaults to :attr:`modes`, in that order.
        backend : str
            ``"numpy"`` or ``"jax"``.
        """
        from .batched import BatchedSurrogate

        key = (
            None if modes is None else tuple((int(l), int(m)) for l, m in modes),
            backend,
        )
        if key not in self._batched_cache:
            self._batched_cache[key] = BatchedSurrogate(self, modes=modes, backend=backend)
        return self._batched_cache[key]

    def predict_modes_amp_phase(
        self,
        intrinsic: np.ndarray,
        total_mass: Union[float, np.ndarray],
        frequencies: np.ndarray,
        modes: Optional[Sequence] = None,
        distance_mpc: Union[float, np.ndarray] = 1.0,
        return_tf: bool = False,
    ) -> tuple[np.ndarray, ...]:
        r"""Amplitude and phase of individual modes, for a batch of binaries.

        The batched counterpart of :meth:`predict_modes_dict`: one call
        evaluates ``N`` parameter sets, only for the requested modes, and
        returns each mode's amplitude and phase rather than its projection
        on the sky. The full conventions (sign of the phase, time shift,
        spherical harmonics, low- and high-frequency behaviour, accuracy)
        are in :mod:`mlgw_bns.batched`; in short, the multipole is
        :math:`\tilde{h}_{\ell m}(f) = A\, e^{+i\phi}` and
        :func:`mlgw_bns.batched.mode_polarizations` projects it on
        :math:`h_+, h_\times` as :meth:`predict_modes_dict` does.

        Parameters
        ----------
        intrinsic : np.ndarray
            Shape ``(N, 5)``: rows of
            :math:`[q \geq 1, \Lambda_1, \Lambda_2, \chi_1, \chi_2]`.
        total_mass : float or np.ndarray
            Shape ``(N,)``: total mass in solar masses, per row.
        frequencies : np.ndarray
            Shape ``(k,)`` (shared) or ``(N, k)``: increasing frequencies
            in Hz. Below the trained band the modes are continued with
            their post-Newtonian expressions.
        modes : sequence of (l, m), optional
            Modes to return, in this order. Defaults to :attr:`modes`.
        distance_mpc : float or np.ndarray
            Luminosity distance in Mpc, per row. Defaults to 1.
        return_tf : bool
            Also return the time :math:`t_{\ell m}(f) = -\frac{1}{2\pi}
            \partial_f \phi_{\ell m}` at which each mode emits each
            frequency, in seconds relative to the merger.

        Returns
        -------
        tuple[np.ndarray, ...]
            ``(amp, phase)``, or ``(amp, phase, tf)``, each of shape
            ``(N, n_modes, k)``. Rows outside :attr:`parameter_ranges`
            are NaN.
        """
        return self.batched_surrogate(modes)(
            intrinsic, total_mass, frequencies, distance_mpc, return_tf=return_tf
        )

    def jax_modes_amp_phase(
        self, modes: Optional[Sequence] = None, return_tf: bool = False
    ) -> Callable:
        """A pure JAX function computing :meth:`predict_modes_amp_phase`.

        Returns ``f(intrinsic, total_mass, frequencies, distance_mpc=1.0)``
        with the same shapes and conventions as
        :meth:`predict_modes_amp_phase`, for fixed ``modes`` and
        ``return_tf``. A single waveform is the batch ``N = 1``, i.e.
        ``intrinsic`` of shape ``(1, 5)``. Wrap it in :func:`jax.jit`;
        every new input shape compiles again. Requires the ``jax`` extra,
        and enables ``jax_enable_x64``, without which the regressors do
        not work.
        """
        surrogate = self.batched_surrogate(modes, backend="jax")

        def predict(intrinsic, total_mass, frequencies, distance_mpc=1.0):
            return surrogate(
                intrinsic, total_mass, frequencies, distance_mpc, return_tf=return_tf
            )

        return predict

    def _hpc_waveform_per_mode(
        self,
        frequencies: np.ndarray,
        params: ParametersWithExtrinsic,
        inclination: float,
        use_pn: bool,
    ) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        r"""Per-mode Cartesian components of :math:`h_+` and :math:`h_\times`.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies at which to evaluate, in Hz.
        params : ParametersWithExtrinsic
            Source parameters.
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
        Ylm_real, Ylm_imag, Ylm_real_mneg, Ylm_imag_mneg = self._compute_Ylm_modes(
            modes=self.modes,
            phi=0.0,
            iota=inclination,
        )

        amp_arr, phase_arr = self._mode_amplitudes_and_phases(
            frequencies=frequencies,
            params=params,
            source="post_newtonian" if use_pn else "surrogate",
        )
        cosphi_arr = np.cos(phase_arr)
        sinphi_arr = np.sin(phase_arr)
        coeffs = _build_mode_coeffs(
            self.modes,
            list(range(len(self.modes))),
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
        :mod:`~mlgw_bns.pn_modes`, in their own (TaylorF2) reference for
        time and phase.

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

    def teob_modes_amp_phase(
        self,
        frequencies: np.ndarray,
        params: ParametersWithExtrinsic,
    ) -> dict[Mode, tuple[np.ndarray, np.ndarray]]:
        r"""Amplitude and phase of every mode from the underlying EOB code.

        The counterpart of :meth:`ModeModel.predict_amplitude_phase` for
        the ground truth: one TEOBResumS call for all of :attr:`modes`,
        in the same units, with the phases referenced to the merger as the
        surrogate's are (the tangent to the (2,2) phase at the top of the
        trained band, see :meth:`ModeModel.merger_reference`) and the
        extrinsic ``coalescence_phase`` and ``merger_time`` applied.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies at which to evaluate the modes, in Hz.
        params : ParametersWithExtrinsic
            Source parameters.

        Returns
        -------
        dict[Mode, tuple[np.ndarray, np.ndarray]]
            ``mode -> (amplitude, phase)``.
        """
        dataset = self.dataset
        # Use a shallow copy of the dataset whose total_mass is set to the
        # requested total mass, so that the EOB generator interprets the
        # natural-unit frequencies consistently.
        dataset_for_teob = copy.copy(dataset)
        dataset_for_teob.total_mass = params.total_mass
        params_teob = params.intrinsic(dataset_for_teob)
        f_natural = frequencies * params.mass_sum_seconds

        # Two more points at the top of the trained band, for the merger
        # reference of the (2,2): the tangent there, as in `merger_reference`.
        reference_model = self.mode_models[Mode(2, 2)]
        f_top = float(
            reference_model.dataset.frequencies[
                reference_model.downsampling_indices.phase_indices[-1]
            ]
        )
        top = np.array([f_top * (1 - 1e-3), f_top])
        grid = np.union1d(f_natural, top)
        modes = list(self.modes)
        generator = self.mode_models[modes[0]].waveform_generator
        waveforms = generator.all_modes_amplitude_phase(params_teob, modes, grid)
        at_top = np.searchsorted(grid, top)
        at_request = np.searchsorted(grid, f_natural)
        phase_22 = waveforms[Mode(2, 2)][2]
        slope = (phase_22[at_top[1]] - phase_22[at_top[0]]) / (top[1] - top[0])
        intercept = phase_22[at_top[1]] - slope * f_top

        eta = params.intrinsic(dataset).eta
        prefactor = dataset.mlgw_bns_prefactor(eta, params.total_mass) / params.distance_mpc
        result = {}
        for mode in modes:
            _, amp, phase = waveforms[mode]
            result[mode] = (
                amp[at_request] * prefactor,
                phase[at_request]
                - slope * f_natural
                - mode.m / 2 * intercept
                + mode.m * params.coalescence_phase
                - 2 * np.pi * params.merger_time * frequencies,
            )
        return result

    def get_teob_modes_dict(
        self,
        frequencies: np.ndarray,
        params: ParametersWithExtrinsic,
        inclination: Optional[float] = None,
    ) -> dict[tuple[int, int], np.ndarray]:
        r"""Per-mode complex contributions from the underlying EOB code.

        The format of :meth:`predict_modes_dict`, from
        :meth:`teob_modes_amp_phase`: directly comparable to the
        surrogate's. Useful as a ground-truth reference when validating it.

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

        Ylm_real, Ylm_imag, Ylm_real_mneg, Ylm_imag_mneg = self._compute_Ylm_modes(
            modes=self.modes,
            phi=0.0,
            iota=inclination,
        )
        amp_arr, phase_arr = self._mode_amplitudes_and_phases(
            frequencies=frequencies,
            params=params,
            source="eob",
        )
        cosphi_arr = np.cos(phase_arr)
        sinphi_arr = np.sin(phase_arr)
        coeffs = _build_mode_coeffs(
            self.modes,
            list(range(len(self.modes))),
            Ylm_real,
            Ylm_imag,
            Ylm_real_mneg,
            Ylm_imag_mneg,
        )

        eta = params.intrinsic(self.dataset).eta
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
        inclination: float,
        use_pn: Optional[bool] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""Cartesian components of :math:`h_+` and :math:`h_\times` summed over modes.

        See :meth:`_hpc_waveform_per_mode`; ``use_pn`` must be given.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            ``(h_plus_real, h_plus_imag, h_cross_real, h_cross_imag)``,
            each of shape ``(n_freq,)``.
        """
        assert use_pn is not None, "use_pn must be provided"
        per_mode = self._hpc_waveform_per_mode(frequencies, params, inclination, use_pn)
        return tuple(np.sum(parts, axis=0) for parts in zip(*per_mode.values()))  # type: ignore[return-value]

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
