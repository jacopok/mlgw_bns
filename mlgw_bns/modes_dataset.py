r"""Training sets of the co-precessing modes surrogate (:class:`~mlgw_bns.model.Model`) at scale.

:meth:`Model.generate <mlgw_bns.model.Model.generate>` makes a training set
in one process and holds it in memory; :class:`ShardedModesDataset` keeps
one on disk, in shards, made by any number of processes on any number of
machines (see :mod:`mlgw_bns.sharding`), and trains models on the first
``N`` waveforms of it a shard at a time. A dataset is a directory::

    config.json         what the waveforms are (ModesDatasetConfig)
    downsampling.npz    the downsampling indices of every mode
    shards/00000.npz    waveforms 0 ... shard_size - 1, and so on
    pca/<n>.npz         the principal components of the first n waveforms

A shard holds, for each waveform, its parameters (the five of
:attr:`ParameterSet.parameter_array
<mlgw_bns.dataset_generation.ParameterSet.parameter_array>`) and the
amplitude (single precision) and phase residuals of every mode at its
downsampling indices, as :meth:`Model._multimode_mode_residuals
<mlgw_bns.model.Model._multimode_mode_residuals>` gives them from one EOB
call: about 36 kB a waveform for the seven modes of ``hom7_big``, and a
second of TEOBResumS. The downsampling indices are trained first, once
(:meth:`ShardedModesDataset.train_downsampling`), and must be the same for
a validation set and the training set it validates
(:meth:`ShardedModesDataset.copy_downsampling`).

Training a model (:meth:`ShardedModesDataset.train`) fits the principal
components on the first ``pca_size`` waveforms (once,
:meth:`ShardedModesDataset.principal_components`), reduces the residuals of
the first ``N`` shard by shard, and fits a regressor of each mode on them
(:meth:`ModeModel.fit_nn <mlgw_bns.mode_model.ModeModel.fit_nn>`), saving
each as it is done: a training interrupted resumes from the modes it had
finished, and from the checkpoint of the one it was training if that is a
:class:`~mlgw_bns.neural_network.JaxMLPNetwork`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional, Sequence

import numpy as np

from .data_management import DownsamplingIndices, ParameterRanges, PrincipalComponentData, Residuals
from .dataset_generation import ParameterSet
from .higher_order_modes import Mode
from .model import Model
from .neural_network import Hyperparameters, KernelRidgeNetwork
from .principal_component_analysis import PrincipalComponentTraining
from .sharding import ShardedStore, save_arrays, write_atomically

#: The modes of ``hom7_big`` (make_hom7_dataset.py).
HOM7_MODES = ((2, 2), (2, 1), (3, 1), (3, 2), (3, 3), (4, 3), (4, 4))


def same_indices(a: DownsamplingIndices, b: DownsamplingIndices) -> bool:
    """Whether two downsamplings keep the same points."""
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def mode_name(mode) -> str:
    """``(l, m)`` as ``"l2_m2"``, as in the file names of a :class:`~mlgw_bns.model.Model`."""
    return f"l{mode[0]}_m{mode[1]}"


@dataclass(frozen=True)
class ModesDatasetConfig:
    """What the waveforms of a :class:`ShardedModesDataset` are, and the
    :class:`~mlgw_bns.model.Model` they train.

    Parameters
    ----------
    n_binaries : int
        How many; a dataset whose last shard is full can be grown.
    shard_size : int
        Waveforms a shard.
    seed : int
        The parameters of shard ``k`` are drawn from the seed sequence
        ``(seed, k)``: datasets with different seeds are independent.
    modes, initial_frequency_hz, srate_hz, reference_amplitude, parameter_ranges
        Of the :class:`~mlgw_bns.model.Model`; the defaults are those of
        ``hom7_big`` (make_hom7_dataset.py).
    """

    n_binaries: int
    shard_size: int = 1024
    seed: int = 1
    modes: tuple = HOM7_MODES
    initial_frequency_hz: float = 5.0
    srate_hz: float = 4096.0
    reference_amplitude: bool = True
    parameter_ranges: ParameterRanges = field(
        default_factory=lambda: ParameterRanges(lambda1_range=(5.0, 12000.0), lambda2_range=(5.0, 12000.0))
    )

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "ModesDatasetConfig":
        values = json.loads(text)
        values["modes"] = tuple(tuple(mode) for mode in values["modes"])
        values["parameter_ranges"] = ParameterRanges(
            **{key: tuple(value) for key, value in values["parameter_ranges"].items()}
        )
        return cls(**values)

    def model(self, filename: Optional[str] = None, **kwargs) -> Model:
        """A :class:`~mlgw_bns.model.Model` of these settings (not trained);
        ``kwargs`` of its modes' :class:`~mlgw_bns.mode_model.ModeModel`,
        such as ``nn_kind``."""
        return Model(
            modes=[Mode(*mode) for mode in self.modes],
            filename=filename or "",
            initial_frequency_hz=self.initial_frequency_hz,
            srate_hz=self.srate_hz,
            reference_amplitude=self.reference_amplitude,
            parameter_ranges=self.parameter_ranges,
            **kwargs,
        )

    def same_waveforms(self, other: "ModesDatasetConfig") -> bool:
        """Whether ``other`` has waveforms of the same model (if not the
        same ones): what a validation set and its training set share."""
        return dataclasses.replace(other, n_binaries=self.n_binaries, shard_size=self.shard_size,
                                   seed=self.seed) == self


class ShardedModesDataset(ShardedStore):
    """A training (or validation) set of a :class:`~mlgw_bns.model.Model` on
    disk; see the module docstring. Open one with
    ``ShardedModesDataset(directory)``, make one with :meth:`create`."""

    config_class = ModesDatasetConfig

    @property
    def modes(self) -> list:
        return [Mode(*mode) for mode in self.config.modes]

    def model(self, filename: Optional[str] = None, **kwargs) -> Model:
        """A :class:`~mlgw_bns.model.Model` of this dataset (see
        :meth:`ModesDatasetConfig.model`), with its downsampling indices if
        they are there."""
        model = self.config.model(filename, **kwargs)
        if os.path.exists(self.downsampling_path):
            for mode, indices in self.downsampling().items():
                model.mode_models[mode].downsampling_indices = indices
        return model

    # -- downsampling ---------------------------------------------------------

    @property
    def downsampling_path(self) -> str:
        return os.path.join(self.directory, "downsampling.npz")

    def downsampling(self) -> dict:
        """``mode -> DownsamplingIndices``."""
        with np.load(self.downsampling_path) as data:
            return {
                mode: DownsamplingIndices(
                    list(data[f"{mode_name(mode)}_amplitude"]), list(data[f"{mode_name(mode)}_phase"])
                )
                for mode in self.modes
            }

    def _save_downsampling(self, indices: dict) -> None:
        arrays = {}
        for mode, value in indices.items():
            arrays[f"{mode_name(mode)}_amplitude"] = np.asarray(value.amplitude_indices, dtype=np.int64)
            arrays[f"{mode_name(mode)}_phase"] = np.asarray(value.phase_indices, dtype=np.int64)
        save_arrays(self.downsampling_path, **arrays)

    def train_downsampling(self, size: int = 64, n_jobs: int = 1) -> None:
        """Train the downsampling indices of every mode on ``size`` waveforms
        each (:meth:`Model.train_downsampling
        <mlgw_bns.model.Model.train_downsampling>`), unless they are there."""
        if os.path.exists(self.downsampling_path):
            logging.info("%s: the downsampling is there", self.directory)
            return
        start = time.time()
        model = self.config.model()
        model.train_downsampling(size, n_jobs=n_jobs)
        self._save_downsampling({mode: model.mode_models[mode].downsampling_indices for mode in self.modes})
        logging.info(
            "%s: downsampling of %i modes on %i waveforms each in %.0f s: %s", self.directory,
            len(self.modes), size, time.time() - start,
            ", ".join(f"{mode_name(m)} {i.amp_length}+{i.phi_length}" for m, i in self.downsampling().items()),
        )

    def copy_downsampling(self, other: "ShardedModesDataset") -> None:
        """Use the downsampling indices of ``other`` (of waveforms of the
        same model): those of the training set, for a validation set."""
        if not self.config.same_waveforms(other.config):
            raise ValueError(f"{other.directory} holds waveforms of another model")
        if os.path.exists(self.downsampling_path):
            mine, theirs = self.downsampling(), other.downsampling()
            if not all(same_indices(mine[m], theirs[m]) for m in self.modes):
                raise ValueError(f"{self.directory} has other downsampling indices than {other.directory}")
            return
        self._save_downsampling(other.downsampling())

    # -- making shards --------------------------------------------------------

    def parameters(self, index: int) -> list:
        """The :class:`~mlgw_bns.dataset_generation.WaveformParameters` of
        shard ``index``."""
        model = self.config.model()
        generator = model.dataset.make_parameter_generator(seed=self.seed(index))
        return [next(generator) for _ in range(self.shard_size(index))]

    def generate(
        self, n_jobs: int = -1, max_seconds: Optional[float] = None, stale_seconds: float = 600.0,
    ) -> bool:
        """Make the shards not yet there (see
        :meth:`~mlgw_bns.sharding.ShardedStore.work_through`), each with
        ``n_jobs`` processes; whether all are there. The downsampling
        indices must be there."""
        if not os.path.exists(self.downsampling_path):
            raise FileNotFoundError(f"{self.downsampling_path}: train or copy the downsampling first")
        return super().generate(n_jobs, max_seconds, stale_seconds)

    def _make(self, index: int, n_jobs: int) -> tuple:
        model = self.model()
        dataset = model.dataset
        params = self.parameters(index)
        parameters, amplitudes, phases = model._multimode_mode_residuals(
            params,
            dataset.frequencies,
            {mode: model.mode_models[mode].downsampling_indices for mode in self.modes},
            {mode: model.mode_models[mode].dataset.amplitude_reference_parameters for mode in self.modes},
            progress_desc=f"shard {index}",
            n_jobs=n_jobs,
            keep_invalid=True,
        )
        arrays = {"parameters": parameters}
        valid = np.ones(len(params), bool)
        for mode in self.modes:
            arrays[f"amplitude_{mode_name(mode)}"] = amplitudes[mode].astype(np.float32)
            arrays[f"phase_{mode_name(mode)}"] = phases[mode]
            valid &= np.all(np.isfinite(amplitudes[mode]), axis=1) & np.all(np.isfinite(phases[mode]), axis=1)
        arrays["valid"] = valid
        return arrays, {}

    # -- reading --------------------------------------------------------------

    def residuals(
        self, mode: Mode, n: Optional[int] = None, select: Optional[Callable] = None,
    ) -> Iterator[tuple]:
        """``(parameters, Residuals)`` of ``mode`` for the first ``n`` valid
        waveforms (all, by default), a shard at a time."""
        name = mode_name(mode)
        for chunk in self.chunks(n, ("parameters", f"amplitude_{name}", f"phase_{name}"), select=select):
            yield chunk["parameters"], Residuals(chunk[f"amplitude_{name}"], chunk[f"phase_{name}"])

    def load_residuals(self, n: Optional[int] = None) -> tuple:
        """The parameters ``(n, 5)`` of the first ``n`` valid waveforms and
        ``mode -> Residuals`` of every mode."""
        data = self.load(n)
        residuals = {
            mode: Residuals(data[f"amplitude_{mode_name(mode)}"], data[f"phase_{mode_name(mode)}"])
            for mode in self.modes
        }
        return data["parameters"], residuals

    # -- training -------------------------------------------------------------

    def principal_components(self, n: int, n_components: int = 30) -> dict:
        """``mode -> PrincipalComponentData`` of the first ``n`` valid
        waveforms, fitted once (as :meth:`ModeModel.generate
        <mlgw_bns.mode_model.ModeModel.generate>` does) and kept in
        ``pca/<n>.npz``."""
        filename = os.path.join(self.directory, "pca", f"{n}.npz" if n_components == 30 else f"{n}_{n_components}.npz")
        if not os.path.exists(filename):
            start = time.time()
            model = self.model()
            arrays = {}
            for mode in self.modes:
                residuals = Residuals(*(
                    np.concatenate(parts) for parts in zip(*(
                        (r.amplitude_residuals, r.phase_residuals) for _, r in self.residuals(mode, n)
                    ))
                ))
                mode_model = model.mode_models[mode]
                data = PrincipalComponentTraining(
                    mode_model.dataset, mode_model.downsampling_indices, n_components
                ).train_on(residuals)
                for key, value in dataclasses.asdict(data).items():
                    arrays[f"{mode_name(mode)}_{key}"] = value
            save_arrays(filename, **arrays)
            logging.info("%s: principal components of %i waveforms in %.0f s", self.directory, n, time.time() - start)
        with np.load(filename) as data:
            names = [f.name for f in dataclasses.fields(PrincipalComponentData)]
            return {
                mode: PrincipalComponentData(*(data[f"{mode_name(mode)}_{key}"] for key in names))
                for mode in self.modes
            }

    def reduced(self, model: Model, mode: Mode, n: int) -> tuple:
        """The parameters ``(n, 5)`` of the first ``n`` valid waveforms, the
        principal components of their residuals of ``mode`` in the
        ``model``'s basis, and their power weights (``None`` for a mode not
        weighted; see :meth:`ModeModel._training_power_weights
        <mlgw_bns.mode_model.ModeModel._training_power_weights>`), reduced a
        shard at a time."""
        mode_model = model.mode_models[mode]
        parameters, reduced, weights = [], [], []
        for chunk_parameters, residuals in self.residuals(mode, n):
            parameters.append(chunk_parameters)
            reduced.append(mode_model.pca_model.reduce_data(residuals.combined, mode_model.pca_data))
            weights.append(mode_model._training_power_weights(residuals.amplitude_residuals))
        return (
            np.concatenate(parameters),
            np.concatenate(reduced),
            None if weights[0] is None else np.concatenate(weights),
        )

    def train(
        self,
        n: int,
        filename: str,
        nn_kind: type = KernelRidgeNetwork,
        hyperparameters: Optional[Callable[[Mode, int], Hyperparameters]] = None,
        pca_size: Optional[int] = None,
        modes: Optional[Sequence] = None,
        checkpoint: Optional[str] = None,
    ) -> Optional[Model]:
        """Train a :class:`~mlgw_bns.model.Model` on the first ``n`` valid
        waveforms and save it as ``filename`` (without its training data,
        like the packaged one).

        Parameters
        ----------
        n : int
            Training waveforms.
        filename : str
            Base filename of the model (its files are ``{filename}_l2_m2_nn.pkl``
            and so on).
        nn_kind : type
            Of the regressors.
        hyperparameters : callable, optional
            ``(mode, n) -> Hyperparameters``; by default those of
            :meth:`ModeModel.default_hyperparameters
            <mlgw_bns.mode_model.ModeModel.default_hyperparameters>`.
        pca_size : int, optional
            The principal components are those of the first ``pca_size``
            waveforms (:meth:`principal_components`); ``n`` by default.
        modes : sequence, optional
            Train only the regressors of these modes (each saved as it is
            done, skipped if it is there): for trainings split in jobs.
            The model is saved once those of all the modes are there.
        checkpoint : str, optional
            Base filename of the checkpoints of the training of each mode
            (``{checkpoint}_l2_m2.joblib``), for the regressors that keep
            one (:class:`~mlgw_bns.neural_network.JaxMLPNetwork`).

        Returns
        -------
        Model, optional
            The model, once all of it is trained.
        """
        for path in (filename, checkpoint):
            if path is not None:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        model = self.model(filename, nn_kind=nn_kind)
        pca = self.principal_components(pca_size or n)
        for mode in self.modes:
            model.mode_models[mode].pca_data = pca[mode]
        for mode in self.modes if modes is None else [Mode(*m) for m in modes]:
            mode_model = model.mode_models[mode]
            if os.path.exists(mode_model.filename_nn):
                logging.info("%s: the regressor of %s is there", filename, mode)
                continue
            start = time.time()
            parameters, reduced, weights = self.reduced(model, mode, n)
            hyper = (
                hyperparameters(mode, n) if hyperparameters is not None
                else mode_model.default_hyperparameters(n)
            )
            logging.info("%s: training %s on %i waveforms (%.0f s to read them)",
                         filename, mode, n, time.time() - start)
            state = None if checkpoint is None else f"{checkpoint}_{mode_name(mode)}.joblib"
            nn = mode_model.fit_nn(hyper, parameters, reduced, weights, checkpoint=state)
            write_atomically(mode_model.filename_nn, lambda file: nn.save(file))  # type: ignore[arg-type]
            if state is not None and os.path.exists(state):
                os.remove(state)  # the regressor saved is all of it that is needed
            logging.info("%s: %s trained in %.0f s", filename, mode, time.time() - start)
        if not all(os.path.exists(model.mode_models[m].filename_nn) for m in self.modes):
            return None
        parameter_set = None
        for mode in self.modes:
            mode_model = model.mode_models[mode]
            mode_model.nn = mode_model.nn_kind.from_file(mode_model.filename_nn)
            if parameter_set is None:
                parameter_set = ParameterSet(self.load(n, ("parameters",))["parameters"])
            mode_model.training_parameters = parameter_set
            if os.path.exists(mode_model.filename_arrays):
                os.remove(mode_model.filename_arrays)
            # (the regressor is there already)
            mode_model.save_metadata()
            mode_model.save_arrays(include_training_data=False)
        model._batched_cache.clear()
        return model


def stored_size(model: Model) -> dict:
    """Bytes on disk of the regressors and of the rest (downsampling, principal
    components, metadata) of a saved model, and their sum."""
    regressors = sum(os.path.getsize(model.mode_models[m].filename_nn) for m in model.modes)
    rest = sum(
        os.path.getsize(path) for m in model.modes
        for path in (model.mode_models[m].filename_arrays, model.mode_models[m].filename_metadata)
    )
    return {"regressors": regressors, "rest": rest, "total": regressors + rest}


def load_model(dataset: ShardedModesDataset, filename: str) -> Model:
    """The model saved as ``filename`` by :meth:`ShardedModesDataset.train`."""
    model = dataset.config.model(filename)
    model.load()  # the metadata name the kind of the regressors
    return model

