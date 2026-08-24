"""Functionality for the generation of a set of indices 
which accurately represent a waveform. 

The default implementation is a greedy one, as defined in 
:class:`GreedyDownsamplingTraining`. 

Alternative implementations include :class:`RDPDownsamplingTraining`
(Ramer-Douglas-Peucker, faster) and :class:`AdaptiveEnvelopeDownsampling`.

To provide an alternate method, just subclass
:class:`DownsamplingTraining`.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from sortedcontainers import SortedList  # type: ignore

import scipy.interpolate as interpolate
from scipy.interpolate import PchipInterpolator
from .data_management import DownsamplingIndices
from .dataset_generation import Dataset

try:
    from rdp import rdp as rdp_simplify
    _RDP_AVAILABLE = True
except ImportError:
    _RDP_AVAILABLE = False

    def _rdp_numpy(points: np.ndarray, epsilon: float) -> np.ndarray:
        """Pure NumPy implementation of Ramer-Douglas-Peucker algorithm."""
        if len(points) <= 2:
            return points
        # Find point with max perpendicular distance from line between first and last
        start, end = points[0], points[-1]
        line_vec = end - start
        line_len_sq = np.dot(line_vec, line_vec) + 1e-20
        # Perpendicular distance: |cross| / |line|
        d = np.abs(
            (points[:, 0] - start[0]) * (end[1] - start[1])
            - (points[:, 1] - start[1]) * (end[0] - start[0])
        ) / np.sqrt(line_len_sq)
        d[0] = 0
        d[-1] = 0
        max_idx = np.argmax(d)
        max_dist = d[max_idx]
        if max_dist <= epsilon:
            return np.array([start, end])
        left = _rdp_numpy(points[: max_idx + 1], epsilon)
        right = _rdp_numpy(points[max_idx:], epsilon)
        return np.vstack([left[:-1], right])


class DownsamplingTraining(ABC):
    """Selection of the downsampling indices.

    Parameters
    ----------
    dataset : Dataset
            dataset to which to refer for the generation
            of training waveforms for the downsampling.
    degree : int
            degree for the interpolation.
            Defaults to 3.
    tol : float
            Tolerance for the interpolation error.
            Defaults to ``1e-4``. Used for both amp and phi when tol_amp/tol_phi
            are not specified.
    tol_amp : float, optional
            Tolerance for amplitude downsampling. If None, uses tol.
    tol_phi : float, optional
            Tolerance for phase downsampling. If None, uses tol.
    """

    degree: int = 3

    def __init__(
        self,
        dataset: Dataset,
        tol: float = 1e-4,
        tol_amp: Optional[float] = 8e-4, # 8e-4
        tol_phi: Optional[float] = 5e-4, # 5e-4
        n_jobs: int = 16,
    ):
        self.dataset = dataset
        self.tol = tol
        self.tol_amp = tol_amp if tol_amp is not None else tol
        self.tol_phi = tol_phi if tol_phi is not None else tol
        self.n_jobs = n_jobs

    @abstractmethod
    def train(self, training_dataset_size: int) -> DownsamplingIndices:
        """Calcalate downsampling with a generic algoritm,
        training on a dataset with a given sizes."""

    def validate_downsampling(
        self, training_dataset_size: int, validating_dataset_size: int
    ) -> tuple[list[float], list[float]]:
        r"""Check that the downsampling is working by looking at the
        reconstruction error on a fresh dataset.

        Parameters
        ----------
        training_dataset_size : int
            How many waveforms to train the downsampling on.
        validating_dataset_size : int
            How many waveforms to validate on.

        Returns
        -------
        tuple[list[float], list[float]]
            Amplitude and phase validation errors;
            these are reported as :math:`L_\infty` errors:
            the absolute maximum of the difference.
        """

        amp_indices, phi_indices = self.train(training_dataset_size)

        frequencies, _, residuals = self.dataset.generate_residuals(
            size=validating_dataset_size, n_jobs=self.n_jobs
        )

        amp_residuals, phi_residuals = residuals

        amp_validation = self.validate_indices(
            amp_indices, frequencies, amp_residuals[-validating_dataset_size:]
        )
        phi_validation = self.validate_indices(
            phi_indices, frequencies, phi_residuals[-validating_dataset_size:]
        )

        return amp_validation, phi_validation

    @classmethod
    def resample(
        cls, x_ds: np.ndarray, new_x: np.ndarray, y_ds: np.ndarray
    ) -> np.ndarray:
        """Resample a function :math:`y(x)` from its values
        at certain points :math:`y_{ds} = y(x_{ds})`.

        Parameters
        ----------
        x_ds : np.ndarray
                Old, sparse :math:`x` values.
        new_x : np.ndarray
                New :math:`x` coordinates at which to evaluate the function.
        y_ds : np.ndarrays
                Old, sparse :math:`y` values.

        Returns
        -------
        new_y : np.ndarray
            Function evaluated at the coordinates ``new_x``.
        """

        if x_ds.shape != y_ds.shape:
            raise ValueError(
                f"""Shape mismatch in the downsampling arrays! 
                The shape of x_ds is {x_ds.shape} while the shape of y_ds is {y_ds.shape}."""
            )

        # Time all 4 interpolation methods
        # t0 = time.perf_counter()
        spline_result = interpolate.CubicSpline(x_ds, y_ds)(new_x)
        # t_spline = time.perf_counter() - t0

        # t0 = time.perf_counter()
        # _ = interpolate.Akima1DInterpolator(x_ds, y_ds)(new_x)
        # t_akima = time.perf_counter() - t0

        # t0 = time.perf_counter()
        # linear_result = np.interp(new_x, x_ds, y_ds)
        # t_linear = time.perf_counter() - t0

        # t0 = time.perf_counter()
        # _ = interpolate.PchipInterpolator(x_ds, y_ds)(new_x)
        # t_pchip = time.perf_counter() - t0

        # print(
        #     f"Interpolation times (s): spline={t_spline}, akima={t_akima}, linear={t_linear}, pchip={t_pchip}",
        # )
        
        return spline_result      

    def validate_indices(
        self, indices: list[int], x_val: np.ndarray, ys_val: list[np.ndarray]
    ) -> list[float]:

        validation = []
        for y_val in ys_val:
            ypred = self.resample(x_val[indices], x_val, y_val[indices])
            validation.append(max(abs(y_val - ypred)))

        return validation


class GreedyDownsamplingTraining(DownsamplingTraining):
    def indices_error(
        self,
        ytrue: np.ndarray,
        ypred: np.ndarray,
        current_indices: SortedList,
        tol: Optional[float] = None,
    ) -> tuple[list[int], list[float]]:
        """Find new indices to add to the sampling.

        Arguments
        ---------
        ytrue : np.ndarray
                True values of y.
        ypred : np.ndarray
                Predicted values of y through interpolation.
                The algorithm minimizes the difference ``abs(y - ypred)``.
        current_indices : SortedList
                Indices to which the algorithm should add.
        tol : float
                Tolerance for the reconstruction error ---
                new indices are not added if the reconstruction error is below this value.

        Returns
        -------
        new_indices : list[int]
                Indices to insert among the current ones.
        errors : list[float]
                Errors (``abs(y - y_pred)``) at the points where the
                algorithm inserted the new indices.
        """
        arr =  np.abs(ytrue - ypred)
        
        _tol = tol if tol is not None else self.tol
        new_indices = []
        errors = []

        for key in range(len(current_indices) - 1):

            i = (
                np.argmax(arr[current_indices[key] : current_indices[key + 1]])
                + current_indices[key]
            )

            err = arr[i]

            if err > _tol:
                new_indices.append(i)
                errors.append(err)
        return new_indices, errors

    def find_indices(
        self,
        x_train: np.ndarray,
        ys_train: list[np.ndarray],
        seeds_number: int = 5,
        tol: Optional[float] = None,
    ) -> list[int]:
        """Greedily downsample y(x) by making sure that the reconstruction error of each of
        the ys (instances of y(x)) is smaller than tol.

        Arguments
        ---------
            x_train : np.ndarray
                    x array
            ys : np.ndarray
                    a list of y arrays
            seeds_number : np.ndarray, optional
                    number of "seed" indices. Defaults to 4.
                    These are placed as equally spaced along the array.
                    Note: this should always be larger than the degree
                    for the interpolation.

        Returns
        -------
        indices : np.ndarray
                indices which make the interpolation errors smaller than
                the tolerance on the training dataset.
        """

        _tol = tol if tol is not None else self.tol
        indices = SortedList(
            list(np.linspace(0, len(x_train) - 1, num=seeds_number, dtype=int))
        )

        err = _tol + 1

        done_with_wf = np.zeros(len(ys_train), dtype=bool)

        logging.info("Starting interpolation (tol=%g)", _tol)
        while not all(done_with_wf):

            for i, y in enumerate(ys_train):
                if done_with_wf[i]:
                    continue
                ypred = self.resample(x_train[indices], x_train, y[indices])

                indices_batch, errs = self.indices_error(
                    y, ypred, indices, tol=_tol
                )

                if len(errs) < 1:
                    done_with_wf[i] = True

                else:
                    indices.update(set(indices_batch))

                    err = min(max(errs), err)

            logging.info(
                "%i indices, error = %f = %f times the tol",
                len(indices),
                err,
                err / _tol,
            )

        return list(indices)

    def train(self, training_dataset_size: int) -> DownsamplingIndices:
        """Compute a close-to-optimal set of indices at which to sample
        waveforms, so that the reconstruction stays below a certain tolerance.

        Parameters
        ----------
        training_dataset_size : int
            Number of waveforms to generate and with which to train.

        Returns
        -------
        tuple[list[int], list[int]]
                Indices for amplitude and phase, respectively.
        """

        generator = self.dataset.make_parameter_generator()
        # For modes (2,1) and (3,3), oversample params to compensate for amp_teob<=0 discards
        oversample = (
            1.5
            if (
                self.dataset.current_mode is not None
                and (self.dataset.current_mode.l, self.dataset.current_mode.m) in ((2, 1), (3, 3))
            )
            else 1.0
        )
        n_params = int(training_dataset_size * oversample)
        param_set = self.dataset.parameter_set_cls.from_parameter_generator(
            generator, n_params
        )

        waveforms, _ = self.dataset.generate_waveforms_from_params(
            param_set, n_jobs=self.n_jobs
        )
        frequencies = self.dataset.frequencies

        # Use first training_dataset_size if we got more (from oversample)
        n_use = min(training_dataset_size, len(waveforms.amplitudes))
        if n_use < len(waveforms.amplitudes):
            waveforms = type(waveforms)(
                waveforms.amplitudes[:n_use], waveforms.phases[:n_use]
            )

        print(f"GreedyDown_tol_amp={self.tol_amp}")
        print(f"GreedyDown_tol_phi={self.tol_phi}")

        amp_indices = self.find_indices(
            frequencies, list(waveforms.amplitudes), tol=self.tol_amp
        )
        phi_indices = self.find_indices(
            frequencies, list(waveforms.phases), tol=self.tol_phi
        )

        return DownsamplingIndices(amp_indices, phi_indices)


class GreedyDownsamplingTrainingWithResiduals(GreedyDownsamplingTraining):
    def train(self, training_dataset_size: int) -> DownsamplingIndices:
        """Compute a close-to-optimal set of indices at which to sample
        waveforms, so that the reconstruction stays below a certain tolerance.

        Parameters
        ----------
        training_dataset_size : int
            Number of waveforms to generate and with which to train.

        Returns
        -------
        tuple[list[int], list[int]]
                Indices for amplitude and phase, respectively.
        """

        frequencies, _, residuals = self.dataset.generate_residuals(
            size=training_dataset_size, n_jobs=self.n_jobs
        )
        amp_residuals, phi_residuals = residuals

        amp_indices = self.find_indices(
            frequencies,
            amp_residuals[:training_dataset_size],
            tol=self.tol_amp,
        )
        phi_indices = self.find_indices(
            frequencies,
            phi_residuals[:training_dataset_size],
            tol=self.tol_phi,
        )

        return DownsamplingIndices(amp_indices, phi_indices)


class RDPDownsamplingTraining(DownsamplingTraining):
    """Downsampling using the Ramer-Douglas-Peucker (RDP) algorithm.

    RDP is optimal for single curves with linear interpolation: it finds
    the minimum number of points for a given max error tolerance.
    Much faster than greedy for large frequency grids.

    Parameters
    ----------
    dataset : Dataset
        Dataset for waveform generation.
    tol : float
        Tolerance for the interpolation error. Defaults to 1e-5.
    grid_step : int
        Subsample the frequency grid by this factor for RDP (speeds up
        training on very long arrays). Defaults to 1 (no subsampling).
    max_waveforms : Optional[int]
        Use at most this many waveforms for the index union. If None,
        use all. Subsampling speeds up training. Defaults to 200.
    """

    def __init__(
        self,
        dataset: Dataset,
        tol: float = 8e-2,
        tol_amp: Optional[float] = 1e-1,
        tol_phi: Optional[float] = 5e-2,
        grid_step: int = 10,
        max_waveforms: Optional[int] = 200,
        n_jobs: int = 16,
    ):
        super().__init__(
            dataset, tol, tol_amp=tol_amp, tol_phi=tol_phi, n_jobs=n_jobs
        )
        self.grid_step = max(1, grid_step)
        self.max_waveforms = max_waveforms

    def _rdp_indices_single(
        self, x: np.ndarray, y: np.ndarray, tol: float
    ) -> np.ndarray:
        """Run RDP on a single curve and return indices."""
        # Use (index, y) so output indices are preserved in first column
        points = np.column_stack((np.arange(len(x), dtype=float), y.astype(float)))
        if _RDP_AVAILABLE:
            simplified = rdp_simplify(points, epsilon=tol)
        else:
            simplified = _rdp_numpy(points, epsilon=tol)
        return np.unique(simplified[:, 0].astype(int)).tolist()

    def find_indices(
        self,
        x_train: np.ndarray,
        ys_train: list[np.ndarray],
        tol: Optional[float] = None,
    ) -> list[int]:
        """Find indices using RDP on each waveform, then union."""
        _tol = tol if tol is not None else self.tol
        n_waveforms = len(ys_train)
        if self.max_waveforms is not None and n_waveforms > self.max_waveforms:
            indices_subset = np.linspace(
                0, n_waveforms - 1, num=min(self.max_waveforms, n_waveforms), dtype=int
            )
            ys_subset = [ys_train[i] for i in indices_subset]
        else:
            ys_subset = ys_train

        # Optionally subsample the grid for speed
        step = self.grid_step
        if step > 1:
            x_sub = x_train[::step]
            ys_sub = [y[::step] for y in ys_subset]
        else:
            x_sub = x_train
            ys_sub = ys_subset

        def process_one(y):
            indices = self._rdp_indices_single(x_sub, y, _tol)
            if step > 1:
                indices = [idx * step for idx in indices]
            return indices

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_one)(y) for y in ys_sub
        )

        all_indices: set[int] = set()
        for indices in results:
            all_indices.update(indices)

        result = sorted(all_indices)
        # Ensure first and last are always included
        result = sorted(set(result) | {0, len(x_train) - 1})
        print(
            f"RDP downsampling: {len(result)} indices from {len(ys_subset)} waveforms (tol={_tol})",
            len(result),
            len(ys_subset),
            _tol,
        )
        return result

    def train(self, training_dataset_size: int) -> DownsamplingIndices:
        """Compute downsampling indices using RDP."""
        generator = self.dataset.make_parameter_generator()
        # For modes (2,1) and (3,3), oversample params to compensate for amp_teob<=0 discards
        oversample = (
            1.5
            if (
                self.dataset.current_mode is not None
                and (self.dataset.current_mode.l, self.dataset.current_mode.m) in ((2, 1), (3, 3))
            )
            else 1.0
        )
        n_params = int(training_dataset_size * oversample)
        param_set = self.dataset.parameter_set_cls.from_parameter_generator(
            generator, n_params
        )
        waveforms, _ = self.dataset.generate_waveforms_from_params(
            param_set, n_jobs=self.n_jobs
        )
        frequencies = self.dataset.frequencies

        # Use first training_dataset_size if we got more (from oversample)
        n_use = min(training_dataset_size, len(waveforms.amplitudes))
        if n_use < len(waveforms.amplitudes):
            waveforms = type(waveforms)(
                waveforms.amplitudes[:n_use], waveforms.phases[:n_use]
            )

        print(f"RDP downsampling: frequencies_length={len(frequencies)}")
        print(f"tol_amp={self.tol_amp}")
        print(f"tol_phi={self.tol_phi}")

        amp_indices = self.find_indices(
            frequencies, list(waveforms.amplitudes), tol=self.tol_amp
        )
        phi_indices = self.find_indices(
            frequencies, list(waveforms.phases), tol=self.tol_phi
        )

        return DownsamplingIndices(amp_indices, phi_indices)


class RDPDownsamplingTrainingWithResiduals(RDPDownsamplingTraining):
    """RDP downsampling trained on residuals instead of raw waveforms."""

    def train(self, training_dataset_size: int) -> DownsamplingIndices:
        """Compute downsampling indices using RDP on residuals."""
        frequencies, _, residuals = self.dataset.generate_residuals(
            size=training_dataset_size, n_jobs=self.n_jobs
        )
        amp_residuals, phi_residuals = residuals
        amp_residuals = list(amp_residuals[:training_dataset_size])
        phi_residuals = list(phi_residuals[:training_dataset_size])

        logging.info("RDP downsampling: frequencies_length=%i", len(frequencies))

        amp_indices = self.find_indices(
            frequencies, amp_residuals, tol=self.tol_amp
        )
        phi_indices = self.find_indices(
            frequencies, phi_residuals, tol=self.tol_phi
        )

        return DownsamplingIndices(amp_indices, phi_indices)
