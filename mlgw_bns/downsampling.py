import logging
from typing import Optional, Tuple

import numpy as np
from scipy import interpolate  # type: ignore
from sortedcontainers import SortedList  # type: ignore


def resample(
    x_ds: np.ndarray, new_x: np.ndarray, y_ds: np.ndarray, degree: int = 3
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
    degree : int, optional
            Degree of the interpolation, by default 3.
            Choose an odd number for this.

    Returns
    -------
    new_y : np.ndarray
        Function evaluated at the coordinates ``new_x``.
    """

    return interpolate.splev(
        new_x, tck=interpolate.splrep(x_ds, y_ds, s=0, k=degree), der=0
    )


def indices_error(
    y: np.ndarray, ypred: np.ndarray, current_indices: SortedList, tol: float
) -> Tuple[list[int], list[float]]:
    """Find new indices to add to the sampling.

    Arguments
    ---------
    y : np.ndarray
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

    arr = np.abs(y - ypred)

    new_indices = []
    errors = []

    for key in range(len(current_indices) - 1):

        i = (
            np.argmax(arr[current_indices[key] : current_indices[key + 1]])
            + current_indices[key]
        )

        err = arr[i]

        if err > tol:
            new_indices.append(i)
            errors.append(err)
    return new_indices, errors


def find_indices(
    x: np.ndarray,
    ys_train: list[np.ndarray],
    tol: float,
    ys_val: Optional[list[np.ndarray]] = None,
    degree: int = 3,
    seeds_number: int = 2,
):
    """Greedily downsample y(x) by making sure that the reconstruction error of each of
    the ys (instances of y(x)) is smaller than tol.

    Args:
        x : np.ndarray
                x array
        ys : np.ndarray
                a list of y arrays
        tol : float, optional
                tolerance for the interpolation.
        ys_val : np.ndarray, optional
                a further list of y arrays to be used for validation.
        deg : int, optional
                degree of the interpolation. Defaults to 3.
        seeds_number : np.ndarray, optional
                number of "seed" indices. Defaults to 2.
                These are placed as equally spaced along the array.

    Returns
    -------
    indices : np.ndarray
            indices which make the interpolation errors smaller than
            the tolerance on the training dataset.
    """

    indices = SortedList(list(np.linspace(0, len(x) - 1, num=seeds_number, dtype=int)))

    err = tol + 1

    done_with_wf = np.zeros(len(ys_train), dtype=bool)

    while not all(done_with_wf):

        for i, y in enumerate(ys_train):
            if done_with_wf[i]:
                continue
            ypred = resample(x[indices], x, y[indices], degree)

            indices_batch, errs = indices_error(y, ypred, indices, tol)

            if len(errs) < 1:
                done_with_wf[i] = True

            else:
                indices.update(set(indices_batch))

                err = min(max(errs), err)

        logging.info(
            f"{len(indices)} indices, error = {err} = {err/tol:.1f} times the tol"
        )

    if ys_val is not None:
        validation = []
        for y_val in ys_val:
            ypred = resample(x[indices], x, y_val[indices], degree)
            validation.append(max(abs(y - ypred)))

        return (validation, list(indices))

    return list(indices)
