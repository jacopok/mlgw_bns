r"""JAX evaluation of the full waveform, :math:`(h_+, h_\times)`.

This used to be a separate JAX port of the prediction pipeline (adapted
from the port of ``mlgw_bns`` 0.12 written by **Saulo Albuquerque**,
``saulo-albuquerque-phys``). It is now a thin wrapper around the single
numpy/JAX-generic pipeline of :mod:`mlgw_bns.batched`, which evaluates
individual modes for batches of binaries; see there, and
:meth:`Model.jax_modes_amp_phase <mlgw_bns.model.Model.jax_modes_amp_phase>`,
for per-mode output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence

import jax.numpy as jnp

from .batched import mode_polarizations

if TYPE_CHECKING:
    from .model import Model


def model_to_jax_waveform(model: "Model", modes: Optional[Sequence] = None) -> Callable:
    r"""A JAX function reproducing :meth:`Model.predict <mlgw_bns.model.Model.predict>`.

    Returns ``predict(params, frequencies_hz, total_mass, distance_mpc,
    inclination, reference_phase=0.0) -> (h_plus, h_cross)``, where
    ``params`` is ``[q, lambda_1, lambda_2, chi_1, chi_2]`` with shape
    ``(5,)`` or ``(n, 5)``; the other arguments are scalars or shape
    ``(n,)``. The output has shape ``(k,)`` for a single row, ``(n, k)``
    otherwise. Pure JAX: wrap it in :func:`jax.jit`.

    Parameters
    ----------
    model : Model
        A trained model.
    modes : sequence of (l, m), optional
        Modes to sum; defaults to all of ``model.modes``.
    """
    modes = [
        tuple(int(i) for i in lm) for lm in (model.modes if modes is None else modes)
    ]
    predict_modes = model.jax_modes_amp_phase(modes)
    emms = jnp.asarray([m for _, m in modes], dtype=float)

    def predict(
        params,
        frequencies_hz,
        total_mass,
        distance_mpc,
        inclination,
        reference_phase=0.0,
    ):
        params = jnp.atleast_2d(jnp.asarray(params, jnp.float64))
        n_rows = params.shape[0]
        amp, phase = predict_modes(
            params,
            jnp.broadcast_to(jnp.asarray(total_mass, jnp.float64), (n_rows,)),
            frequencies_hz,
            jnp.broadcast_to(jnp.asarray(distance_mpc, jnp.float64), (n_rows,)),
        )
        # `reference_phase` is a coalescence phase: it rotates mode (l, m)
        # by exp(i m phi_c).
        reference_phase = jnp.reshape(
            jnp.asarray(reference_phase, jnp.float64), (-1, 1, 1)
        )
        phase = phase + emms[None, :, None] * reference_phase
        h_plus, h_cross = mode_polarizations(
            amp, phase, modes, inclination, 0.0, xp=jnp
        )
        h_plus, h_cross = h_plus.sum(axis=1), h_cross.sum(axis=1)
        if n_rows == 1:
            return h_plus[0], h_cross[0]
        return h_plus, h_cross

    return predict
