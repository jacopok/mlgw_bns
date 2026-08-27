r"""Low-discrepancy sampling of the training parameters.

The training parameters are drawn i.i.d. uniform over the five-dimensional
box. That is the right thing to do if the goal is an unbiased Monte Carlo
estimate, but it is not what a regression wants: independent draws leave
clumps and gaps, and the worst-case error of a smooth interpolant is set
by the largest gap. The discrepancy of :math:`n` i.i.d. points falls like
:math:`n^{-1/2}`, while a scrambled Sobol sequence falls like
:math:`(\log n)^d / n`.

Under a budget of fewer than ten thousand waveforms, and with a target
function as smooth as the EOB-PN residual is in these five parameters,
that difference is worth measuring: it costs nothing at generation time
and nothing at evaluation time.

The sequence is scrambled (``scipy.stats.qmc.Sobol(scramble=True)``) so
that it remains a valid random sample --- a validation set drawn the same
way is still unbiased --- and drawn in powers of two, which is where
Sobol's balance properties hold exactly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import qmc  # type: ignore

from mlgw_bns.data_management import ParameterRanges
from mlgw_bns.dataset_generation import (
    Dataset,
    ParameterGenerator,
    WaveformParameters,
)


class SobolParameterGenerator(ParameterGenerator):
    """Parameters from a scrambled Sobol sequence over the parameter box."""

    def __init__(
        self,
        dataset: Dataset,
        parameter_ranges: ParameterRanges,
        seed: Optional[int] = None,
        n_points: int = 8192,
    ):
        super().__init__(dataset=dataset, seed=seed)
        self.lower = np.array(
            [
                parameter_ranges.q_range[0],
                parameter_ranges.lambda1_range[0],
                parameter_ranges.lambda2_range[0],
                parameter_ranges.chi1_range[0],
                parameter_ranges.chi2_range[0],
            ]
        )
        self.upper = np.array(
            [
                parameter_ranges.q_range[1],
                parameter_ranges.lambda1_range[1],
                parameter_ranges.lambda2_range[1],
                parameter_ranges.chi1_range[1],
                parameter_ranges.chi2_range[1],
            ]
        )
        # Drawn all at once: Sobol's balance properties hold for the
        # sequence as a whole, in powers of two, not for an arbitrary
        # prefix taken one point at a time.
        engine = qmc.Sobol(d=5, scramble=True, seed=seed)
        self.points = qmc.scale(
            engine.random(_next_power_of_two(n_points)), self.lower, self.upper
        )
        self.index = 0

    def __next__(self) -> WaveformParameters:
        if self.index >= len(self.points):
            raise StopIteration("Sobol sequence exhausted")
        point = self.points[self.index]
        self.index += 1
        return WaveformParameters(*point, self.dataset)


def _next_power_of_two(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()
