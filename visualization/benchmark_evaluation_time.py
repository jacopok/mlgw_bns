r"""Randomised wall-clock comparison of the surrogate's ``Model.predict``
against TEOBResumS-SPA (and, if LALSimulation is importable, a couple of
LAL frequency-domain approximants), as a function of the number of
frequency-grid evaluation points.

Ported from ``mlgw_bns_paper/fig/benchmarking_evaluation.py`` and updated
for the current API:

* the pretrained model is loaded with :meth:`Model.default_for_testing`
  (the only shipped model, ``default_hom``, covers the (2,2), (2,1),
  (3,1), (3,2), (3,3), (4,3) and (4,4) modes -- :data:`mlgw_bns.model.DEFAULT_MODES`),
  so ``predict`` here exercises the full multi-mode observer-frame
  reconstruction;
* parameters are drawn from ``model.dataset.make_parameter_generator``,
  with distance and inclination randomised on top.

Also includes a reduced-mode variant (:class:`MlgwBnsReducedModes`, only
predicting :data:`REDUCED_MODES`, the (2,2)/(2,1)/(3,3)/(4,4) subset the
original shipped model was limited to) for comparison against the full
model.

For each ``(approximant, seed, n_points)`` triple the setup (parameter
draw, grid construction) is done outside the timed region and only
:meth:`calculate` is timed; the triples are shuffled and re-run for
several epochs so that transient slowdowns are spread across
configurations rather than landing on one.

Run with::

    python visualization/benchmark_evaluation_time.py [--fast] [--seeds N]
        [--epochs N] [--n-grid N] [--out PATH]

Outputs ``<out>.npz`` (raw per-test timings) and ``<out>.png`` (the
loglog figure with a ``c1 + c2 * N`` fit per approximant).
"""

from __future__ import annotations

import argparse
import logging
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

import mlgw_bns
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model import Model
from mlgw_bns.mode_model import ParametersWithExtrinsic

try:  # optional -- only needed for the LAL approximants
    import lal
    import lalsimulation as lalsim

    _HAVE_LAL = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_LAL = False

try:  # optional -- only needed for the JAX approximants
    import jax

    _HAVE_JAX = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_JAX = False

#: Modes for :class:`MlgwBnsReducedModes` -- the (2,2)/(2,1)/(3,3)/(4,4)
#: subset the original shipped model was limited to, kept as a fixed
#: reference point now that :data:`mlgw_bns.model.DEFAULT_MODES` covers
#: all 7 modes of the current shipped model.
REDUCED_MODES = [Mode(2, 2), Mode(2, 1), Mode(3, 3), Mode(4, 4)]


def random_parameters(model: Model, seed: int) -> ParametersWithExtrinsic:
    """Draw one intrinsic point from the model's own generator, then add
    a random distance (log-uniform, 0.1--1e4 Mpc) and inclination."""
    param_generator = model.dataset.make_parameter_generator(seed)
    intrinsic = next(param_generator)

    return ParametersWithExtrinsic(
        mass_ratio=intrinsic.mass_ratio,
        lambda_1=intrinsic.lambda_1,
        lambda_2=intrinsic.lambda_2,
        chi_1=intrinsic.chi_1,
        chi_2=intrinsic.chi_2,
        distance_mpc=10 ** param_generator.rng.uniform(-1, 4),
        inclination=param_generator.rng.uniform(-np.pi, np.pi),
        total_mass=2.8,
    )


class Approximant(ABC):
    name: str = ""
    #: number of waveforms produced per `calculate()` call; `TestCase`
    #: divides the measured time by this to report a per-waveform cost.
    n_waveforms: int = 1
    #: grid sizes above this are skipped for this approximant (the batched
    #: JAX path needs O(batch * n_points) memory).
    max_points: float = float("inf")

    def __init__(self, model_name: str = "default_hom"):
        self.model = Model.default_for_testing(model_name)
        self.dataset = self.model.dataset

    @abstractmethod
    def setup(self, seed: int, n_points: int) -> None:
        ...

    @abstractmethod
    def calculate(self) -> None:
        ...

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Approximant) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


class MlgwBns(Approximant):
    name = "mlgw_bns"

    def setup(self, seed: int, n_points: int) -> None:
        self.params = random_parameters(self.model, seed)
        self.frequencies = np.linspace(
            float(np.min(self.dataset.frequencies_hz)),
            float(np.max(self.dataset.frequencies_hz)) - 1,
            num=n_points,
        )
        assert len(self.frequencies) == n_points

    def calculate(self) -> None:
        self.model.predict(self.frequencies, self.params)


class MlgwBnsReducedModes(MlgwBns):
    """`MlgwBns`, but predicting only :data:`REDUCED_MODES`
    ((2,2), (2,1), (3,3), (4,4)), for comparison against the full model."""

    name = "mlgw_bns (22, 21, 33, 44 only)"

    def __init__(self, model_name: str = "default_hom") -> None:
        self.model = Model.default_for_testing(model_name, modes=list(REDUCED_MODES))
        self.dataset = self.model.dataset


class MlgwBnsJax(Approximant):
    """`mlgw_bns.jax_predict.model_to_jax_waveform`, JIT-compiled, one waveform."""

    name = "mlgw_bns (JAX)"

    def __init__(self, model_name: str = "default_hom") -> None:
        super().__init__(model_name)
        import jax

        from mlgw_bns.jax_predict import model_to_jax_waveform

        self._jax = jax
        self._predict = jax.jit(model_to_jax_waveform(self.model))
        self._warm: set[int] = set()

    def _pack(self, seed: int, n_points: int):
        import jax.numpy as jnp

        params = random_parameters(self.model, seed)
        freqs = jnp.asarray(
            np.linspace(
                float(np.min(self.dataset.frequencies_hz)),
                float(np.max(self.dataset.frequencies_hz)) - 1,
                num=n_points,
            )
        )
        args = (
            jnp.asarray(
                [params.mass_ratio, params.lambda_1, params.lambda_2,
                 params.chi_1, params.chi_2]
            ),
            freqs,
            jnp.asarray(params.total_mass),
            jnp.asarray(params.distance_mpc),
            jnp.asarray(params.inclination),
            jnp.asarray(params.reference_phase),
        )
        return args

    def setup(self, seed: int, n_points: int) -> None:
        self.args = self._pack(seed, n_points)
        if n_points not in self._warm:  # compile once per grid size
            self._jax.block_until_ready(self._predict(*self.args))
            self._warm.add(n_points)

    def calculate(self) -> None:
        self._jax.block_until_ready(self._predict(*self.args))


class MlgwBnsJaxBatch(Approximant):
    """`model_to_jax_waveform` under `jax.vmap`, `batch` waveforms per call;
    the reported time is divided by `batch` to give a per-waveform cost."""

    def __init__(self, batch: int = 1024, model_name: str = "default_hom") -> None:
        super().__init__(model_name)
        import jax

        from mlgw_bns.jax_predict import model_to_jax_waveform

        self._jax = jax
        self.batch = batch
        self.n_waveforms = batch
        self.max_points = 20_000  # O(batch * n_points) memory
        self.name = f"mlgw_bns (JAX, batch {batch})"
        self._predict = jax.jit(
            jax.vmap(
                model_to_jax_waveform(self.model),
                in_axes=(0, None, None, None, None, None),
            )
        )
        self._rng = np.random.default_rng(0)
        self._warm: set[int] = set()

    def setup(self, seed: int, n_points: int) -> None:
        import jax.numpy as jnp

        centre = random_parameters(self.model, seed)
        base = np.array(
            [centre.mass_ratio, centre.lambda_1, centre.lambda_2,
             centre.chi_1, centre.chi_2]
        )
        params = base * (1.0 + 0.05 * self._rng.standard_normal((self.batch, 5)))
        freqs = jnp.asarray(
            np.linspace(
                float(np.min(self.dataset.frequencies_hz)),
                float(np.max(self.dataset.frequencies_hz)) - 1,
                num=n_points,
            )
        )
        self.args = (
            jnp.asarray(params),
            freqs,
            jnp.asarray(centre.total_mass),
            jnp.asarray(centre.distance_mpc),
            jnp.asarray(centre.inclination),
            jnp.asarray(centre.reference_phase),
        )
        if n_points not in self._warm:  # compile once per grid size
            self._jax.block_until_ready(self._predict(*self.args))
            self._warm.add(n_points)

    def calculate(self) -> None:
        self._jax.block_until_ready(self._predict(*self.args))


class TEOBResumSPA(Approximant):
    name = "TEOBResumSPA"

    def setup(self, seed: int, n_points: int) -> None:
        params = random_parameters(self.model, seed)
        teob_dict = params.teobresums_dict(self.dataset)
        flen = (
            teob_dict["srate_interp"] / 2 - teob_dict["initial_frequency"]
        ) / teob_dict["df"]
        teob_dict["df"] *= flen / n_points
        self.teob_dict = teob_dict

    def calculate(self) -> None:
        self.dataset.waveform_generator.eobrun_callable(self.teob_dict)


def lalwf_maker(lal_approx: str) -> type:
    class LALWf(Approximant):
        approx = lal_approx
        name = lal_approx

        def setup(self, seed: int, n_points: int) -> None:
            params = random_parameters(self.model, seed)
            teob_dict = params.teobresums_dict(self.dataset)
            flen = (
                teob_dict["srate_interp"] / 2 - teob_dict["initial_frequency"]
            ) / teob_dict["df"]
            teob_dict["df"] *= flen / n_points

            lal_params = lal.CreateDict()
            modearr = lalsim.SimInspiralCreateModeArray()
            for mode in [(2, 2)]:
                lalsim.SimInspiralModeArrayActivateMode(modearr, *mode)
            lalsim.SimInspiralWaveformParamsInsertModeArray(lal_params, modearr)

            q = params.mass_ratio
            M = params.total_mass
            DL = params.distance_mpc * 1e6 * lal.PC_SI
            iota = params.inclination
            phir = params.reference_phase
            df = teob_dict["df"] / params.mass_sum_seconds
            flow = teob_dict["initial_frequency"] / params.mass_sum_seconds
            srate = teob_dict["srate_interp"] / params.mass_sum_seconds
            m1SI = M * q / (1.0 + q) * lal.MSUN_SI
            m2SI = M / (1.0 + q) * lal.MSUN_SI
            app = lalsim.GetApproximantFromString(self.approx)

            self.param_list = [
                m1SI, m2SI,
                0.0, 0.0, params.chi_1, 0.0, 0.0, params.chi_2,
                DL, iota, phir, 0.0, 0.0, 0.0,
                df, flow, srate / 2, flow,
                lal_params, app,
            ]

        def calculate(self) -> None:
            lalsim.SimInspiralFD(*self.param_list)

    return LALWf


@dataclass
class TestCase:
    seed: int
    approximant: Approximant
    n_points: int
    times_ms: list[float] = field(default_factory=list)

    @property
    def avg_time(self) -> float:
        return float(np.average(self.times_ms))

    @property
    def std_time(self) -> float:
        return float(np.std(self.times_ms))

    def run(self) -> None:
        self.approximant.setup(self.seed, self.n_points)
        start = perf_counter()
        self.approximant.calculate()
        elapsed_ms = (perf_counter() - start) * 1e3
        self.times_ms.append(elapsed_ms / self.approximant.n_waveforms)


def make_test_cases(
    approximants: list[Approximant],
    seeds: list[int],
    n_points_list: list[int],
) -> list[TestCase]:
    ntot = len(n_points_list) * len(seeds) * len(approximants)
    print(
        f"Made {len(n_points_list)}x{len(seeds)}x{len(approximants)}={ntot} tests; "
        f"~{ntot * 0.12 / 60:.1f} min per epoch (rough)"
    )
    return [
        TestCase(approximant=approx, seed=int(seed), n_points=int(n_points))
        for approx, seed, n_points in product(approximants, seeds, n_points_list)
        if n_points <= approx.max_points
    ]


def run_tests_n_times(
    tests: list[TestCase],
    epochs: int,
    shuffler: random.Random = random.Random(1),
) -> None:
    for _ in tqdm(range(epochs), unit="epoch"):
        shuffler.shuffle(tests)
        for test in tests:
            test.run()


def get_attribute_by_n_points_for_approx(
    tests: list[TestCase],
    approximant: Approximant,
    attribute: str,
    operation: Callable[[list[Any]], Any],
) -> list[Any]:
    relevant = [t for t in tests if t.approximant == approximant]
    by_n: dict[int, list[Any]] = defaultdict(list)
    for test in sorted(relevant, key=lambda t: t.n_points):
        by_n[test.n_points].append(getattr(test, attribute))
    return [operation(v) for _, v in sorted(by_n.items())]


def create_and_run_tests(
    n_seeds: int,
    n_epochs: int,
    n_points_list: list[int],
    approximants: list[Approximant],
) -> list[TestCase]:
    # factor out any one-off warm-up (jit, caches) before timing
    for approx in approximants:
        approx.setup(100, 100)
        approx.calculate()

    tests = make_test_cases(approximants, list(np.arange(n_seeds)), n_points_list)
    run_tests_n_times(tests, n_epochs)
    return tests


def linear_constant_fit(n_points: np.ndarray, times: np.ndarray) -> np.ndarray:
    popt, _ = curve_fit(
        lambda x, c1, c2: c1 + x * c2, n_points, times, sigma=times
    )
    return popt


def format_popt(popt: np.ndarray) -> str:
    c1, c2 = popt
    return f"{c1:.2f} ms + {1e6 * c2:.0f} ns x N"


def make_figure(
    tests: list[TestCase],
    approximants: list[Approximant],
    n_points_list: list[int],
) -> None:
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0.15, 0.85, num=len(approximants))]

    plt.figure(figsize=(7, 4.5))
    for approximant, color in zip(approximants, colors):
        points = [n for n in n_points_list if n <= approximant.max_points]
        times = get_attribute_by_n_points_for_approx(
            tests, approximant, "avg_time", np.average
        )
        popt = linear_constant_fit(np.array(points, dtype=float), np.array(times))
        model = lambda x, c1, c2: c1 + x * c2
        plt.loglog(
            points,
            model(np.array(points), *popt),
            c=color,
            lw=0.9,
            label=format_popt(popt),
        )
        plt.scatter(points, times, s=3.0, color=color, label=approximant.name)

    plt.grid(True, which="both", lw=0.3)
    plt.gca().set_axisbelow(True)
    plt.xlabel("Number of evaluation points")
    plt.ylabel("Evaluation time [ms]")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], ncol=2, fontsize=8)
    plt.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="fewer seeds/epochs/grid points, for a smoke run",
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n-grid", type=int, default=50)
    parser.add_argument(
        "--log-max",
        type=float,
        default=6.0,
        help="largest grid size is 10**log_max evaluation points",
    )
    parser.add_argument("--no-lal", action="store_true", help="skip LAL approximants")
    parser.add_argument(
        "--no-jax",
        action="store_true",
        help="skip the JAX port (single call + a batch under jax.vmap)",
    )
    parser.add_argument("--jax-batch", type=int, default=1024)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("benchmark_evaluation_time"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    if args.fast:
        args.seeds, args.epochs, args.n_grid, args.log_max = 3, 2, 10, 4.0

    n_points_list = sorted(
        {int(x) for x in np.logspace(2, args.log_max, num=args.n_grid, dtype=int)}
    )

    approximants: list[Approximant] = [
        MlgwBns(),
        MlgwBnsReducedModes(),
        TEOBResumSPA(),
    ]
    if _HAVE_JAX and not args.no_jax:
        approximants.append(MlgwBnsJax())
        approximants.append(MlgwBnsJaxBatch(batch=args.jax_batch))
    elif not _HAVE_JAX:
        logging.warning("JAX not importable -- JAX approximants skipped")
    if _HAVE_LAL and not args.no_lal:
        for approx_name in ("SEOBNRv4_ROM_NRTidalv2", "SEOBNRv4T_surrogate"):
            try:
                approximants.append(lalwf_maker(approx_name)())
            except Exception as exc:  # pragma: no cover
                logging.warning("skipping LAL approximant %s (%s)", approx_name, exc)
    elif not _HAVE_LAL:
        logging.warning("LALSimulation not importable -- LAL approximants skipped")

    tests = create_and_run_tests(
        args.seeds, args.epochs, n_points_list, approximants
    )

    raw = {
        f"{t.approximant.name}|{t.seed}|{t.n_points}": np.array(t.times_ms)
        for t in tests
    }
    np.savez(args.out.with_suffix(".npz"), n_points_list=n_points_list, **raw)
    print(f"wrote {args.out.with_suffix('.npz')}")

    make_figure(tests, approximants, n_points_list)
    plt.savefig(args.out.with_suffix(".png"), dpi=150)
    print(f"wrote {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
