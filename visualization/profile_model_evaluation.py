r"""Profile :meth:`Model.predict`, averaged over random parameter draws.

Ported and extended from
``mlgw_bns_paper/fig/single_evaluation_breakdown.py`` (which ran a single
``cProfile`` per grid size and dumped ``.prof`` files). This version:

* loads the shipped multi-mode ``default_hom`` model
  (:meth:`Model.default_for_testing`);
* profiles ``predict`` for a range of grid sizes, each averaged over
  several random parameter draws from the model's own generator;
* prints a summary table of total time vs ``N`` and a per-component
  cumulative-time breakdown, so the fixed (``N``-independent) and
  marginal (per-point) costs can be told apart;
* optionally still dumps a ``.prof`` per grid size (``--dump-prof``).

Run with::

    python visualization/profile_model_evaluation.py [--n-freq ...] [--draws N]
        [--f-min 5] [--f-max 2500] [--dump-prof] [--top 15]
"""

from __future__ import annotations

import argparse
import cProfile
import logging
import pstats
from pathlib import Path

import numpy as np

from mlgw_bns.model import Model
from mlgw_bns.mode_model import ParametersWithExtrinsic

#: (filename substring, function name) rows for the component table.
#: ``"*"`` as the function name aggregates every function in that file.
COMPONENTS = [
    ("model.py", "predict"),
    ("model.py", "_hpc_waveform"),
    ("mode_model.py", "predict_amplitude_phase_optimized"),
    ("mode_model.py", "predict_residuals_bulk"),
    ("mode_model.py", "_predicted_mode_phase0"),
    ("kernel_ridge.py", "predict"),
    ("kernel_ridge.py", "_get_kernel"),
    ("neural_network.py", "predict"),
    ("validation.py", "check_array"),
    ("_param_validation.py", "wrapper"),
    ("dataset_generation.py", "hz_to_natural_units"),
    ("downsampling_interpolation.py", "*"),
    ("_interpolate.py", "*"),
    ("taylorf2.py", "*"),
    ("higher_order_modes.py", "*"),
]


def make_draw(model: Model):
    dataset = model.dataset

    def draw(seed: int) -> ParametersWithExtrinsic:
        gen = dataset.make_parameter_generator(seed)
        intrinsic = next(gen)
        return ParametersWithExtrinsic(
            mass_ratio=intrinsic.mass_ratio,
            lambda_1=intrinsic.lambda_1,
            lambda_2=intrinsic.lambda_2,
            chi_1=intrinsic.chi_1,
            chi_2=intrinsic.chi_2,
            distance_mpc=10 ** gen.rng.uniform(-1, 4),
            inclination=gen.rng.uniform(-np.pi, np.pi),
            total_mass=2.8,
        )

    return draw


def component_ms(stats: pstats.Stats, filesub: str, func: str, n_draws: int) -> float:
    """Mean cumulative ms per predict call for one component."""
    total_ct = 0.0
    for (fn, _lineno, name), (_cc, _nc, _tt, ct, _callers) in stats.stats.items():  # type: ignore[attr-defined]
        if (func == "*" or name == func) and filesub in fn:
            total_ct += ct
    return total_ct / n_draws * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-freq",
        type=int,
        nargs="+",
        default=[100, 300, 1000, 3000, 10000, 30000, 100000],
    )
    parser.add_argument("--draws", type=int, default=12)
    parser.add_argument("--f-min", type=float, default=5.0)
    parser.add_argument("--f-max", type=float, default=2500.0)
    parser.add_argument("--model", default="default_hom")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--dump-prof", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = Model.default_for_testing(args.model)
    draw = make_draw(model)

    model.predict(np.array([20.0]), draw(0))  # warm-up

    totals: dict[int, list[float]] = {}
    aggregated: dict[int, pstats.Stats] = {}

    for n_freq in args.n_freq:
        freqs = np.linspace(args.f_min, args.f_max, num=n_freq)
        totals[n_freq] = []
        combined: pstats.Stats | None = None
        for s in range(args.draws):
            params = draw(1000 + s)
            prof = cProfile.Profile()
            prof.enable()
            model.predict(freqs, params)
            prof.disable()
            totals[n_freq].append(pstats.Stats(prof).total_tt)
            combined = pstats.Stats(prof) if combined is None else (combined.add(prof) or combined)
        assert combined is not None
        combined.strip_dirs()
        aggregated[n_freq] = combined
        if args.dump_prof:
            combined.dump_stats(args.out_dir / f"profile_{n_freq}.prof")

    print(f"\ntotal predict time [ms], mean +/- std over {args.draws} draws\n")
    print(f"{'N':>9}  {'mean':>8}  {'std':>7}")
    for n_freq in args.n_freq:
        arr = np.array(totals[n_freq]) * 1e3
        print(f"{n_freq:>9}  {arr.mean():>8.2f}  {arr.std():>7.2f}")

    header = f"\n{'component (cumulative ms/call)':<46}" + "".join(
        f"{n:>9}" for n in args.n_freq
    )
    print(header)
    print("-" * len(header.strip("\n")))
    for filesub, func in COMPONENTS:
        label = f"{filesub}:{func}"
        row = f"{label:<46}" + "".join(
            f"{component_ms(aggregated[n], filesub, func, args.draws):>9.2f}"
            for n in args.n_freq
        )
        print(row)

    for n_freq in (args.n_freq[0], args.n_freq[-1]):
        print(f"\nself-time (tottime) top {args.top}, N={n_freq}:")
        stats = aggregated[n_freq]
        rows = [
            (tt / args.draws * 1e3, f"{fn.split('/')[-1]}:{name}", nc / args.draws)
            for (fn, _l, name), (_cc, nc, tt, _ct, _cal) in stats.stats.items()  # type: ignore[attr-defined]
        ]
        for tt, name, nc in sorted(rows, reverse=True)[: args.top]:
            print(f"  {tt:>7.2f} ms  {nc:>8.1f} calls  {name}")


if __name__ == "__main__":
    main()
