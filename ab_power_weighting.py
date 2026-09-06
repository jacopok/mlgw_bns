"""A/B the odd-m power weighting without a full retrain.

The shipped ``default_hom`` model already carries its 24576-waveform
training set and its fitted PCA basis in ``*_arrays.h5``, and power
weighting only changes the *regressor* fit. So the two arms of the
comparison can reuse everything upstream: this script re-fits the (2,1)
and (3,3) networks twice --- once unweighted, once power-weighted --- and
writes two models that differ in nothing else.

Everything the two arms share (the ``.h5`` arrays, the even-m networks,
the time-shift and mode-phase predictors) is symlinked or copied rather
than regenerated, so each arm costs two kernel-ridge solves and a few
megabytes rather than an EOB sweep and half a gigabyte.

Run with::

    python ab_power_weighting.py                    # writes pw_off_*, pw_on_*
    python ab_power_weighting.py --modes 2,1

Then compare::

    python slice_mode_pca.py --model pw_off --modes 2,1 --range q,1.0,1.6 \
        --out-prefix slice_pw_off
    python slice_mode_pca.py --model pw_on  --modes 2,1 --range q,1.0,1.6 \
        --out-prefix slice_pw_on
    python visualization/validate_model.py --model pw_on --n-mismatches 1000
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model import DEFAULT_MODES, Model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def link_or_copy(source: Path, destination: Path, symlink: bool) -> None:
    """Materialise ``destination`` from ``source``, replacing what is there."""
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    if symlink:
        destination.symlink_to(source.resolve())
    else:
        shutil.copy2(source, destination)


def build_arm(source_base: str, target_base: str, weighting: bool,
              retrain_modes: list[Mode], modes: list[Mode],
              exponent: float = 1.0) -> None:
    """Write one arm of the comparison under ``target_base``."""

    model = Model(modes=list(modes), filename=source_base)
    model.load()
    if not model.nn_available:
        raise SystemExit(f"no trained network for {source_base!r}")

    # Shared predictors and every mode's arrays/metadata come straight
    # from the source model; only the odd-m networks are re-fit.
    for suffix in ("_timeshifts.pkl", "_mode_phases.pkl"):
        link_or_copy(Path(f"{source_base}{suffix}"), Path(f"{target_base}{suffix}"),
                     symlink=False)

    for mode in modes:
        src = f"{source_base}_l{mode.l}_m{mode.m}"
        dst = f"{target_base}_l{mode.l}_m{mode.m}"
        # the arrays are hundreds of MB and identical across both arms
        link_or_copy(Path(f"{src}_arrays.h5"), Path(f"{dst}_arrays.h5"), symlink=True)
        link_or_copy(Path(f"{src}.yaml"), Path(f"{dst}.yaml"), symlink=False)
        if mode not in retrain_modes:
            link_or_copy(Path(f"{src}_nn.pkl"), Path(f"{dst}_nn.pkl"), symlink=False)

    for mode in retrain_modes:
        mode_model = model.mode_models[mode]
        mode_model.power_weighting = weighting
        mode_model.power_weight_exponent = exponent

        start = time.perf_counter()
        mode_model.set_hyper_and_train_nn()
        logging.info(
            "[%s] mode %s re-fit (%s) in %.0f s",
            target_base, mode, "weighted" if weighting else "unweighted",
            time.perf_counter() - start,
        )

        mode_model.nn.save(f"{target_base}_l{mode.l}_m{mode.m}_nn.pkl")

    logging.info("[%s] done", target_base)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=str, default="default_hom")
    parser.add_argument("--modes", type=str, default="2,1;3,3",
                        help="modes to re-fit; the rest are copied unchanged")
    parser.add_argument("--exponent", type=float, default=1.0)
    parser.add_argument("--off-name", type=str, default="pw_off")
    parser.add_argument("--on-name", type=str, default="pw_on")
    parser.add_argument("--only", type=str, default="both",
                        choices=("both", "off", "on"))
    args = parser.parse_args()

    modes = list(DEFAULT_MODES)
    retrain = [Mode(*(int(x) for x in m.split(","))) for m in args.modes.split(";")]
    for mode in retrain:
        if mode.m % 2 == 0:
            raise SystemExit(
                f"mode {mode} has even m, where power weighting is a no-op by design"
            )

    if args.only in ("both", "off"):
        build_arm(args.source, args.off_name, False, retrain, modes, args.exponent)
    if args.only in ("both", "on"):
        build_arm(args.source, args.on_name, True, retrain, modes, args.exponent)


if __name__ == "__main__":
    main()
