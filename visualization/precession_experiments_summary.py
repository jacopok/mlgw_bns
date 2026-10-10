r"""Tabulate the validation runs of the regressed-precession experiments.

Reads the ``validate --out`` files of :mod:`precession_regression_study` in
``visualization/precession_experiments/`` and prints, for each, the
waveform mismatch percentiles, overall and by how anharmonic the nutation is
(the largest elliptic parameter :math:`m` over the band)::

    python visualization/precession_experiments_summary.py [names ...]
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

DIRECTORY = os.path.join(os.path.dirname(__file__), "precession_experiments")
M_BINS = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="validate_<name>.npz to show (default: all)")
    args = parser.parse_args()
    files = (
        [os.path.join(DIRECTORY, f"validate_{name}.npz") for name in args.names]
        if args.names else sorted(glob.glob(os.path.join(DIRECTORY, "validate_*.npz")))
    )
    header = f"{'run':24s} {'n':>5s} {'median':>8s} {'90%':>8s} {'99%':>8s} {'max':>8s} {'>1e-2':>6s}"
    header += "".join(f" {'m ' + str(lo) + '-' + str(hi):>12s}" for lo, hi in M_BINS)
    print(header + "   (median by max m)")
    for name in files:
        result = np.load(name)
        mismatches, max_m = result["mismatches"], result["max_m"]
        row = f"{os.path.basename(name)[9:-4]:24s} {len(mismatches):5d}"
        row += "".join(f" {v:8.1e}" for v in (
            np.median(mismatches), *np.percentile(mismatches, [90, 99]), mismatches.max()
        ))
        row += f" {np.mean(mismatches > 1e-2):6.1%}"
        for lo, hi in M_BINS:
            cell = (max_m >= lo) & (max_m < hi)
            row += f" {np.median(mismatches[cell]):12.1e}" if cell.any() else f" {'-':>12s}"
        print(row)


if __name__ == "__main__":
    main()
