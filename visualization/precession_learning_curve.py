r"""Learning curves of the regressed precession angles (and of the
co-precessing modes model).

Waveform mismatches against the integration (1024 held-out binaries,
:math:`1 \leq q \leq 1.5`) as a function of the number of training
binaries, for the regressors of ``visualization/precession_experiments``
(:mod:`precession_regression_study` ``validate --out`` files), or for those
of a run of ``slurm/precession/submit.sh`` or ``slurm/modes/submit.sh``
(``validate_<series>_<N>.npz``, the series and sizes found there)::

    python visualization/precession_learning_curve.py
    python visualization/precession_learning_curve.py --directory DATA/runs
    # the modes model: the full waveforms, or a mode, and a lower threshold
    python visualization/precession_learning_curve.py --directory DATA/runs --threshold 1e-4
    python visualization/precession_learning_curve.py --directory DATA/runs --threshold 1e-4 --key "mode l2_m1"
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

DIRECTORY = os.path.join(os.path.dirname(__file__), "precession_experiments")

#: label, file pattern (with the training-set size), colour (fixed order of
#: the categorical palette), marker
SERIES = [
    ("kernel ridge", "validate_krr_{}.npz", "#2a78d6", "o"),
    ("kernel ridge, smoothing", "validate_krr_smooth_{}.npz", "#eb6834", "s"),
    ("kernel ridge, smoothing, refined", "validate_krr_smooth_refined_{}.npz", "#1baf7a", "D"),
    ("MLP", "validate_mlp_{}.npz", "#eda100", "^"),
    ("MLP, smoothing, refined", "validate_mlp_smooth_refined_{}.npz", "#e87ba4", "v"),
]
SIZES = [1024, 2048, 4096, 8192, 16384]
#: the colours and markers of the series found in a directory, in order
PALETTE = [(colour, marker) for _, _, colour, marker in SERIES]
def panels(threshold: float) -> list:
    """Title, statistic of the mismatches and scale of each panel."""
    exponent = np.log10(threshold)
    above = f"$10^{{{exponent:.0f}}}$" if exponent == round(exponent) else f"{threshold:g}"
    return [
        ("median mismatch", lambda m: np.median(m), "log"),
        ("90th percentile", lambda m: np.percentile(m, 90), "log"),
        (f"fraction above {above}", lambda m: np.mean(m > threshold), "linear"),
    ]


def scientific(value, _) -> str:
    """``2e-4`` as :math:`2 \times 10^{-4}`, powers of ten bare."""
    exponent = int(np.floor(np.log10(value) + 1e-9))
    mantissa = value / 10.0**exponent
    if abs(mantissa - 1) < 1e-6:
        return f"$10^{{{exponent}}}$"
    return f"${mantissa:.0f}\\times10^{{{exponent}}}$"


def discover(directory: str):
    """The series and sizes of the ``validate_<series>_<N>.npz`` files of
    ``directory``, the series in alphabetical order."""
    found = {}
    for name in glob.glob(os.path.join(directory, "validate_*_*.npz")):
        match = re.fullmatch(r"validate_(.+)_(\d+)\.npz", os.path.basename(name))
        if match:
            found.setdefault(match[1], set()).add(int(match[2]))
    if len(found) > len(PALETTE):
        raise ValueError(f"more series than colours: {sorted(found)}")
    series = [(name, f"validate_{name}_{{}}.npz", *style) for name, style in zip(sorted(found), PALETTE)]
    return series, sorted(set().union(*found.values()))


def size_labels(sizes) -> list:
    """The sizes, as powers of two where there are many of them."""
    if len(sizes) <= 6 or any(n & (n - 1) for n in sizes):
        return [str(n) for n in sizes]
    return [f"$2^{{{n.bit_length() - 1}}}$" for n in sizes]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--directory", default=DIRECTORY)
    parser.add_argument("--key", default="mismatches", help="the array of the files to plot")
    parser.add_argument("--threshold", type=float, default=1e-2, help="of the third panel")
    args = parser.parse_args()
    directory = args.directory
    series, sizes = (SERIES, SIZES) if os.path.samefile(directory, DIRECTORY) else discover(directory)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True)
    figure.patch.set_facecolor("#fcfcfb")
    for ax, (title, statistic, scale) in zip(axes, panels(args.threshold)):
        ax.set_facecolor("#fcfcfb")
        ends = []
        for label, pattern, colour, marker in series:
            points = [
                (n, statistic(np.load(os.path.join(directory, pattern.format(n)))[args.key]))
                for n in sizes if os.path.exists(os.path.join(directory, pattern.format(n)))
            ]
            if not points:
                continue
            n, value = np.array(points).T
            ax.plot(n, value, color=colour, lw=2, marker=marker, ms=6, label=label,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5)
            ends.append((label.replace("kernel ridge", "KRR"), n[-1], value[-1]))
        ax.set(xscale="log", yscale=scale, title=title, xlabel="training binaries")
        ax.set_xticks(sizes, size_labels(sizes))
        ax.xaxis.set_minor_locator(NullLocator())
        if scale == "log":
            ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_major_formatter(FuncFormatter(scientific))
            ax.yaxis.set_minor_locator(NullLocator())
        if ax is axes[0]:
            # direct labels at the line ends, nudged apart where they would overlap
            ends.sort(key=lambda end: end[2])
            placed = []
            for text, x, y in ends:
                offset = 0.0
                for _, other in placed:
                    if abs(np.log10(y) - np.log10(other)) < 0.04:
                        offset += 9.0
                placed.append((text, y))
                ax.annotate(text, (x, y), xytext=(6, offset), textcoords="offset points",
                            va="center", fontsize=8, color="#52514e")
        ax.grid(True, which="major", color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_xlim(sizes[0] * 800 / 1024, sizes[-1] * 60000 / 16384)
    axes[2].set_ylim(0, None)
    axes[2].legend(frameon=False, fontsize=8, loc="lower left")
    suffix = "" if args.key == "mismatches" else "_" + args.key.replace(" ", "_")
    out = os.path.join(directory, f"learning_curve{suffix}.png")
    figure.savefig(out, dpi=150, facecolor=figure.get_facecolor())
    print(out)


if __name__ == "__main__":
    main()
