r"""The two pictures that summarise what the sweeps found.

The left panel is the answer to "how far can a fixed training budget be
pushed": accuracy against the number of training waveforms, for the
shipped pipeline and for the tuned kernel that replaces its regressor.
The shipped pipeline flattens out well inside the budget, so buying more
waveforms does nothing for it; the kernel keeps paying, down to the
floor set by the downsampling itself.

The right panel is the budget: what each mode's own mismatch is worth
once the modes are summed, weighted by the share of the noise-weighted
power it carries.

Run with: python -m experiments.plot_results
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"

#: Measured by `experiments.downsampling_floor`: the mismatch between a
#: true EOB waveform and its own resampling from the downsampling nodes.
#: No surrogate built on these nodes can do better, however good its
#: regression is.
DOWNSAMPLING_FLOOR = {"22": 2.392e-9, "21": 3.543e-9, "33": 6.857e-9}


def _training_size_curves() -> dict[str, dict[int, float]]:
    rows = json.loads((RESULTS / "sizes_l2_m2_n8192.json").read_text())
    curves: dict[str, dict[int, float]] = {}
    for row in rows:
        label = row["label"]
        match = re.search(r"n_train=(\d+)", label)
        size = int(match.group(1)) if match else 8192
        if "krr" in label:
            name = "tuned kernel ridge"
        elif "batch_size:5" in label:
            name = "shipped pipeline"
        else:
            name = "shipped, mini-batch repaired"
        curves.setdefault(name, {})[size] = row["median"]
    return curves


def main() -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))

    curves = _training_size_curves()
    styles = {
        "shipped pipeline": ("tab:orange", "o", "-"),
        "shipped, mini-batch repaired": ("tab:red", "s", "--"),
        "tuned kernel ridge": ("tab:blue", "D", "-"),
    }
    for name, curve in curves.items():
        colour, marker, dashes = styles[name]
        sizes = sorted(curve)
        left.plot(
            sizes,
            [curve[s] for s in sizes],
            marker=marker,
            color=colour,
            linestyle=dashes,
            label=name,
        )

    left.axhline(
        DOWNSAMPLING_FLOOR["22"],
        color="grey",
        linestyle=":",
        label="floor set by the downsampling nodes",
    )
    left.set_xscale("log", base=2)
    left.set_yscale("log")
    left.set_xlabel("training waveforms")
    left.set_ylabel("median mismatch, (2,2) mode")
    left.set_title("Accuracy against training budget")
    left.grid(True, which="both", alpha=0.3)
    left.legend(fontsize=9)

    # -- right panel: the full-waveform budget, mode by mode -------------
    modes = ["(2,2)", "(2,1)", "(3,3)"]
    shipped = [7.183e-06, 2.747e-08, 5.148e-08]
    best = [3.514e-09, 2.747e-08, 3.452e-09]

    x = np.arange(len(modes))
    width = 0.36
    right.bar(x - width / 2, shipped, width, label="shipped", color="tab:orange")
    right.bar(x + width / 2, best, width, label="best found", color="tab:blue")
    right.set_yscale("log")
    right.set_xticks(x, modes)
    right.set_ylabel(r"contribution to the full-waveform mismatch, $w_k \mathcal{M}_k$")
    right.set_title("Where the remaining error sits")
    right.grid(True, axis="y", which="both", alpha=0.3)
    right.legend(fontsize=9)

    # leave headroom above the tallest bar for the annotations
    right.set_ylim(top=max(shipped + best) * 6)

    for index, (before, after) in enumerate(zip(shipped, best)):
        right.annotate(
            f"{before / after:.0f}x" if before / after > 1.5 else "unchanged",
            (index, max(before, after) * 1.6),
            ha="center",
            fontsize=9,
        )

    fig.suptitle(
        "mlgw_bns: what limits the aligned-spin surrogate, and what moves it",
    )
    fig.tight_layout()
    out = RESULTS / "summary.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
