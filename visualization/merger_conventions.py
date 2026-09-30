"""How far apart are the usual definitions of the BNS merger time, compared
with the error mlgw_bns makes on its own?

mlgw_bns references every mode to the tangent to the (2,2) phase at the top
of the trained band (:meth:`ModeModel.merger_reference`). In TEOBResumS's
frequency-domain output the phase there is exactly linear: its ``SPA``
routine (``C/src/TEOBResumSWaveform.c``) continues the stationary-phase
approximation linearly, with slope :math:`2 \\pi t`, from the time at which
the (2,2) frequency derivative :math:`\\dot F` stops increasing. That time,
:math:`t_{\\rm SPA}`, is the merger time mlgw_bns reproduces.

For the binaries cached by ``merger_reference_corner.py`` (whose model
errors against :math:`t_{\\rm SPA}` are stored there), this script calls
TEOBResumS once per binary, exactly as mlgw_bns does, and measures on the
time axis of that same run:

- ``t_SPA``: from the slope of the FD (2,2) phase at the model's top two
  phase knots;
- ``A22 peak``: the peak of :math:`|h_{22}|` (the usual BNS merger);
- ``sum A^2 peak``: the peak of :math:`\\sum_{\\ell m} |h_{\\ell m}|^2` over
  the model's modes;
- ``TEOB TD shift``: TEOBResumS's own time-domain alignment
  (``time_shift_mrg_to_0``, as implemented: scanning back from the end, the
  sample before the last peak of :math:`\\sum |h_{\\ell m}|^2`);
- ``Omega peak``: the peak of the orbital frequency;
- ``end``: the last sample, the ``tc`` that ``time_shift_FD`` removes from
  :math:`h_+, h_\\times` (never from the multipoles mlgw_bns uses).

Peaks are refined with a parabola through the three samples around the
maximum. As a check of the reading of ``SPA``, the time at which
:math:`\\dot F_{22}` (by finite differences) stops increasing is compared
with ``t_SPA``.

The offsets from ``t_SPA`` are then compared, binary by binary, with the
model's merger-time error. Results in ``merger_conventions.npz`` (``--recompute``
to regenerate) and ``merger_conventions.png``.

Run with: python visualization/merger_conventions.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from mlgw_bns.dataset_generation import WaveformParameters
from mlgw_bns.higher_order_modes import Mode, mode_to_k, start_integration_early
from mlgw_bns.model import DEFAULT_MODES, Model

HERE = Path(__file__).resolve().parent
CONVENTIONS = ["A22 peak", "sum A^2 peak", "TEOB TD shift", "Omega peak", "end"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def parabolic_peak(t: np.ndarray, y: np.ndarray) -> float:
    i = int(np.argmax(y))
    if i == 0 or i == len(y) - 1:
        return float(t[i])
    a, b, _ = np.polyfit(t[i - 1 : i + 2] - t[i], y[i - 1 : i + 2], 2)
    return float(t[i] - b / (2 * a))


def teob_time_shift(t: np.ndarray, power: np.ndarray) -> float:
    """``time_shift_mrg_to_0`` of ``C/src/TEOBResumSUtils.c``, verbatim."""
    previous = 0.0
    for i in range(len(t) - 1, 0, -1):
        if power[i] < previous:
            return float(t[i])
        previous = power[i]
    return float(t[-1])


def measure(row: np.ndarray, dataset, top_frequencies: np.ndarray) -> dict:
    """Merger times of one binary, in seconds on TEOBResumS's time axis."""
    from EOBRun_module import EOBRunPy  # type: ignore

    q, lambda_1, lambda_2, chi_1, chi_2 = row
    params = WaveformParameters(
        mass_ratio=q, lambda_1=lambda_1, lambda_2=lambda_2,
        chi_1=chi_1, chi_2=chi_2, dataset=dataset,
    )
    par_dict = params.teobresums()
    to_slice = start_integration_early(par_dict, top_frequencies, DEFAULT_MODES)
    par_dict["arg_out"] = "yes"
    par_dict["use_mode_lm"] = [mode_to_k(mode) for mode in DEFAULT_MODES]
    f_spa, _, _, _, _, hflm, htlm, dyn = EOBRunPy(par_dict)

    # geometric units: times in units of the total mass
    seconds = dataset.mass_sum_seconds
    k22 = str(mode_to_k(Mode(2, 2)))
    f_top, phase_top = f_spa[to_slice], hflm[k22][1][to_slice]
    t_spa = (phase_top[1] - phase_top[0]) / (f_top[1] - f_top[0]) / (2 * np.pi)

    t = htlm["t"]
    amplitude_22, phase_22 = htlm[k22]
    power = sum(htlm[str(mode_to_k(mode))][0] ** 2 for mode in DEFAULT_MODES)

    f_gw = np.gradient(phase_22, t) / (2 * np.pi)
    f_dot = np.gradient(f_gw, t)
    n = 1
    while n < len(t) and f_dot[n] > f_dot[n - 1]:
        n += 1

    times = {
        "t_SPA": t_spa,
        "F_dot stop (TD check)": t[n - 1],
        "A22 peak": parabolic_peak(t, amplitude_22),
        "sum A^2 peak": parabolic_peak(t, power),
        "TEOB TD shift": teob_time_shift(t, power),
        "Omega peak": parabolic_peak(dyn["t"], dyn["MOmega"]),
        "end": t[-1],
    }
    result = {key: value * seconds for key, value in times.items()}
    # time resolution of the TD output at the SPA point
    result["sample spacing"] = float(np.diff(t)[min(n - 1, len(t) - 2)]) * seconds
    return result


def compute(parameters: np.ndarray) -> dict:
    model = Model.default_for_testing()
    mode_model = model.mode_models[Mode(2, 2)]
    dataset = mode_model.dataset
    top_frequencies = dataset.frequencies[mode_model.downsampling_indices.phase_indices[-2:]]
    rows = Parallel(n_jobs=-1)(
        delayed(measure)(row, dataset, top_frequencies) for row in parameters
    )
    return {key: np.array([r[key] for r in rows]) for key in rows[0]}


def quantiles(x: np.ndarray) -> str:
    q05, q50, q95 = np.quantile(x, [0.05, 0.5, 0.95]) * 1e6
    return f"median {q50:8.1f}, 5-95% [{q05:8.1f}, {q95:8.1f}] us"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    corner = np.load(HERE / "merger_reference_corner.npz")
    model_error = np.abs(corner["time_errors"])

    cache = HERE / "merger_conventions.npz"
    if args.recompute or not cache.exists():
        np.savez(cache, **compute(corner["parameters"]))
    times = np.load(cache)
    t_spa = times["t_SPA"]

    check = times["F_dot stop (TD check)"] - t_spa
    print(
        f"{len(t_spa)} binaries at M = 2.8. Check of the SPA reading: "
        f"TD F_dot-stop minus t_SPA {quantiles(check)}; "
        f"TD sample spacing there {quantiles(times['sample spacing'])}"
    )
    print(f"model |t_mrg - t_SPA|:            {quantiles(model_error)}")
    offsets = {}
    for name in CONVENTIONS:
        offset = times[name] - t_spa
        offsets[name] = offset
        spread = np.abs(offset - np.median(offset))
        print(
            f"{name:>14} - t_SPA: {quantiles(offset)}; "
            f"|offset| > model error in {np.mean(np.abs(offset) > model_error):6.1%}, "
            f"|offset - median| > model error in {np.mean(spread > model_error):6.1%}; "
            f"median |offset| / model error {np.median(np.abs(offset) / model_error):7.1f}"
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(-2, np.log10(2e3), 70)
    ax.hist(model_error * 1e6, bins=bins, histtype="stepfilled", color="0.8",
            label=r"mlgw_bns error, $|t^{\rm model}_{\rm SPA} - t_{\rm SPA}|$")
    for name, color in zip(CONVENTIONS, COLORS):
        ax.hist(np.abs(offsets[name]) * 1e6, bins=bins, histtype="step", lw=2,
                color=color, label=rf"$|t_{{\rm {name.replace(' ', '~')}}} - t_{{\rm SPA}}|$")
    ax.set_xscale("log")
    ax.set_xlabel(r"time difference [$\mu$s], $M = 2.8\,M_\odot$")
    ax.set_ylabel("binaries")
    ax.set_title(f"Merger-time conventions vs mlgw_bns error ({len(t_spa)} uniform draws)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "merger_conventions.png", dpi=150)
    print(f"Saved {HERE / 'merger_conventions.png'}")


if __name__ == "__main__":
    main()
