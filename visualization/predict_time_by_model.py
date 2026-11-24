"""Wall-clock ``Model.predict`` time of saved models, side by side: median
and 90% over interleaved repeats, on a grid of N frequencies from 5 Hz to
2.5 kHz. A model is a shipped name (``default_hom_lambda2000``) or a saved
base path (``.../runs/log60_32768/model``).

    python visualization/predict_time_by_model.py N MODEL [MODEL ...]
"""

import sys
from pathlib import Path
from time import perf_counter

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from modes_scale import open_model  # noqa: E402
from profile_model_evaluation import make_draw  # noqa: E402

from mlgw_bns.model import Model  # noqa: E402


def load(name):
    return open_model(name) if "/" in name else Model.default_for_testing(name)


def main(n, *names, repeats=60):
    frequencies = np.geomspace(5.0, 2500.0, int(n))
    models = {name: load(name) for name in names}
    draws = {name: make_draw(model) for name, model in models.items()}
    times = {name: [] for name in names}
    for name, model in models.items():
        model.predict(frequencies, draws[name](0))
    for seed in range(1, repeats + 1):
        for name, model in models.items():
            params = draws[name](seed)
            start = perf_counter()
            model.predict(frequencies, params)
            times[name].append(perf_counter() - start)
    for name in names:
        t = np.array(times[name]) * 1e3
        k = models[name].mode_models[next(iter(models[name].mode_models))].pca_components_number
        print(f"{np.median(t):7.1f} / {np.quantile(t, .9):7.1f} ms  K={k:3d}  {name}")


if __name__ == "__main__":
    main(*sys.argv[1:])
