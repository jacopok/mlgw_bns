"""Plot the parameter importance of each mode's kernel-ridge hyperparameter
study, produced by ``optimize_n_hours.py``.

There is no Pareto front to plot any more: the search is over a single
objective (reconstruction accuracy at the fixed training-set size), not
accuracy vs. training time.

Run with: python visualization/plot_optimization_pareto.py
"""

if __name__ == "__main__":
    from mlgw_bns.hyperparameter_optimization import HyperparameterOptimization
    from mlgw_bns.mode_model import ModeModel
    from mlgw_bns.model import DEFAULT_MODES
    from mlgw_bns.neural_network import KernelRidgeNetwork
    from optimize_n_hours import dataset_filename

    for mode in DEFAULT_MODES:
        m = ModeModel(dataset_filename(mode), mode=mode, nn_kind=KernelRidgeNetwork)
        m.load()
        ho = HyperparameterOptimization(m)
        outfile = f"param_importance_{mode.l}{mode.m}.png"
        ho.plot_param_importance(outfile)
        print(f"Saved {outfile}")
