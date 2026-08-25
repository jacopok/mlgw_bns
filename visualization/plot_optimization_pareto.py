if __name__ == "__main__":
    from mlgw_bns import *

    m = Model("optimization_dataset")
    m.load()
    ho = HyperparameterOptimization(m)
    ho.plot_pareto()
