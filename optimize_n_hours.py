import argparse
from mlgw_bns import *

if __name__ == "__main__":
    m = Model("optimization_dataset")
    print(m.filename)
    m.load()
    ho = HyperparameterOptimization(m)

    parser = argparse.ArgumentParser(description="Optimize the hyperparameters")

    parser.add_argument("hours", metavar="h", type=float)
    parser.add_argument("--generate", metavar="gen", default=False, nargs=1, type=int)

    args = parser.parse_args()

    if args.generate:
        m.generate(None, None, args.generate[0])
        m.save()

    ho.optimize_and_save(args.hours)
