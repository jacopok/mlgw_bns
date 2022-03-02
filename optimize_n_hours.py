import argparse
from mlgw_bns import *

if __name__ == "__main__":
    try:
        m = Model("optimization_dataset")
        m.load()
    except FileNotFoundError:
        m.generate(512, 1 << 14)
    ho = HyperparameterOptimization(m)

    n_hours_before = ho.total_training_time().total_seconds() / 3600

    print(f"Optimized for {n_hours_before:2f} hours so far")

    parser = argparse.ArgumentParser(description="Optimize the hyperparameters")

    parser.add_argument("hours", metavar="h", type=float)
    parser.add_argument(
        "-g", "--generate", metavar="gen", default=False, nargs=1, type=int
    )

    args = parser.parse_args()

    if args.generate:
        m.generate(None, None, args.generate[0])
        m.save()

    ho.optimize_and_save(args.hours)

    n_hours_after = ho.total_training_time().total_seconds() / 3600

    print(f"Optimized for {n_hours_after - n_hours_before:2f} more hours")
