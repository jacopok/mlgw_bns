from mlgw_bns import *


def main():
    m = Model("default_dataset")
    m.generate(1 << 10, 1 << 15, 1 << 16)
    m.set_hyper_and_train_nn()
    m.save(False)

    # import os; os.replace...


if __name__ == "__main__":
    main()
