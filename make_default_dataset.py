from mlgw_bns import *


def main():
    m = Model("default_dataset")
    m.generate(512, 1 << 15, 1 << 17)
    m.set_hyper_and_train_nn()
    m.save()

    # import os; os.replace...


if __name__ == "__main__":
    main()
