from mlgw_bns import *


def main():
    m = Model("default_dataset", initial_frequency_hz=5.)
    m.generate(1 << 10, 1 << 14, 1 << 16)
    m.set_hyper_and_train_nn()
    m.save(False)

if __name__ == "__main__":
    main()
