from mlgw_bns.model import Model
from mlgw_bns.modes_model import ModesModel
from mlgw_bns.higher_order_modes import Mode
import logging
logging.basicConfig(level=logging.INFO)

def main_model():
    m = Model("default", initial_frequency_hz=5.)
    m.generate(2 ** 10, 2 ** 13, 2 ** 17)
    m.set_hyper_and_train_nn()
    m.save(include_training_data=False)

def model_2():
    m = Model("fast", initial_frequency_hz=15.)
    m.generate(2 ** 9, 2 ** 10, 2 ** 10)
    m.set_hyper_and_train_nn()
    m.save(include_training_data=False)

def modes_model():
    m = ModesModel(
        modes=[Mode(2, 2), Mode(2, 1), Mode(3, 3), Mode(4, 4)],
        filename="default_hom",
        initial_frequency_hz=5.,
    )
    m.generate(2 ** 8, 2 ** 8, 2 ** 8)
    m.set_hyper_and_train_nn()
    m.save(include_training_data=False)

if __name__ == "__main__":
    # main_model()
    # model_2()
    modes_model()