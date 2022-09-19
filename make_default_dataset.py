from mlgw_bns import Model

def main_model():
    m = Model("default_dataset", initial_frequency_hz=5.)
    m.generate(2 ** 10, 2 ** 16, 2 ** 17)
    m.set_hyper_and_train_nn()
    m.save(include_training_data=False)

def model_2():

    m = Model("default_dataset_faster", initial_frequency_hz=15., pca_components_number=25)
    m.generate(2 ** 10, 2 ** 16, 2 ** 17)
    m.set_hyper_and_train_nn()
    m.save(include_training_data=False)

if __name__ == "__main__":
    # main_model()
    model_2()