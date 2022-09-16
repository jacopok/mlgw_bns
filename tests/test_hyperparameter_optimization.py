from mlgw_bns.hyperparameter_optimization import HyperparameterOptimization


def test_study_creation(generated_model):
    ho = HyperparameterOptimization(generated_model)

    assert ho.training_data_number == 100
    assert ho.study_filename == "test_model_study.pkl"
