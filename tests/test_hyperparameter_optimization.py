from datetime import timedelta

from mlgw_bns.hyperparameter_optimization import HyperparameterOptimization


def test_study_creation(generated_model):
    ho = HyperparameterOptimization(generated_model)

    assert ho.training_data_number == 100
    assert ho.study_filename == "test_model_study.pkl"


def test_optimization_smoketest(generated_model):
    ho = HyperparameterOptimization(generated_model)

    ho.optimize(timeout_min=1 / 60)

    assert ho.total_training_time() > timedelta(seconds=1)
