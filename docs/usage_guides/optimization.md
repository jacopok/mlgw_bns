(hyperparameter_optimization_section)=
# Optimizing hyperparameters

See {class}`HyperparameterOptimization <mlgw_bns.hyperparameter_optimization.HyperparameterOptimization>`.

The following code, which assumes a `model` was already created,
runs an optimization job with a timeout of 2 hours.

```python
from mlgw_bns import HyperparameterOptimization

ho = HyperparameterOptimization(model)
ho.optimize_and_save(2.)
```