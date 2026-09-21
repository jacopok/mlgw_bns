(hyperparameter_optimization_section)=
# Optimizing hyperparameters

See {class}`HyperparameterOptimization <mlgw_bns.hyperparameter_optimization.HyperparameterOptimization>`.

It optimizes one mode's {class}`KernelRidgeNetwork <mlgw_bns.neural_network.KernelRidgeNetwork>`
hyperparameters at a time, so given a `model` as described in
[](overview_section), the following code runs an optimization job, with
a timeout of 2 hours, for its $(2,2)$ mode.

```python
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.hyperparameter_optimization import HyperparameterOptimization

ho = HyperparameterOptimization(model.mode_models[Mode(2, 2)])
ho.optimize_and_save(2.)
```