# JAX predictor — upstream reference

Verbatim copies of the JAX port of `mlgw_bns` written by
**Saulo Albuquerque** (`saulo.soaresdealbuquerquefilho@uniurb.it`,
GitHub `saulo-albuquerque-phys`), from the `mlgw-bns-jax` repository:

| commit | file(s) |
| --- | --- |
| `2ee3bbd1c4ed887e600d6c419bef230e61b80e67` (2026-05-17) | `jax_predict.py`, `jax_import_n_predict.py`, `jax_export.py` |
| `c348432a07b4fe130f011e61fcbcd8432fde2fd4` (2026-05-17) | `example_generate_waveforms.py` |

That work targets the single-mode `mlgw_bns` 0.12.1 interface (a five-parameter
`SklearnNetwork` MLP → PCA → TaylorF2 → cubic-spline resample, one `(2, 2)`
polarisation). Same GPL-3.0-only licence as this repository.

- `jax_predict.py` — builds JAX functions from a live `Model` object
  (`model_to_jax_predict`, `model_to_jax_waveform{,_ds}`).
- `jax_import_n_predict.py` — standalone predictor that loads the HDF5 file
  written by `jax_export.py`, with no `mlgw_bns` dependency.
- `jax_export.py` — dumps a trained `Model` to that HDF5 format.

The adaptation to the current multi-mode interface (`KernelRidgeNetwork`
residual regressor, `ModePhasesNN`, per-mode TaylorF2, the `Y_lm` projection)
lives in `mlgw_bns/jax_predict.py`; the PN expansions and the JAX cubic-spline
evaluator there are taken from these files.
