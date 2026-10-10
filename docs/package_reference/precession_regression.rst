Regressed precession angles
===========================

.. automodule:: mlgw_bns.precession_regression

Angles and frames
-----------------

.. autoclass:: mlgw_bns.precession_regression.AngleGrid
    :members:

.. autoclass:: mlgw_bns.precession_regression.ReferenceFrame
    :members:

.. autofunction:: mlgw_bns.precession_regression.reference_frame

.. autoclass:: mlgw_bns.precession_regression.RegressedAngles
    :members:

.. autofunction:: mlgw_bns.precession_regression.j_frame_angles

.. autofunction:: mlgw_bns.precession_regression.euler_angles_of

Carriers and nutation
---------------------

.. autofunction:: mlgw_bns.precession_regression.carrier_rates

.. autofunction:: mlgw_bns.precession_regression.carrier_table

.. autofunction:: mlgw_bns.precession_regression.carrier_tables

.. autofunction:: mlgw_bns.precession_regression.g_baseline

.. autofunction:: mlgw_bns.precession_regression.frozen_precession_constants

.. autofunction:: mlgw_bns.precession_regression.nutation

.. autofunction:: mlgw_bns.precession_regression.nutation_frequency

.. autofunction:: mlgw_bns.precession_regression.mean_precession_rate

Training data
-------------

.. autoclass:: mlgw_bns.precession_regression.TrainingRanges
    :members:

.. autofunction:: mlgw_bns.precession_regression.generate_training_set

.. autofunction:: mlgw_bns.precession_regression.integration_nodes

.. autofunction:: mlgw_bns.precession_regression.training_data

.. autofunction:: mlgw_bns.precession_regression.fit_envelopes

.. autofunction:: mlgw_bns.precession_regression.fit_all_envelopes

.. autofunction:: mlgw_bns.precession_regression.fit_switch_spins

.. autofunction:: mlgw_bns.precession_regression.refine_envelopes

Regressor
---------

.. autoclass:: mlgw_bns.precession_regression.PrecessionRegressor
    :members:

.. autofunction:: mlgw_bns.precession_regression.regression_features

.. autofunction:: mlgw_bns.precession_regression.spin_projections
