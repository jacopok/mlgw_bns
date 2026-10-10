Training at scale
=================

Datasets on disk, made by many processes at once, and the training sets of
the two models built on them; see :ref:`cluster-training` for the SLURM
pipelines that use them.

Shards
------

.. automodule:: mlgw_bns.sharding

.. autoclass:: mlgw_bns.sharding.ShardedStore
    :members:

.. autoclass:: mlgw_bns.sharding.ShardLock
    :members:

.. autofunction:: mlgw_bns.sharding.write_atomically

.. autofunction:: mlgw_bns.sharding.save_arrays

.. autofunction:: mlgw_bns.sharding.stop_on_signals

.. autodata:: mlgw_bns.sharding.EXIT_REQUEUE

The co-precessing modes model
-----------------------------

.. automodule:: mlgw_bns.modes_dataset

.. autoclass:: mlgw_bns.modes_dataset.ModesDatasetConfig
    :members:

.. autoclass:: mlgw_bns.modes_dataset.ShardedModesDataset
    :members:

.. autofunction:: mlgw_bns.modes_dataset.load_model

.. autofunction:: mlgw_bns.modes_dataset.stored_size

The regressed precession angles
-------------------------------

.. automodule:: mlgw_bns.precession_dataset

.. autoclass:: mlgw_bns.precession_dataset.DatasetConfig
    :members:

.. autoclass:: mlgw_bns.precession_dataset.ShardedDataset
    :members:
