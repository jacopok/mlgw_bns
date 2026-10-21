Installation
============

How to install this module? 

From pip
--------

.. code-block:: bash
    
    pip install mlgw-bns

This is enough to generate waveforms with the pretrained models (see
:ref:`shipped_models`); their regressors are downloaded the first time
they are used.
Precessing waveforms (see :ref:`precessing_waveforms`) need JAX, which is
an optional extra:

.. code-block:: bash
    
    pip install "mlgw-bns[jax]"

.. _install_teobresums:

TEOBResumS
----------

Having ``TEOBResumS`` installed is optional, but it is needed to generate
new models and to compare against the reference waveforms.
The shipped models are trained with a fork of it, the
``initial-spa-fix`` branch of
`github.com/jacopok/teobresums <https://github.com/jacopok/teobresums>`_:
the public version aborts on some of the waveforms of the training sets,
and its frequency-domain precessing waveforms have spurious :math:`2\pi`
steps in the precession angle :math:`\alpha`.
It can be installed with

.. code-block:: bash

    pip install "git+https://github.com/jacopok/teobresums@initial-spa-fix#subdirectory=Python"

It is distributed as a source package which links against the 
`GNU Scientific Library <https://www.gnu.org/software/gsl/>`_, so its 
development headers need to be available at install time 
(``sudo apt-get install libgsl-dev`` on Debian and derivatives).
To check whether it is correctly installed, try to 

.. code-block:: python

    import EOBRun_module

in a python session.


From the repo
-------------

Once you have cloned the `repo <https://github.com/jacopok/mlgw_bns>`_, 
install `uv <https://docs.astral.sh/uv/getting-started/installation/>`_, 
and then run

.. code-block:: bash
    
    uv sync

in the project directory.
This creates a virtual environment in ``.venv/`` with all the dependencies, 
the fork of ``TEOBResumS`` included; add ``--extra jax`` for the precessing
waveforms.


Testing and dev functionality
-----------------------------

To see whether everything is working properly, run the tests with

.. code-block:: bash
    
    uv run pytest


To build the documentation locally, in the folder `docs/html/`
(one can access it starting from `index.html`), do

.. code-block:: bash
    
    uv run --extra docs sphinx-build docs docs/html


Make a pretty dependency graph with 

.. code-block:: bash
    
    uv run pydeps mlgw_bns/


To make an html page showing the test coverage of the code, do

.. code-block:: bash
    
    uv run coverage html


There are pre-commit hooks which will clean up the code, 
format everything with `black`, check that there are no large files,
check that the typing is correct with `mypy`. 
