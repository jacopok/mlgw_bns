Installation
============

How to install this module? 

From pip
--------

.. code-block:: bash
    
    pip install mlgw-bns

Having ``TEOBResumS`` installed is optional, but 
the functionality of the package is currently severely hampered without it ---
no new models can be generated, and the only possibility is to use the 
one which is provided by default.

It is available on PyPI, so it can be installed with 

.. code-block:: bash

    pip install teobresums

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
``TEOBResumS`` included.


Testing and dev functionality
-----------------------------

To see whether everything is working properly, run 

.. code-block:: bash
    
    uv run tox
    
This will run the tests and also build the documentation locally, 
in the folder `docs/html/`; one can access it starting from `index.html`.

To only run the tests, do 

.. code-block:: bash
    
    uv run pytest


To only build the documentation, do

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
