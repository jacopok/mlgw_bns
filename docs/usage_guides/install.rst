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

To install it, follow the instructions to install the Python wrapper 
`here <https://bitbucket.org/eob_ihes/teobresums/src/master/>`_.  
To check whether it is correctly installed, try to 

.. code-block:: python

    import EOBRun_module

in a python session.


From the repo
-------------

Once you have cloned the `repo <https://github.com/jacopok/mlgw_bns>`_, 
install `poetry <https://python-poetry.org/docs/#installation>`_, 
and then run

.. code-block:: bash
    
    poetry install

in the project directory.
In this case, the `TEOBResumS` repository must be installed in the same folder 
as `mlgw_bns`:

.. code-block:: bash

    some_folder/
    |--- mlgw_bns/
        |--- mlgw_bns/
        |--- docs/
        |--- tests/
        |--- ...
    |--- teobresums/
        |--- Python/
        |--- C/ 
        |--- ...


Testing and dev functionality
-----------------------------

To see whether everything is working properly, run 

.. code-block:: bash
    
    poetry run tox -e python3.9
    
where you can swap the python version to another one you have available
on your system (between the supported ones --- 3.7 to 3.9).

This will install all missing dependencies, 
run tests and also build the documentation locally, in the folder `docs/html/`;
one can access it starting from `index.html`.

To only run the tests, do 

.. code-block:: bash
    
    poetry run pytest


To only build the documentation, do

.. code-block:: bash
    
    poetry run sphinx-build docs docs/html


Make a pretty dependency graph with 

.. code-block:: bash
    
    poetry run pydeps mlgw_bns/


To make an html page showing the test coverage of the code, do

.. code-block:: bash
    
    poetry run coverage html


There are pre-commit hooks which will clean up the code, 
format everything with `black`, check that there are no large files,
check that the typing is correct with `mypy`. 
