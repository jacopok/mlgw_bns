Installation
============

How to install this module? 

From pip
--------

If you have the archive 
(for now, I can email it to you --- it will be published to pip eventually)
you can install it with 

.. code-block:: bash
    
    pip install mlgw-bns-tarball.tar.gz

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

Once you have cloned the `repo <https://github.com/jacopok/mlgw_bns>`_ 
(which is private currently), 
install `poetry <https://python-poetry.org/docs/#installation>`_, 
and then run

.. code-block:: bash
    
    poetry install

in the project directory.

To see whether everything is working properly, run 

.. code-block:: bash
    
    poetry run tox -e python3.9
    
where you can swap the python version to another one you have available
on your system (between the supported ones).
