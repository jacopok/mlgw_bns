Installation
============

How to install this module? 
First, install `poetry <https://python-poetry.org/docs/#installation>`_, 
and then run

.. code-block:: bash
    
    poetry install

in the project directory.

To see whether everything is working properly, run 

.. code-block:: bash
    
    poetry run tox -e python3.9
    
where you can swap the python version to another one you have available
on your system (between the supported ones).
