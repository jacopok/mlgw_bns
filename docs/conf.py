# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "mlgw_bns"
copyright = "2021, Jacopo Tissino"
author = "Jacopo Tissino"

# The full version, including alpha/beta/rc tags
release = "0.0.2-alpha.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
    "myst_parser",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "sphinx_search.extension",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_defaultargs"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Add the bibliography
bibtex_bibfiles = ['refs.bib']

rst_prolog = """
.. |default| raw:: html

    <div class="default-value-section">""" + \
    ' <span class="default-value-label">Default:</span>'


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

napoleon_custom_sections = [
    ("Class Attributes", "params_style"),
    ("Abstract Properties", "params_style"),
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

add_module_names = False

master_doc = "index"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/devdocs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
    "optuna": ("https://optuna.readthedocs.io/en/stable/", None),
}
