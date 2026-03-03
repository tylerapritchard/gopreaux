# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'gopreaux'
copyright = '2026, Craig Pellegrino, Tyler Pritchard'
author = 'Craig Pellegrino, Tyler Pritchard'
release = 'v1.0'

try:
    from sphinx_astropy.conf.v2 import *
except ImportError:
    print("sphinx_astropy not found, default configuration will be used.")   

import os
import sys

# Add the modules source to the path
sys.path.insert(0, os.path.abspath("../../src"))
# sphinx-action still cant find things, try:
for x in os.walk("../../src"):
    sys.path.insert(0, x[0])
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src/"))
sys.path.insert(0, os.path.abspath("../src/caat"))
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../src/"))
sys.path.insert(0, os.path.abspath("../../src/caat"))

# Define path to the code to be documented **relative to where conf.py (this file) is kept**
sys.path.insert(0, os.path.abspath("../src/"))
sys.path.insert(0, os.path.abspath("../src/caat/"))

autodoc_mock_imports = ['matplotlib']

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "nbsphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_automodapi.automodapi",
    "numpydoc",
    "sphinx.ext.intersphinx",
    "sphinx_collections",
    'sphinxcontrib.mermaid'
]

collections = {
    'notebooks': {
        'driver': 'copy_folder',
        'source': '../examples',
        'target': 'examples/',
        'ignore': ['*.py', '.sh'],
    }
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# PyData Theme options
# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/navigation.html
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]


#This makes the side bar
html_sidebars = {
    "**": [
        "globaltoc.html",
#        "localtoc.html",
    ]
}
# This Removes the side bar
#html_sidebars = {
#    "**": [],
#}

html_title = "gopreaux"