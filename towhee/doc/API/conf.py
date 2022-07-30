# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('../../../../towhee'))


# -- Project information -----------------------------------------------------

project = 'Towhee'
copyright = '2022, Towhee Team'
author = 'Towhee Team'

# The full version, including alpha/beta/rc tags
release = 'v0.6.0'
show_authors = True


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.napoleon',
    # 'sphinx_rtd_theme',
    # 'myst_parser',
    # 'numpydoc',
    # 'sphinx.ext.autodoc',
    # 'sphinx_mdinclude',
    'sphinx_design',
    'sphinx.ext.viewcode']



# add_module_names = False

autodoc_default_options = {
    'show-inheritance': True,
    'members': True,
    'special-members': '__init__, __iter__, __getattr__, __getitem__, __setitem__, __add__, __repr__',
    'undoc-members': True,
    'hidden': True
}


# sphinx-apidoc -f -t=./_templates -e -o /Users/filiphaltmayer/Documents/code/towhee/towhe
# #e/doc/API/_source /Users/filiphaltmayer/Documents/code/towhee/towhee

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#


html_theme = 'pydata_sphinx_theme'
html_show_sphinx = False


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named 'default.css' will overwrite the builtin 'default.css'.
html_static_path = ['_static']
html_theme_options = {
    'logo': {
        'link': 'https://www.towhee.io',
        'image_light': 'towhee_logo_light.png',
        'image_dark': 'towhee_logo_dark.png',
    },
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/towhee-io/towhee',
            'icon': 'fab fa-github-square',
            'type': 'fontawesome',
        }
   ],
    'pygment_light_style': 'tango',
    'pygment_dark_style': 'monokai',
    'navbar_center': [],
    'page_sidebar_items': [],
}
