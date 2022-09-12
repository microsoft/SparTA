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
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# -- Project information -----------------------------------------------------
from datetime import datetime
project = 'SparTA'
copyright = f'{datetime.now().year}, Microsoft'
author = 'Microsoft'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    # 'sphinx.ext.doctest',
    'myst_parser',
    'sphinxawesome_theme',
]

myst_enable_extensions = ["colon_fence", "deflist", "replacements", "substitution"]

# doctest_global_setup = '''
# try:
#     import pandas torch
# except ImportError:
#     torch = None
# finally:
#     skip_doctest = torch is None or not torch.cuda.is_available()
# '''

#'sphinx.ext.mathjax', ??

# imgmath settings
# imgmath_image_format = 'svg'
# imgmath_font_size = 14

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst', '.md']
# source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'README.md']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = "sphinx_rtd_theme"

# html_theme = "sphinx_book_theme"
# html_theme_options = {
#     "announcement": "It is currently an <i>alpha</i> version and under active development",
#     "repository_url": "https://github.com/microsoft/SparTA",
#     "use_repository_button": True,
#     "use_fullscreen_button": False,
# }

html_theme = "sphinxawesome_theme"
# extra options from the sphinxawesome_theme
html_collapsible_definitions = True
html_awesome_docsearch = False
html_awesome_code_headers = True

# DocSearch is configured via an `.env` key here.
# You can also use the following dictionary
#
# docsearch_config = {
# "api_key": "",
# "container": "",
# "index_name": "",
# }
html_theme_options = {
    "show_scrolltop": True,
    "show_prev_next": True,
    "extra_header_links": {
        "repository on GitHub": {
            "link": "https://github.com/microsoft/SparTA",
            "icon": (
                '<svg style="height: 26px; margin-top: -2px;" viewBox="0 0 45 44" '
                'fill="currentColor" xmlns="http://www.w3.org/2000/svg">'
                '<path fill-rule="evenodd" clip-rule="evenodd" '
                'd="M22.477.927C10.485.927.76 10.65.76 22.647c0 9.596 6.223 17.736 '
                "14.853 20.608 1.087.2 1.483-.47 1.483-1.047 "
                "0-.516-.019-1.881-.03-3.693-6.04 "
                "1.312-7.315-2.912-7.315-2.912-.988-2.51-2.412-3.178-2.412-3.178-1.972-1.346.149-1.32.149-1.32 "  # noqa
                "2.18.154 3.327 2.24 3.327 2.24 1.937 3.318 5.084 2.36 6.321 "
                "1.803.197-1.403.759-2.36 "
                "1.379-2.903-4.823-.548-9.894-2.412-9.894-10.734 "
                "0-2.37.847-4.31 2.236-5.828-.224-.55-.969-2.759.214-5.748 0 0 "
                "1.822-.584 5.972 2.226 "
                "1.732-.482 3.59-.722 5.437-.732 1.845.01 3.703.25 5.437.732 "
                "4.147-2.81 5.967-2.226 "
                "5.967-2.226 1.185 2.99.44 5.198.217 5.748 1.392 1.517 2.232 3.457 "
                "2.232 5.828 0 "
                "8.344-5.078 10.18-9.916 10.717.779.67 1.474 1.996 1.474 4.021 0 "
                "2.904-.027 5.247-.027 "
                "5.96 0 .58.392 1.256 1.493 1.044C37.981 40.375 44.2 32.24 44.2 "
                '22.647c0-11.996-9.726-21.72-21.722-21.72" '
                'fill="currentColor"/></svg>'
            ),
        },
    },
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    # 'css/modify.css',
]
