# Configuration file for the Sphinx documentation app.

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import inspect
import os
from os.path import relpath, dirname
import re
import sys

from pyxem import release_info
import pyxem
import hyperspy.api as hs


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.append("../")

# Project information
project = "pyxem"
copyright = release_info.copyright
author = release_info.author
version = release_info.version
release = release_info.version

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/logo.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/logo.ico"


master_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinxcontrib.bibtex",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.graphviz",
    "sphinx.ext.mathjax",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx_codeautolink",
    "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
    "nbsphinx",
]

# Create links to references within pyxem's documentation to these packages.
intersphinx_mapping = {
    "dask": ("https://docs.dask.org/en/stable", None),
    "diffpy.structure": ("https://www.diffpy.org/diffpy.structure", None),
    "diffsims": ("https://diffsims.readthedocs.io/en/stable", None),
    "hyperspy": ("https://hyperspy.org/hyperspy-doc/current", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "orix": ("https://orix.readthedocs.io/en/stable", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "skimage": ("https://scikit-image.org/docs/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "rosettasciio": ("https://hyperspy.org/rosettasciio/", None),
}

linkcheck_ignore = [
    "https://doi.org/10.1002/smll.201904738",  # 403 Client Error: Forbidden for url
    "http://dx.doi.org/10.1088/0965-0393/23/8/083501",  # 403 Client Error: Forbidden for url
    "https://doi.org/10.1111/jmi.12065",  # 403 Client Error: Forbidden for url
    "https://doi.org/10.1111/j.0022-2720.2004.01293.x",  # 403 Client Error: Forbidden for url
]


# Add any paths that contain templates here, relative to this directory.
templates_path = [
    "_templates",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files. This image also affects
# html_static_path and html_extra_path.

# The theme to use for HTML and HTML Help pages.  See the documentation for a
# list of builtin themes.
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/pyxem/pyxem",
    "header_links_before_dropdown": 7,
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "use_edit_page_button": True,
    "announcement": "Check out the Free "
    " <a href='https://www.ntnu.edu/web/temgemini/nordtemhub'>(Big) Data analysis of 4D-STEM</a> "
    " workshop in Trondheim, Norway, 11-13 June 2024",
}

html_context = {
    "github_user": "pyxem",
    "github_repo": "pyxem",
    "github_version": "main",
    "doc_path": "doc",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files, so
# a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# Syntax highlighting
pygments_style = "friendly"

napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

# Figure references
numfig = True

# nbsphinx configuration
# Taken from nbsphinx' own nbsphinx configuration file, with slight
# modification to point nbviewer and Binder to the GitHub master links
# when the documentation is launched from a pyxem version with
# "dev" in the version.
if "dev" in version:
    release_version = "master"
else:
    release_version = "v" + version
# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = (
    r"""
{% set docname = 'doc/' + env.doc2path(env.docname, base=None) %}
.. raw:: html
    <style>a:hover { text-decoration: underline; }</style>
    <div class="admonition note">
      This page was generated from
      <a class="reference external" href="https://github.com/pyxem/pyxem-demos/blob/"""
    + f"{release_version}"
    + r"""/{{ docname|e }}">{{ docname|e }}</a>.
      Interactive online version:
      <span style="white-space: nowrap;"><a href="https://mybinder.org/v2/gh/pyxem/pyxem-demos/"""
    + f"{release_version}"
    + r"""?filepath={{ docname|e }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>.</span>
      <script>
        if (document.location.host) {
          $(document.currentScript).replaceWith(
            '<a class="reference external" ' +
            'href="https://nbviewer.jupyter.org/url' +
            (window.location.protocol == 'https:' ? 's/' : '/') +
            window.location.host +
            window.location.pathname.slice(0, -4) +
            'ipynb">View in <em>nbviewer</em></a>.'
          );
        }
      </script>
    </div>
.. raw:: latex
    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
"""
)
# https://nbsphinx.readthedocs.io/en/0.8.0/never-execute.html
nbsphinx_execute = "never"  # auto, always, never
nbsphinx_kernel_name = "python3"
nbsphinx_allow_errors = True
exclude_patterns = ["_build", "**.ipynb_checkpoints", "examples/*/*.ipynb"]

# sphinxcontrib-bibtex configuration
bibtex_bibfiles = ["bibliography.bib"]


# -- Sphinx-Gallery---------------
# https://sphinx-gallery.github.io
sphinx_gallery_conf = {
    "backreferences_dir": "reference/generated",
    "doc_module": ("pyxem",),
    "examples_dirs": "../examples",  # path to your example scripts
    "gallery_dirs": "examples",  # path to where to save gallery generated output
    "filename_pattern": "^((?!sgskip).)*$",  # pattern to define which will be executed
    "ignore_pattern": "_sgskip.py",  # pattern to define which will not be executed
    "reference_url": {"pyxem": None},
    "show_memory": True,
}

autodoc_default_options = {
    "show-inheritance": True,
}

graphviz_output_format = "svg"


# -- Linkcode ----------------------------------
def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object.
    This is taken from SciPy's conf.py:
    https://github.com/scipy/scipy/blob/master/doc/source/conf.py.
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    startdir = os.path.abspath(os.path.join(dirname(pyxem.__file__), ".."))
    fn = relpath(fn, start=startdir).replace(os.path.sep, "/")

    if fn.startswith("pyxem/"):
        m = re.match(r"^.*dev0\+([a-f0-9]+)$", pyxem.__version__)
        pre_link = "https://github.com/pyxem/pyxem/blob/"
        if m:
            return pre_link + "%s/%s%s" % (m.group(1), fn, linespec)
        elif "dev" in pyxem.__version__:
            return pre_link + "master/%s%s" % (fn, linespec)
        else:
            return pre_link + "v%s/%s%s" % (pyxem.__version__, fn, linespec)
    else:
        return None


# sphinx.ext.autodoc
# ------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
autosummary_ignore_module_all = False
autosummary_imported_members = True
autodoc_typehints_format = "short"
autodoc_default_options = {
    "show-inheritance": True,
}

autosummary_generate = True


# Enable ipywidgets GUIs to avoid error when building the documentation
hs.preferences.GUIs.enable_ipywidgets_gui = True
