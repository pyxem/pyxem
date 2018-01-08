.. pyxem documentation master file, created by
   sphinx-quickstart on Fri Sep 16 14:34:23 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyXem's documentation!
=====================================

pyXem (Python Crystallographic Electron Microscopy) is an open-source Python
library for crystallographic electron microscopy.

pyXem builds on the tools for multi-dimensional data analysis provided by
the HyperSpy library for treatment of experimental electron diffraction data and
tools for atomic structure manipulation provided by the PyMatGen library.

pyXem is released under the GPL v3 license.

This version is currently under construction.


Contents
========

.. toctree::
    :maxdepth: 2

    introduction
    diffraction_simulation
    post_facto_imaging
    indexation_integration
    orientation_imaging
    strain_mapping
    modules


Installation
============

1. Download or clone the repository_.
    1. Unzip the archive if necessary.
2. In the root directory of pyxem, run::

        python setup.py install

.. note:: if installing in a virtualenv, ensure the virtualenv has been
    activated before running the above script. To check if the correct Python
    version is being used, run::

        which python

    and ensure the result is something like ``/path/to/your/venv/bin/python``


.. _repository: https://github.com/dnjohnstone/pyxem

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
