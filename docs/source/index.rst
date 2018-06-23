.. pyxem documentation master file, created by
   sphinx-quickstart on Fri Sep 16 14:34:23 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. figure:: images/forkme_right_orange_ff7600.png
    :align: right
    :target: https://github.com/pyxem/pyxem

pyXem - Pythonic Crystallographic Electron Microscopy
=====================================================

.. image:: https://travis-ci.org/pyxem/pyxem.svg?branch=master
    :target: https://travis-ci.org/pyxem/pyxem

.. image:: https://coveralls.io/repos/github/pyxem/pyxem/badge.svg?branch=master
    :target: https://coveralls.io/github/pyxem/pyxem?branch=master

pyXem is an open-source Python library for crystallographic electron microscopy.
The code is primarily developed as a platform for hybrid diffraction-imaging
microscopy based on scanning (precession) electron diffraction (S(P)ED) data.
This approach may be illustrated schematically, as follows:

.. figure:: images/sped_scheme.png
   :align: center
   :width: 600

pyXem builds heavily on the tools for multi-dimensional data analysis provided
by the `HyperSpy <http://hyperspy.org>`__ library and draws on `PyMatGen <http://pymatgen.org>`__
for atomic structure manipulation.

pyXem is released under the GPL v3 license.

Tutorial notebooks are available `here <https://github.com/pyxem/pyxem-demos>`__.

.. warning::

    The pyXem project is in early stages of development and there will be bugs.
    Many methods must be used with care and could be improved significantly. The
    developers take no responsibility for inappropriate application of the code
    or incorrect conclusions drawn as a result of their application. Users must
    take responsibility for ensuring results obtained are sensible. We hope that
    by releasing the code at this stage the community can move forward more quickly
    together.


Contents
========

.. toctree::
    :maxdepth: 2

    introduction
    post_facto_imaging
    contributing
    bibliography
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


.. _repository: https://github.com/pyxem/pyxem

New to Python?
--------------

If you are new to python the simplest way to install everything you need is using
`Anaconda <http://www.continuum.io/downloads>`__  ( make sure to install the
Python 3 version).

From a clean install the following commands to install everything you need should
be entered into the terminal, or anaconda prompt terminal in Windows::

        conda install hyperspy -c conda-forge

        conda install --channel matsci pymatgen Cython

        pip install transforms3d

        python setup.py install


Credits
=======

pyXem is developed by numerous `contributors <https://github.com/pyxem/pyxem/graphs/contributors>`__.


Citing pyXem
============

If pyXem has enabled significant parts of an academic publication, please
acknowledge that by citing the software. Until a specific publication is written
about pyXem please site the github URL: www.github.com/pyxem/pyxem

We also recommend that you cite `HyperSpy <http://hyperspy.org/hyperspy-doc/current/citing.html>`__
and `PyMatGen <http://pymatgen.org/#how-to-cite-pymatgen>`__


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
