.. pyxem documentation master file, created by
   sphinx-quickstart on Fri Sep 16 14:34:23 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
    :hidden:

    self
    conventions
    literature


.. figure:: images/forkme_right_orange_ff7600.png
    :align: right
    :target: https://github.com/pyxem/pyxem

Introduction
============

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
by the `HyperSpy <http://hyperspy.org>`__ library and draws on `DiffPy <http://diffpy.org>`__
for atomic structure manipulation. pyXem is released under the GPL v3 license.


Installation
------------

pyXem requires python 3 and conda (to install a Py3 compatitble version of diffpy) - we suggest using the python 3 version of `Miniconda <https://conda.io/miniconda.html>`__. and creating a new environment for pyxem using the following commands in the anaconda prompt:

      $ conda create -n pyxem
      $ conda activate pyxem

The following commands will install everything you need if entered into the anaconda promt (or terminal) when located in the pyxem directory:::

      $ conda install -c conda-forge diffpy.structure
      $ conda install -c anaconda cython
      $ conda install -c conda-forge spglib
      $ conda install -c conda-forge traits
      $ pip install . -r requirements.txt


Getting Started
---------------

To get started using pyxem, especially if you are unfamiliar with python, we recommend you use jupyter notebooks. Having installed pyxem as above a jupyter notebook can be opened using the following commands entered into an anaconda prompt or terminal:

      $ conda activate pyxem
      $ jupyter notebook

`Tutorials and Example Workflows <https://github.com/pyxem/pyxem-demos>`__ are curated as a series of jupyter notebooks that you can work through and then modify to perform many common analyses.


Documentation
-------------

`Documentation <http://pyxem.github.io/pyxem/pyxem>`__is available for all pyxem modules at the following links:

- `pyxem.signals <http://pyxem.github.io/pyxem/pyxem.signals>`__ - for manipulating raw data and analysis results.

- `pyxem.generators <http://pyxem.github.io/pyxem/pyxem.generators>`__ - for establishing simulation and analysis conditions.

- `pyxem.components <http://pyxem.github.io/pyxem/pyxem.components>`__ - for fitting in model based analyses.

- `pyxem.libraries <http://pyxem.github.io/pyxem/pyxem.libraries>`__ - that store simulation results needed for analysis.


4D-SED Analysis Workflows
-------------------------

pyXem provides numerous tools for the analysis of 4D-S(P)ED data comprising many 
thousands of electron diffraction patterns. The :py:class:`~.ElectronDiffraction` class 
is a specialized class for this data. If the data array is imagined
as a tensor, D, of rank n then entries are addressed by n indices, D_{i,j,...,n}.
The HyperSpy Signal() class allows some indices, or equivalently some axes, to
be defined as navigation axes and others to be defined as signal axes. In the
context of a 4D-S(P)ED data, the two axes corresponding to the real-space scan
dimensions (i, j) are set as navigation axes and the two axes corresponding to
the diffraction pattern plane (a, b) are set as signal axes, which can be
written:

.. code-block:: python

    >>> <i, j | a, b>

There are numerous ways to obtain physical insight from 4D-S(P)ED data all of which 
ultimately require the assignment of an atomic arrangement to each probe position that 
explains the observed diffraction. Different approaches to achieve this goal are 
summarized in the following schematic.

.. figure:: images/pyxem-flow.png
  :align: center
  :width: 600

.. warning::

    The pyXem project is under continual development and there may be bugs. All methods must be used with care.
