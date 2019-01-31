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
for atomic structure manipulation. pyXem is released under the GPL v3 license. One can find citation information, installation instructions and credits at our github page `here <https://github.com/pyxem/pyxem>`__. This website outlines some of the basic functionality avaliable in pyxem.

pyXem provides a library of tools primarily developed for the analysis of
4D-S(P)ED data, although many methods are applicable to electron diffraction
data in general. 4D-S(P)ED datasets comprise many thousands of electron
diffraction patterns and the :py:class:`~.ElectronDiffraction` class provides a
specialized HyperSpy Signal() class for this data. If the data array is imagined
as a tensor, D, of rank n then entries are addressed by n indices, D_{i,j,...,n}.
The HyperSpy Signal() class allows some indices, or equivalently some axes, to
be defined as navigation axes and others to be defined as signal axes. In the
context of a 4D-S(P)ED data, the two axes corresponding to the real-space scan
dimensions (i, j) are set as navigation axes and the two axes corresponding to
the diffraction pattern plane (a, b) are set as signal axes, which can be
written:

.. code-block:: python

    <i, j | a, b>

There are numerous ways to obtain physical insight from 4D-S(P)ED data all of which 
ultimately require the assignment of an atomic arrangement to each probe position that 
explains the observed diffraction. Different approaches to achieve this goal are 
summarized in the following schematic.

.. figure:: images/pyxem-flow.png
  :align: center
  :width: 600

.. warning::

    The pyXem project is under continual development and there may be bugs. All methods must be used with care.
