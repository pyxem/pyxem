Electron Diffraction - Signal Class
===================================

pyXem provides a library of tools primarily developed for the analysis of
4D-S(P)ED data, although many methods are applicable to electron diffraction
data in general. 4D-S(P)ED datasets comprise many thousands of electron
diffraction patterns and the ElectronDiffraction() class provides a specialized
HyperSpy Signal() class for this data. If the data array is imagined as a
tensor, D, of rank n then entries are addressed by n indices, D_{i,j,...,n}.
The HyperSpy Signal() class allows some indices, or equivalently some axes, to
be defined as navigation axes and others to be defined as signal axes. In the
context of a 4D-S(P)ED data, the two axes corresponding to the real-space scan
dimensions (i, j) are set as navigation axes and the two axes corresponding to
the diffraction pattern plane (a, b) are set as signal axes, which can be
written:

<i, j | a, b>

There are numerous ways to obtain physical insight from 4D-S(P)ED data all of
which ultimately require the assignment of an atomic arrangement to each probe
position that explains the observed diffraction. Different approaches to achieve
this goal are summarized in the following schematic.

.. figure:: docs/images/sed_analysis_scheme.png


Alignment & Calibration
-----------------------



Background Subtraction
----------------------



Peak Finding
------------



Unsupervised Learning
---------------------
