Post-facto Imaging
==================

Post-facto imaging refers to the formation of images from a scanning electron
diffraction dataset by applying some operation to the diffraction pattern
associated with each pixel in the scan. Numerous operations may be applied to
the diffraction data and a number of methods are implemented accordingly as
described in this section. Many operations involve selecting a subset of pixels
in the diffraction pattern and this is achieved in pyXem using the roi
functionality implemented in HyperSpy.

Intensity Imaging
=================

Vitual diffraction imaging involves plotting the intensity within a subset of
pixels in the recorded diffraction patterns as a function of probe position.

Variance Imaging
================

Variance imaging involves plotting the variance within a subset of pixels in
the recorded diffraction patterns as a function of probe position.
