Post-facto Imaging
==================

Post-facto imaging refers to the formation of images from a scanning electron
diffraction dataset by applying some operation to the diffraction pattern
associated with each pixel in the scan. Numerous operations may be applied to
the diffraction data and a number of methods are implemented accordingly as
described in this section. Many operations involve selecting a subset of pixels
in the diffraction pattern and this is achieved in pyXem using the roi
functionality implemented in HyperSpy.

.. code-block:: python

    >>> dp = pxm.load('')
    >>> dp.add_gaussian_noise(std=100)

Virtual bright-field imaging
----------------------------

Vitual diffraction imaging involves plotting the intensity within a subset of
pixels in the recorded diffraction patterns as a function of probe position.


Virtual annular dark-field imaging
----------------------------------



Virtual dark-field imaging
--------------------------
