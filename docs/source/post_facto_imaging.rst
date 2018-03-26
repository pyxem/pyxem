Post-facto Imaging
==================

'Virtual' images are formed post-facto from scanning electron diffraction data
by plotting the scattered intensity in a subset of pixels in the diffraction
plane as a function of probe position. This is achieved by specifying a region
of interest (roi) in the diffraction plane within which intensity is summed.

Post-facto imaging can be performed interactively, as follows:

.. code-block:: python

    >>> dp = pxm.load('nanowire_precession.hdf5')
    >>> roi = pxm.roi.CircleROI(cx=0.,cy=0, r_inner=0, r=0.07)
    >>> dp.plot_interactive_virtual_image(roi=roi)

Note: It is important to ensure that the SED data is well aligned when interpreting
contrast in virtual images.



.. figure:: images/vdf_example.png
