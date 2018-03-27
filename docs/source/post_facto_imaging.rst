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

Virtual images are typically formed to obtain diffraction contrast. A virtual
bright-field (VBF) image is formed by plotting the intensity within a disk positioned
at the centre of the diffraction pattern, around the direct beam. Virtual dark-field
(VDF) images are formed by plotting the intensity within pixels not containing the
direct beam. This integration window may typically be either an annulus positioned
around the centre of the diffraction pattern, or a disk centered on a particular
diffracted beam. Examples of such virtual images are shown below.

.. figure:: images/vdf_example.png
   :align: center
   :width: 600
