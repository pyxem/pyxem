Conventions
===========

Various conventions are adopted within pyXem, as detailed here.

Diffraction Pattern Coordinates
-------------------------------

Following alignment and calibration of two-dimensional diffraction data in the
ElectronDiffraction class, i.e.

.. code-block:: python

    >>> dp.set_diffraction_calibration(0.01)

Coordinates in the two-dimensional diffraction pattern are referred to an origin
at the center of the diffraction pattern and the positive quadrant is the lower
right hand quadrant.

.. figure:: images/cartesian_coordinates.png
   :align: center
   :width: 600



.. figure:: images/polar_coordinates.png
  :align: center
  :width: 600


Physical Units
--------------

When physical units are used it is anticipated that we have:

1) Diffraction plane units of reciprocal Angstroms i.e. g = 1/d.
2) Atomic structure coordinates in Angstroms.
3) Scan coordinates in nanometres.


Crystallographic Axes
---------------------


Rotations
---------

These are in the rzxz convention, as defined in transform3d. This means that we
rotate about z, then x and then z again but the axes of rotation follow the
sample, as illustrated below:

.. figure:: images/euler_angles.png
   :align: center
   :width: 600


References
----------

.. _[Zaeferrer2010]:

:ref:`[Zaeferrer2010] <[Zaeferrer2010]>`
  S. Zaefferer, “New developments of computer-aided crystallographic analysis
  in transmission electron microscopy research” J. Appl. Crystallogr., vol. 33,
  no. v, pp. 10–25, 2000.
  [`link <https://journals.iucr.org/j/issues/2000/01/00/hz0046/hz0046.pdf`_].
