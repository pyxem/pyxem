Conventions
===========

Various conventions are adopted within pyXem, as detailed here.

Diffraction Pattern Coordinates
-------------------------------

Following alignment and calibration of two-dimensional diffraction data in the
ElectronDiffraction class, i.e.

.. code-block:: python

    >>> dp.set_diffraction_calibration(0.01)

Coordinates in the two-dimensional diffraction pattern are typically Cartesian
coordinates referred to an origin at the center of the diffraction pattern with
 the lower right hand quadrant positive. Coordinates may also be expressed as
 polar coordinates with the positive rotation an anticlockwise rotation. These
 conventions are depicted below:

.. figure:: images/axis_conventions_pyxem.png
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

Atomic structures are manipulated in pyXem using the diffpy.Structure module and
crystallographic conventions are therefore primarily inherited from there.
Unless otherwise stated it will be assumed that a crystal structures is
described in the standard setting as defined in the International Tables of
Crystallography. [IUCr]

Crystal orientations/rotations are typically described with respect to an
orthonormal basis, which must be related to the crystallographic basis in a
consistent manner [Rowenhorst]. In pyXem it is assumed that these axes are
related according to the metric tensor defined in a diffpy.structure.lattice
object.


Rotations
---------

These are in the rzxz convention, as defined in transform3d. This means that we
rotate about z (by alpha), then x (by beta) and then the new z (by gamma), as
illustrated below:

.. figure:: images/euler_angles.png
   :align: center
   :width: 600
