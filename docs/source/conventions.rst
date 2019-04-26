Conventions
===========

The following conventions are adopted within pyXem.

Physical Units
^^^^^^^^^^^^^^

When physical units are used it is anticipated that we have:

1) Diffraction plane units of reciprocal Angstroms i.e. g = 1/d.
2) Atomic structure coordinates in Angstroms.
3) Scan coordinates in nanometres.


Diffraction Pattern Coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Following alignment and calibration of two-dimensional diffraction data in the 
ElectronDiffraction class coordinates in the two-dimensional diffraction pattern 
are Cartesian coordinates referred to an origin at the center of the diffraction 
pattern with the lower right hand quadrant positive. Coordinates may also be expressed 
as polar coordinates with the positive rotation an anticlockwise rotation. These 
conventions are depicted below:

.. figure:: images/axis_conventions_pyxem.png
   :align: center
   :width: 600


Crystallographic Axes
^^^^^^^^^^^^^^^^^^^^^

Atomic structures are manipulated in pyXem using the diffpy.Structure module and
crystallographic conventions are therefore primarily inherited from there.
Unless otherwise stated it will be assumed that a crystal structures is
described in the standard setting as defined in the `International Tables for Crystallography <https://it.iucr.org/A/>`__

Crystal orientations/rotations are typically described with respect to an
orthonormal basis, which must be related to the crystallographic basis in a
consistent manner. For further discussion see, for example, the following
article by `Rowenhorst et al <http://iopscience.iop.org/article/10.1088/0965-0393/23/8/083501/meta>`__. In pyXem it is assumed that these axes are related according to the metric tensor defined in a
diffpy.structure.lattice object.


Rotations
^^^^^^^^^

These are in the rzxz convention, as defined in `transforms3d <https://matthew-brett.github.io/transforms3d/reference/transforms3d.euler.html>`__. This means that we
rotate about z (by alpha), then the new x (by beta), and finally the new z (by gamma),
as illustrated below:

.. figure:: images/euler_angles.png
   :align: center
   :width: 600
