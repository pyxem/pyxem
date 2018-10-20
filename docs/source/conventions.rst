Conventions
===========

.. warning::

    This page is under development - for discussion see the relevant pull request

On this page we detail some conventions adopted within the code framework.

Axes Conventions
----------------

As per hyperspy we have that (for reciprocal space)

1) The center of the image square is at (0,0)
2) The top left corner is in negative x (left to right) and negative y (top to bottom)


Rotations
---------

These are in the rzxz convention, as defined in transform3d. This means that we rotate about z, then x and then z again but the axes of rotation follow the sample.
