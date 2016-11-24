Strain Mapping
==============

Mapping strain and small changes in orientation with nanoscale resolution and
across micron size regions is a core strength of the scanning electron
diffraction approach.

Direct Indexation
=================

Direct indexation is the most commonly implemented method for achieving strain
mapping from scanning electron diffraction data. The premise of this approach is
to determine the recorded reciprocal lattice vectors in each diffraction pattern
and then calculate the affine transformation between these

Image Processing
================

The image processing approach involves optimisation of the two-dimensional
affine transformation that maps a selected reference diffraction pattern from
within the data onto every other diffraction pattern in the scanned dataset.
This can be viewed as a model fitting problem in which the model to the raw data
is a single pattern distored by a different affine transformation matrix at each
point in the scan and this is how the method is implemented. The model can be
set up and the reference pattern specified as follows:

:python

Multiple matching metrics may be used to find the optimal affine transformation
between the reference pattern and the other diffraction patterns in the data.

Forward Modelling
=================

Simulating the diffraction that agrees with the data by optimising the strain in
the structure directly is perhaps the most direct route to strain mapping. This
forward modelling approach involves simulating the diffraction from a structure
and optimising the deformation of that structure to best match the data. A model
is set up with an electron diffraction simulation as follows:

:python
