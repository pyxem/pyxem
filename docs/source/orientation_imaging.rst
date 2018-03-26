Orientation Mapping
===================

Orientation imaging can be achieved in a number of ways based on the acquisition
of numerous electron diffraction patterns in the scanning transmission
electron microscope.

Pattern Matching
----------------

Pattern matching, whereby each diffraction pattern in the 4D-S(P)ED dataset is
compared to a library of pre-computed theoretical templates, has become the most
common method of assigning crystal phase and orientation to each recorded pattern.
The implementation described here, and incorporated in pyxem, follows work by
Rauch et al. [Rauch2005].

Diffraction Library Generation
******************************

Generating a library of theoretical templates involves specification of the
crystal phases and orientations to be included, followed by simulation of a
template diffraction pattern corresponding to each phase and orientation. Although
it might be appealing to include all known phases and orientations this will
result in a very library generation times. Thus, in practice, it is preferable to use prior
knowledge to limit the template library to a minimal but sufficient size. For phases, it is usually simple
limited to a small number of likely phases based on previous characterisation. The orientations to be
included are most generally selected as a uniform sampling of symmetry inequivalent
orientations corresponding to the crystallography of the given phases. The symmetry inequivalence is very important.
True uniform sampling of orientation space requires consideration of the topology of orientation space. Pyxem contains utility
functions, (eg - get_equispaced_so3_mesh()) to assist in this matter.

Simulation of template patterns may, in principle, be achieved using any of the
well-known kinematical or dynamical simulation methods. The complexity justified
depends on the precision and accuracy required as well as the extent to which an
ideal simulation can be properly compared with experimental data. In the pattern
matching framework presented here, a simple kinematical simulation is sufficient
and a library of kinematically simulated templates can be obtained from a list of
phases and orientations using the DiffractionLibraryGenerator() class in pyxem.


Pattern Matching Metric
***********************

Pattern matching between point like geometric templates and experimental spot
diffraction patterns is based on a correlation index,
Q_{i}, defined as:

.. math::
Q_{i} = \frac{\sum_{x,y} P(x,y) T_{i}(x,y)}{\sqrt[]{\sum_{x,y}P^{2}(x,y)}}}

where P(x,y) is the intensity of the pixel with coordinates (x,y) in the
experimental diffraction pattern and T_{i}(x,y) is the intensity of template i
at (x,y). This metric is in common use in orientation mapping.
