Diffraction Vector Analysis
===========================

The :py:class:`~.DiffractionVectors` class defines an object that contains
experimentally measured two-dimensional diffraction vectors as well as methods
that may be applied using these vectors.

:py:class:`~.DiffractionVectors` are obtained by applying
:py:method:`~.ElectronDiffraction.find_peaks` to an :py:class:`~.ElectronDiffraction`
object.

Plotting Diffraction Vectors
----------------------------



Vector Magnitudes & Phase Identification
----------------------------------------

Diffraction vectors correspond to reciprocal lattice vectors, to a first
approximation, with some absences due to lattice type. A list of allowed
diffraction vector magnitudes therefore provides a fingerprint for a particular
crystal lattice. The diffraction vector magnitudes present in a 4D-S(P)ED dataset
may be evaluated using :py:method:`~.DiffractionVectors.get_`.

.. figure:: images/peaks_histogram.png

Diffraction vector magnitude histograms effectively provide a denoised radial
profile of the diffraction data that may be used for phase identification by
comparison with


Vector Based Imaging & Segmentation
-----------------------------------

A complete set of diffraction contrast images revealing the spatial variation in
diffraction condition in a given 4D-S(P)ED dataset can be obtained by forming a
virtual diffraction contrast image with an integration window positioned at each
unique diffraction vector in the dataset. This is achieved using the
:py:method:`~.DiffractionVectors.get_virtual_images` method. The complete set of images obtained can be analyzed further to reveal
microstructure and may be used to segment the data by identifying diffraction
vectors produced by the same crystal.
