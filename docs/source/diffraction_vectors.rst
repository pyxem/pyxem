Diffraction Vector Analysis
===========================

The DiffractionVectors class defines an object that contains experimentally
measured two-dimensional diffraction vectors as well as methods that may be
applied using these vectors. An object in this class is returned when peak
finding methods are applied to an ElectronDiffraction object. The class inherits
from the hyperspy Signal class enabling the navigation axes to be preserved and
for the number of peaks at each position to vary. The vector representation of
the diffraction data is the most compact way to consider the spot pattern
geometry. This geometry may be used to: identify phases present, identify
diffraction conditions for virtual image formation, and as a route to orientation
and strain mapping.

Vector Magnitudes & Phase Identification
----------------------------------------

Diffraction vectors correspond to reciprocal lattice vectors, to a first
approximation, with some absences due to lattice type. A list of allowed
diffraction vector magnitudes therefore provides a fingerprint for a particular
crystal lattice. The diffraction vector magnitudes present in a 4D-S(P)ED dataset
may be evaluated by considering the radially integrated diffraction profile both
locally and globally in analogy to powder X-ray diffraction. However, the sampling
over crystal orientations is typically limited in 4D-S(P)ED data meaning that peak
intensities will not be reliable and low volume fraction phases are difficult to
identify in such integrated data. An alternative is to plot a histogram of the
found diffraction vector magnitudes, as shown below. Such diffraction vector
histograms effectively provide a denoised view of the diffraction vectors present
and may be used for phase identification.

Vector Based Imaging & Segmentation
-----------------------------------

Globally unique diffraction vectors are of interest for the formation of
diffraction contrast images and the segmentation of individual crystallites based
on those images \cite{Meng}. To identify unique vectors in a 4D-S(P)ED dataset
can be found in this way using the get_unique_vectors() method, which returns a
new DiffractionVectors() object containing only the unique vectors.

A complete set of diffraction contrast images revealing the spatial variation in
diffraction condition in a given 4D-S(P)ED dataset can be obtained by forming a
virtual diffraction contrast image with an integration window positioned at each
unique diffraction vector in the dataset. This is achieved using the get_virtual_images()
method. The complete set of images obtained can be analyzed further to reveal
microstructure and may be used to segment the data by identifying diffraction
vectors produced by the same crystal.
