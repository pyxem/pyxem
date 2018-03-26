Electron Diffraction - Signal Class
===================================

pyXem provides a library of tools primarily developed for the analysis of
4D-S(P)ED data, although many methods are applicable to electron diffraction
data in general. 4D-S(P)ED datasets comprise many thousands of electron
diffraction patterns and the ElectronDiffraction() class provides a specialized
HyperSpy Signal() class for this data. If the data array is imagined as a
tensor, D, of rank n then entries are addressed by n indices, D_{i,j,...,n}.
The HyperSpy Signal() class allows some indices, or equivalently some axes, to
be defined as navigation axes and others to be defined as signal axes. In the
context of a 4D-S(P)ED data, the two axes corresponding to the real-space scan
dimensions (i, j) are set as navigation axes and the two axes corresponding to
the diffraction pattern plane (a, b) are set as signal axes, which can be
written:

.. code-block:: python

    <i, j | a, b>

There are numerous ways to obtain physical insight from 4D-S(P)ED data all of
which ultimately require the assignment of an atomic arrangement to each probe
position that explains the observed diffraction. Different approaches to achieve
this goal are summarized in the following schematic.

.. figure:: images/sed_analysis_scheme.png

To illustrate the data methods implemented in pyXem we will consider data from a
model system of GaAs nanowires comprising a series of twinned regions along its
length, as shown below. (We acknolwedge Prof. Ton van Helvoort, NTNU, Norway, for
providing these samples).

.. figure:: images/model_system.png


Alignment, Corrections & Calibration
------------------------------------

Experimental artifacts in 4D-S(P)ED commonly include: (1) geometric distortions
due to projection optics, (2) small translations of the direct beam in the diffraction
plane, and (3) recorded intensities that depend on the response of the detector.
Methods to correct these effects to a first order are made available in pyXem.

Projection distortions may be corrected by the application of an opposite image
distortion, often an affine transformation, to all recorded diffraction patterns.
The appropriate transformation may be determined using diffraction patterns acquired
from a reference sample and then applied using the apply_affine_transformation()
method.


Translation of the direct beam is corrected for by aligning the stack of diffraction
patterns. A simple and sufficiently successful routine to achieve this is to crop
a region around the direct beam and apply a two-dimensional alignment based on phase
correlation. This is achieved through the hyperspy method, \textit{align2D()}, which
incorporates a statistical estimation of the optimal alignment position. Intensity
corrections most simply involve gain normalization based on dark-reference and
bright-reference images. Such gain normalization may be performed in pyXem using
the \textit{apply\_gain\_normalisation()} method.



Radial Ingegration
------------------

Radial integration of a two-dimensional electron diffraction pattern about its
centre provides a one-dimensional plot of diffracted intensity as a function of
scattering angle. This integration can be performed on every diffraction pattern
in the 4D-S(P)ED dataset using the get_radial_profile() method, as follows:

Background Subtraction
----------------------

Background subtraction is important for: (1) extracting accurate diffracted
intensities and (2) achieving reliable pattern matching or peak finding. The aims
in these two cases are significantly different. In case (1) the focus is on
intensity and the goals are to remove diffuse background from integrated Bragg
intensities and to reveal any structured diffuse scattering. In case (2) the
focus is on subsequent image processing and the goal is to obtain a representation
of the data with enhanced peaks and ideally zero intensity between peaks that is
more amenable to subsequent analysis even if intensities are compromised.
Background subtraction may be achieved in pycrystem via the remove_background()
method, which has multiple options. The background may be modelled using a model
containing a Lorentzian to model the direct beam, an exponential function to model
the tail of diffuse scattering, and a first order polynomial to model slower decay
at higher scattering angles, which was empirically found to give a good fit, as
illustrated in Figure \ref{fig:initial-processing}e,f. A morphological h-dome
method is also implemented. This involves forming a \textit{seed} image by
subtracting a constant offset, \textit{h}, from the raw image. A morphological
reconstruction by dilatation is then performed in which high-intensity values
replace nearby low intensity values. The seed image specifies the values that are
subject to dilatation and the raw image specifies the maximum value at each pixel.
The reconstructed image then appears similar to the original image but with peak
above the \textit{h} value cut off.

Peak Finding
------------

Electron diffraction patterns contain strong scattering near to particular
diffraction vectors, corresponding to the Bragg diffraction conditions. Finding
the position of these diffraction vectors, is therefore the initial step in many
subsequent analyses and amounts to a peak finding problem either because the
recorded diffraction patterns approximate sharp spot patterns or because they
can be made to e.g. by applying a Gaussian convolution to a pattern containing
intense disks. Peak finding in two dimensional signals (or images) is a general
problem and numerous methods are implemented in the find_peaks() method.

Unsupervised Machine Learning
-----------------------------

Usupervised machine learning algorithms may be applied to SED as a route to
obtain representative component diffraction patterns
