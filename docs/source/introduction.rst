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

<i, j | a, b>

There are numerous ways to obtain physical insight from 4D-S(P)ED data all of
which ultimately require the assignment of an atomic arrangement to each probe
position that explains the observed diffraction. Different approaches to achieve
this goal are summarized in the following schematic.

.. figure:: images/sed_analysis_scheme.png


Alignment & Calibration
-----------------------

Many analyses, from radial integration to pattern matching, require knowledge of
the diffraction pattern centre. It was found that a sufficiently accurate
estimate could often be obtained simply by blurring the central region of the
diffraction pattern with a broad Gaussian kernel to give a unique maximum and
then finding the position of this value, as follows:

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
problem and numerous methods are implemented in the \textit{find\_peaks()} method.

Unsupervised Machine Learning
-----------------------------
