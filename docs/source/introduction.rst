Electron Diffraction - Signal Class
===================================

pyXem provides a library of tools primarily developed for the analysis of
4D-S(P)ED data, although many methods are applicable to electron diffraction
data in general. 4D-S(P)ED datasets comprise many thousands of electron
diffraction patterns and the :py:class:`~.ElectronDiffraction` class provides a
specialized HyperSpy Signal() class for this data. If the data array is imagined
as a tensor, D, of rank n then entries are addressed by n indices, D_{i,j,...,n}.
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

Projection distortions may be (approximately) corrected by the application of an
opposite image distortion, often an affine transformation, to all recorded
diffraction patterns. The appropriate transformation may be determined using
diffraction patterns acquired from a reference sample and then applied using
:py:meth:`~.ElectronDiffraction.apply_affine_transformation`.

Translation of the direct beam is corrected for by aligning the stack of diffraction
patterns. A simple routine to achieve this is to crop a region around the direct
beam and apply a two-dimensional alignment based on phase correlation. This is
achieved through the method, align2D(), which incorporates a statistical estimation
of the optimal alignment position.

Intensity corrections most simply involve gain normalization based on dark-reference
and bright-reference images. Such gain normalization may be performed in pyXem using
:py:meth:`~.ElectronDiffraction.apply_gain_normalisation`.


Radial Ingegration
------------------

Radial integration of a two-dimensional electron diffraction pattern about its
centre provides a one-dimensional plot of diffracted intensity as a function of
scattering angle. This integration can be performed on every diffraction pattern
in a 4D-S(P)ED dataset using the :py:meth:`~.ElectronDiffraction.get_radial_profile`.


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

The :py:meth:`~.ElectronDiffraction.find_peaks` method provides access to a
number of algorithms for that achieve peak finding in two dimensional signals.
The methods available are as follows:

Zaeferrer peak finder
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> dp.find_peaks(method='zaefferer')

This algorithm was developed by Zaefferer [1]_ and the
implementation here is after the description of the algorithm in the Ph.D.
thesis of Thomas A. White. It is based on a gradient threshold followed by a
local maximum search within a square window, which is moved until it is
centered on the brightest point, which is taken as a peak if it is within a
certain distance of the starting point.

Ball statistical peak finder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> dp.find_peaks(method='stat')

Developed by Gordon Ball, and described in the Ph.D. thesis of Thomas A.
White, this method is based on finding points which have a statistically
higher value than the surrounding areas, then iterating between smoothing and
binarising until the number of peaks has converged. This method is slow, but
very robust to a variety of image types.

Matrix based peak finding
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> dp.find_peaks(method='laplacian_of_gaussians')
    >>> dp.find_peaks(method='difference_of_gaussians')

These methods are essentially wrappers around the
`scikit-image <http://scikit-image
.org/docs/dev/auto_examples/plot_blob.html>`_ Laplacian
of Gaussian and Difference of Gaussian methods, based on stacking the
Laplacian/difference of images convolved with Gaussian kernels of various
standard deviations. Both are very rapid and relatively robust, given
appropriate parameters.

Interactive Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> dp.find_peaks_interactive()

Many of the peak finding algorithms implemented here have a number of
tuneable parameters that significantly affect their accuracy and speed. Finding
the correct parameters can be difficult. An interactive tool for the Jupyter
(originally IPython) notebook has been developed to help.

Several widgets are available:

.. figure::  images/interactive_peaks.png
   :align: center
   :width: 600

* The method selector is used to compare different methods. The last-set
  parameters are maintained.
* The signal navigator is used where a signal has navigation axes. The
  randomizer will select random indices.
* The parameter adjusters will update the parameters of the method and re-plot
  the new peaks.

.. note:: Some methods take significantly longer than others, particularly
    where there are a large number of peaks to be found. The plotting window
    may be inactive during this time.


Unsupervised Machine Learning
-----------------------------

Usupervised machine learning algorithms may be applied to SED as a route to
obtain representative "component diffraction patterns" and their respective
"loadings" in real space. This is achieved through various decomposition methods:

.. code-block:: python

    >>> dp.decomposition()

The decomposition method is inherited directy from HyperSpy and is documented
`here <http://hyperspy.org/hyperspy-doc/current/user_guide/mva.html>`__.
