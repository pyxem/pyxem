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
   :align: center
   :width: 600

To illustrate the data methods implemented in pyXem we will consider data from a
model system of GaAs nanowires comprising a series of twinned regions along its
length, as shown below. (We acknowledge Prof. Ton van Helvoort, NTNU, Norway, for
providing these samples).

.. figure:: images/model_system.png
   :align: center
   :width: 600

The methods described in this documentation are demonstrated in a series of
[Jupyter Notebooks](http://jupyter.org/), which can be used as analysis
templates on which to build. These are available `here <https://github.com/pyxem/pyxem-demos>`__.

Experimental parameters associated with the data acquisition can be stored in
metadata for future reference using the utility function
:py:meth:`~.ElectronDiffraction.set_experimental_parameters`, for example:

.. code-block:: python

    >>> dp.set_experimental_parameters(accelerating_voltage=300.,
                                       camera_length=21.,
                                       scan_rotation=277.,
                                       convergence_angle=0.7,
                                       exposure_time=10.)


Alignment, Corrections & Calibration
------------------------------------

Experimental artifacts in 4D-S(P)ED commonly include: (1) geometric distortions
due to projection optics, (2) small translations of the direct beam in the
diffraction plane, and (3) recorded intensities that depend on the response of
the detector. Methods to correct these effects to a first order approximation
are made available in pyXem.

Projection distortions may be (approximately) corrected by the application of an
opposite image distortion, often an affine transformation, to all recorded
diffraction patterns. The appropriate transformation may be determined using
diffraction patterns acquired from a reference sample and then applied using
:py:meth:`~.ElectronDiffraction.apply_affine_transformation`. E.g.

.. code-block:: python

    >>> dp.apply_affine_transformation(np.array([[0.99,0   ,0],
                                                 [0   ,0.69,0],
                                                 [0   ,0   ,1]]))

Translation of the direct beam is corrected for by aligning the stack of
diffraction patterns. This can be achieve with

.. code-block:: python

    >>> dp.center_direct_beam()

This method has an argument (sigma) that should be smaller (in pixel terms) than the distance from the edge of
the nearest diffraction spot to the direct beam. Furthermore, the code assumes the direct beam is brightest spot.

Intensity corrections most simply involve gain normalization based on
dark-reference and bright-reference images. Such gain normalization may be
performed in pyXem using :py:meth:`~.ElectronDiffraction.apply_gain_normalisation`.

.. code-block:: python

    >>> dp.apply_gain_normalisation(bref=bright_reference, dref=dark_reference)

Following alignment and the application of necessary corrections to the data (ESSENTIAL DO NOT SKIP!), one
may be calibrate the signals. Utility functions exist to apply calibrations to the diffraction and scan axes respectively.

.. code-block:: python

    >>> dp.set_diffraction_calibration(0.01)
    >>> dp.set_scan_calibration(10)

.. note:: The diffraction axes should be calibrated in A^{-1}/px and the scan
    axes should be calibrated in nm/px.


Radial Integration
------------------

The :py:meth:`~.ElectronDiffraction.get_radial_profile` method integrates every
two-dimensional electron diffraction pattern about its and is applied as:

.. code-block:: python

    >>> dp.get_radial_profile()

The result is a one-dimensional plot of diffracted intensity as a function of
scattering angle.

.. figure:: images/radial_profile.png
   :align: center
   :width: 400

Again, this will not work if you fail to center all of the patterns in your S(P)ED scan.

Background Removal
------------------

Background subtraction is important for extracting accurate diffracted
intensities and achieving reliable pattern matching or peak finding. The aims in
these two cases are significantly different. Background subtraction may be
achieved in pyXem via the :py:meth:`~.ElectronDiffraction.remove_background`
method, which has multiple options.


Background Modelling
^^^^^^^^^^^^^^^^^^^^

The background may be modelled by fitting a model to the radial profile of the
diffraction data. The model may then be made ciruclarly symmetric and subtracted.
Numerous models could in principle be used and one option that has been useful
for data acquired on fluorescent screens, but is difficult to justify physically,
contains a Lorentzian to model the direct beam, an exponential function to
model the tail of diffuse scattering, and a first order polynomial to model
slower decay at higher scattering angles. This is applied as:

.. code-block:: python

    >>> dp.remove_background(method='model')

Backgound modelling, as described above yields the following:

.. figure:: images/background_model.png
   :align: center
   :width: 600


Morphological Background Removal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Background removal based on morphological operations provides a fast and
versatile method for removing non-smooth background. A so-called h-dome method
is implemented here. This involves forming a 'seed' image by subtracting a
constant offset, h, from the raw image. A morphological reconstruction by
dilatation is then performed in which high-intensity values replace nearby low
intensity values. The seed image specifies the values that are subject to
dilatation and the raw image specifies the maximum value at each pixel. The
reconstructed image then appears similar to the original image but with peak
above the h value cut off.

.. code-block:: python

    >>> dp.remove_background(method='h-dome', h=0.4)

Morphological background removal, as described above yields the following:

.. figure:: images/background_morphological.png
   :align: center
   :width: 600


Peak Finding
------------

The :py:meth:`~.ElectronDiffraction.find_peaks` method provides access to a
number of algorithms for that achieve peak finding in electron diffraction
patterns. The found peak positions are returned as
The methods available are as follows:

Zaeferrer peak finder
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> dp.find_peaks(method='zaefferer')

This algorithm was developed by Zaefferer and the implementation here is after
the description of the algorithm in the Ph.D. thesis of Thomas A. White. It is
based on a gradient threshold followed by a local maximum search within a square
window, which is moved until it is centered on the brightest point, which is
taken as a peak if it is within a certain distance of the starting point.

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

Many of the peak finding algorithms implemented here have a number of tuneable
parameters that significantly affect their accuracy and speed. Finding the
correct parameters can be difficult. An interactive tool for the Jupyter
notebook has been developed to help.

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

Unsupervised machine learning algorithms may be applied to SED as a route to
obtain representative "component diffraction patterns" and their respective
"loadings" in real space. These methods involve unfolding each diffraction
pattern into an image vector and stacking these vectors together to construct a
data matrix, which is then factorized:

.. figure::  images/ml_sed_scheme.png
   :align: center
   :width: 600

Various matrix decomposition methods are available through the decomposition()
method, which is inherited directy from HyperSpy and is documented
`here <http://hyperspy.org/hyperspy-doc/current/user_guide/mva.html>`__.

.. code-block:: python

    >>> dp.decomposition()
