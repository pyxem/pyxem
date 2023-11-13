=========
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.


Unreleased
==========

Added
-----
- Added `magnitude_limits` to `DPCSignal2D` methods (#949)
- Added :py:class:`~pyxem.signals.LazyCorrelation1D` for lazy Correlation1D signals
- Improved the documentation and added Examples
- Add N-D filtering using :py:meth:`~pyxem.signals.Diffraction2D.filter`

Fixed
-----
- Remove ``ipywidgets`` from requirements as it is not a dependency
- Set skimage != to version 0.21.0 because of regression
- Do not reverse the y-axis of diffraction patterns when template matching (#925)
- Fixed bug in :py:class:`pyxem.generators.indexation_generator.AcceleratedIndexationGenerator` when
  passing orientations as tuples.
- Fix bug in calculating strain (#958)

Added
-----
- Add n-d and 2-d filters #935 for filtering datasets

2023-05-08 - version 0.15.1
===========================

Fixed
-----
- Fixed type error in ``separate_watershed`` with scikit-image 0.21 (#921)
- Fixed VDF creation from peaks using generators.VirtualDarkFieldGenerator.get_virtual_dark_field_images (#926)
- Updating and correcting Zenodo (#924)
- Bug fix for center_direct_beam and `half_square_width` (#928 Thanks to @PVacek )


2023-04-06 - version 0.15.0
===========================

Added
-----
- Added damp_extrapolate_to_zero to ReducedIntensity1D
- Added in deprecation wrapper class to wrap deprecated functions in pyxem.
- Center-of-mass algorithm added to get_direct_beam_position (#845)
- Added `VectorSignal1D` class to handle 1 dimensional signals
- Added kwargs to find_beam_offset_cross_correlation allowing for parameters
to be passed to `phase_cross_correlation` (#907)
- Added `LazyVirtualDarkField` signal

Changed
-------
- Sklearn is now required to be on the 1.0 series.
- Changed `set_signal_dimension` to `Hyperspy.signals.BaseSignal.transpose`
- Moved code from `VectorSignal` to `VectorSignal2D`.  Change is more inline with stated dimensions
- `VectorSignal` pixel_calibration deprecated and replaced with scales.
- Fixed bugs resulting from API change in hyperspy/hyperspy#3045. Markers explicitly initialized
- DiffractionVectors.get_diffraction_pixels_map returns a ragged signal
- VirtualDarkFieldImage.get_vdf_segment changed to properly handle setting of axes
- Increased minimal version of scikit-image to >= 0.19.0
- Increased minimal version of Matplotlib to >= 3.3

Fixed
-----
- Fixed a factor of 1/2 missing in ScatteringFitComponentXTables
- Fixed error related to `DiffractionSignal2D.apply_affine_transformation` when multiple affine transformations are given. (#870)
- Bugfix related to Numpy 1.24.0. Strict array creation with dtype=object is needed
  for ragged arrays. (#880 & #881)
- Bug fix for doubling of inplane rotation in template matching.  (#905 & #853)
- Bug fix for filtering vectors using a basis and DBSCAN
- Bug fix for passing vector attributes when signal is copied or changed.


2022-06-15 - version 0.14.2
===========================

Changed
-------
- Increase minimal version of orix to >= 0.9.
- Increase minimal version of diffsims to >= 0.5.

Fixed
-----
- Fix bug in `get_DisplacementGradientMap` (#852)
- Fix template matching bugs (originally fixed in #771 but omitted from 0.14 series by accident)

2022-04-29 - version 0.14.1
===========================

Added
-----
- Getting and plot integrated intensity now support signals containing nan (#722)
- Add Symmetry1D signal class and symmetry analysis methods (#724)
- BeamShift class, which includes the `make_linear_plane` method for better correction of the beam shift when scanning large regions in STEM (#746)
- Add unit testing of docstring examples (#766)
- Add function for optimizing calibration of SPED data (#785)
- Add function for creating a orix CrystalMap from indexation results (#794)
- Speed optimizations for the fast template matching on CPU and GPU, improving speeds by 200% and 40% respectively (#796)
- Added the ability to determine the center and ellipticity using the `determine_ellipse` function.

Removed
-------
- lazy_* virtual imaging has been removed, use get_integrated_intensity (#722)
- `big_data_utils` has been removed as this is not the modern way of attacking this problem
- similarly, `TemplateIndexationGenerator` has been removed as the Accelerated approach is far better (#823)
Fixed
^^^^^
- Symmetry STEM Class updated to allow for better interpolation and lazy operation. (#809)
- Generalized plotting diffraction vectors on ND stacks of images (#783)
- Small bugfix with dask/cuda scheduler to prevent running out of VRAM (#779)
- Bugfix:AzimuthalIntegral1D accepts masks and uses updated `map` function (#826)

Deprecated
^^^^^^^^^^
- The `lazy_result` keyword, which has been changed to `lazy_output` to conform to similar keyword in HyperSpy

Changed
^^^^^^^
- For developers: HyperSpy's `.map` function will now be used to process big datasets, instead of pyXem's `process_dask_array`

2022-04-29 - version 0.14.0
===========================

The code contained in this version is identical to 0.14.1, the release was
recreated to fix an error with the Zenodo files.


2021-04-14 - version 0.13.2
===========================

Added
-----
- Code now support python 3.9
- Code now runs on hyperspy 1.6.2

Fixed
-----
- np.bool replaced by bool
- np.object replaced by object

2021-03-21 - version 0.13.1
===========================

Fixed
-----
- load_mib (#734)
- correct_bad_pixels now returns the same result when lazy/not-lazy (bug #723, fix #735)
- mirrored templates now correctly dealt with in radial template matching (#740)
- further bugfixes for AcceleratedIndexationGenerator (#744)
- a k-space error effecting azimuthal integration (#738)
- bug in .to_crystal_map()

Deprecated
----------
- lazy_virtual_bright_field, use get_integrated_intensity instead
- lazy_virtual_dark_field, use get_integrated_intensity instead

2021-01-13 - version 0.13.0
===========================

Added
-----
- Faster rotation indexing, using in plane speeds up, added as AcceleratedIndexationGenerator (#673)
- get_direct_beam_position now supports lazy processing (#648)
- center_direct_beam now supports lazy processing (#658)
- Several functions for processing large datasets using dask (#648, #658)
- Methods to retrieve phase from DPC signal are added (#662)
- Add VirtualImageGenerator.set_ROI_mesh method to set mesh of CircleROI (#700)
- Added a setup.cfg

Changed
-------
- The importing of pyxem objects has been standardized (#704)
- get_direct_beam_position now has reversed order of the shifts [y, x] to [x, y] (#653)
- .apply_affine_transform now uses a default order of 1 (changed from 3)
- find_peaks is now provided by hyperspy, method 'xc' now called 'template_matching'
- virtual_annular_dark_field and virtual_bright_field renamed; now have a `lazy\_` prefixing (#698)
- Plotting large, lazy, datasets will be much faster now (#655)
- Calibration workflow has been altered (see PR #640 for details)
- Azimuthal integration has been refactored (see PRs #625,#676 for details)

Removed
-------
- Diffraction2D.remove_dead_pixels has been removed, use .correct_bad_pixels (#681)
- Diffraction2D.remove_background, has been moved to .subtract_diffraction_background (#697)
- The diffraction_component and scalable_reference_pattern modules have been removed (#674)
- local_gaussian_method for subpixel refinement has been removed
- utils.plot removed, functionality now in signals.diffraction_vectors
- utils.subpixelrefinement_utils removed, functionality in subpxielrefinement_generator
- utils.dpc_tools removed, either downstreamed to diffsims or up to differential_phase_contrast.py
- utils.diffraction_tools removed, downstreamed to diffsims
- utils.sim_utils removed, instead use the relevant diffsims functionality
- utils.calibration_utils removed, downstreamed to diffsims

2020-12-02 - version 0.12.3
===========================

Changed
-------
- CI is now provided by github actions
- Code now depends on hyperspy==1.6.1 and skimage>=0.17.0

2020-10-04 - version 0.12.2
===========================

Added
-----
- This project now keeps a Changelog

Changed
-------
- Slow tests now don't run by default
- Depend only on hyperspy-base and pyfai-base
