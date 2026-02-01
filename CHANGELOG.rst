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
- Add ``show_slice_on_plot`` argument to :meth:`pyxem.signals.Diffraction2D.get_direct_beam_position`. (#1148)
- Add ``spacing`` and ``spot_radius`` argument to :func:`pyxem.data.tilt_boundary_data` and add docstring. (#1148)
- Add :meth:`pyxem.signals.BeamShift.plot_on_signal` function to visualise beam position on signal plot. (#1148)
- Add ``subpixel`` precision for ``"blur"`` method in :meth:`pyxem.signals.Diffraction2D.get_direct_beam_position`. (#1148)
- Add Gaussian prefiltering in navigation space in :meth:`pyxem.signals.Diffraction2D.get_direct_beam_position`. (#1148)
- Implement lazy import to speed up ``pyxem`` import time and reduce memory usage by deferring the import of dependencies until they are actually needed. (#1176)


Changed
-------
- Drop support for python 3.8. (#1147)
- Refactor :meth:`~pyxem.signals.Diffraction2D.center_direct_beam` and :meth:`~pyxem.signals.Diffraction2D.shift_diffraction` to use consistent signature and avoid code duplication and  (#1144)
- Speed up documentation build by caching the gallery of examples and enable parallel build. (#1152)
- Set axes offset after centering in :meth:`pyxem.signals.Diffraction2D.center_direct_beam`. (#1148)

Fixed
-----
- Add explicit support for python 3.12 and 3.13 and update test matrix. (#1147)
- Silence axes warning in functions using :meth:`~hyperspy.api.signals.BaseSignal.map`. (#1168)


Deprecated
----------
- The ``shift_x`` and ``shift_y`` arguments in :meth:`~pyxem.signals.Diffraction2D.shift_diffraction` are deprecated in favor of the ``shifts`` argument. (#1144)
- The ``interpolation_order`` argument in :meth:`~pyxem.signals.Diffraction2D.shift_diffraction` have been renamed to ``order`` to be consistent with the signature of :meth:`~pyxem.signals.Diffraction2D.center_direct_beam` and :func:`scipy.ndimage.shift`. (#1144)


2025-06-04 - version 0.21.0
===========================
Added
-----
- Added examples for strain mapping and FEM (#1138)
- Added a new ZrCuAl dataset (#1138)
- Added a `get_strain_maps` method to simplify strain mapping (#1138)
- Added the ability to pass units for beam energy and camera length. (#1139)
- Added methods for calculating electric field. (#1139)
- Added a method for showing a 1d profile (#1139)
- Added new examples for loading binary data, calibrating and plotting with different units (#1139)


Fixed
-----
- Fixed bug in strain mapping for passing `DiffractionVectors` as a basis (#1138)
- Changed `dqe` to `gain` in Azimuthal integration (#1138)

2024-12-09 - version 0.20.0
===========================
Fixed
-----
- Fixed inconsistency with vector markers' positions when plotting orientation mapping results (#1126)
- Speed up creation of markers for Orientation Mapping (#1125)

Added
-----
- Added Examples for general plotting functions focusing on plotting diffraction patterns (#1108)
- Added support for marker plotting for multi-phase orientation mapping results (#1092)
- Cleaned up the Documentation for installation (#1109)
- Added an Example for determining the calibration (#1085)
- Added support for multi-phase orientation mapping markers on polar signals (#1126)
- Added support for numpy 2.0.0

Removed
-------
- Removed Dependency on pyfai.  Azimuthal integration is all handled internally (#1103)


2024-06-10 - version 0.19.1
===========================
Fixed
-----
- Fixed bugs releated to orientation mapping with multiple phases and the plotting of the
  vector annotations. (#1090)

Added
-----
- Added a way to fit linear planes in :meth:`pyxem.signals.BeamShift.get_linear_plane` by minimizing magnitude variance. (#1116)
- In :meth:`pyxem.signals.BeamShift.get_linear_plane` exposed initial values for optimization to make it more flexible. (#1116)

2024-06-08 - version 0.19.0
===========================

Restructuring of DPC processing
-------------------------------
- Total restructure of data processing of DPC data. This has now all been moved to the :class:`pyxem.signals.BeamShift` class.
- `DPCBaseSignal`, `DPCSignal1D`, `DPCSignal2D`, `LazyDPCBaseSignal`, `LazyDPCSignal1D`, `LazyDPCSignal2D` has been deprecated.
- `Diffraction2D.center_of_mass` has been deprecated. The functionality now resides in :meth:`pyxem.signals.Diffraction2D.get_center_beam_position`. Use `get_center_beam_position(method="center_of_mass")`
- Several `dummy_data` functions has been renamed to reflect this change: `dpc_signal` to `beam_shift_signal`
- `get_color_image_with_indicator` has been renamed and moved, :func:`pyxem.utils.plotting.plot_beam_shift_color`
- `correct_ramp` has been deprecated, with the functionality now residing in :meth:`pyxem.signals.BeamShift.get_linear_plane`. You can use `s_bs_lp = s_bs.get_linear_plane(fit_corners=0.05)`, then `s_bs -= s_bs_lp` to correct for the dscan shifts
- `:meth:`pyxem.signals.BeamShift.make_linear_plane` is being deprecated, and replaced with `:meth:`pyxem.signals.BeamShift.get_linear_plane`. It now returns a new signal, instead of altering the old one.
- `gaussian_blur` and `flip_axis_90_degrees` has been deprecated, as this can easily be done using `s.map`

Fixed
-----
- Fixed indexing error in :meth:`~pyxem.signals.Diffraction2D.get_direct_beam_position` (#1080)

Added
-----
- Added Examples for doing a Circular Hough Transform and Increased Documentation for Filtering Data (#1082)
- Added `circular_background` to :meth:`~pyxem.signals.Diffraction2D.template_match_disk` to account for
  an amorphous circular background when template matching (#1084)
- Added new datasets of in situ crystalization, Ag SPED,
  Organic Semiconductor Orientation mapping, Orientation Mapping, and DPC (#1081)
- Added a new method for calibrating the camera length
  based on a :class:`pyxem.signals.ElectronDiffraction1D` signal (#1085)
- Added Vectors to mask in :meth:`~pyxem.signals.DiffractionVectors.to_mask` (#1087)
- Add :meth:`pyxem.signals.PolarDiffraction2D.get_orientation` to get the phase orientation (#1073)
- Add :class:`pyxem.signals.OrientationMap` to organize and visualize orientation results (#1073)

2024-05-08 - version 0.18.0
===========================
Fixed
-----
- Fixed pytest failure. Changed ``setup`` --> ``setup_method`` (#997)
- :meth:`pyxem.signals.Diffraction2D.center_of_mass` now uses the :meth:`hyperspy.api.BaseSignal.map` function. (#1005)
- Replace ``matplotlib.cm.get_cmap`` (removed in matplotlib 3.9) with ``matplotlib.colormaps``. (#1023)
- Documentation fixes and improvement. (#1028)
- Fixed bug with flattening diffraction Vectors when there are different scales (#1024)
- Fixed intersphinx links and improved api documentation (#1056)
- Fix an off-by-one error in the :meth:`pyxem.signals.Diffraction2D.get_azimuthal_integral2d` (#1058)
- Fix handling of azimuthal range in :meth:`pyxem.signals.Diffraction2D.get_azimuthal_integral2d` (#1060)

Added
-----
- Added :class:`pyxem.utils.calibration_utils.Calibration` class  for calibrating the signal axes of a 4-D STEM dataset(#993)
- Added :meth:`pyxem.signals.Diffraction2D.get_azimuthal_integral1D` method to calculate the azimuthal integral of a 2D diffraction pattern (#1008)
- Added example for doing azimuthal integration of a 2d diffraction pattern (#1009)
- Added :meth:`pyxem.signals.CommonDiffraction.get_virtual_image` method to calculate multiple virtual images
  from a 4D STEM dataset (#1014)
- Added GPU support for lazy signals. (#1012)
- Added GPU processing for :meth:`pyxem.signals.Diffraction2D.get_azimuthal_integral2d` (#1012)
- Added :meth:`pyxem.signals.Diffraction2D.get_diffraction_vectors` to directly return the diffraction vectors (#1053)
- Added method for calibrating the detector gain (#1046)
- Added :meth:`pyxem.signals.PolarDiffraction2D.subtract_diffraction_background` for polar-specific background subtraction (#1062)

Deprecated
----------
- The module & all functions within ``utils.reduced_intensity1d`` are deprecated in favour of using the methods of `ReducedIntensity1D` (#994).
- Deprecated ``CalibrationGenerator`` and ``CalibrationLibrary`` in favour of :class:`pyxem.utils.calibration.Calibration` class (#1000)
- Detector module as we move away from pyfai
- Deprecated ``pyxem.generators.virtual_image_generator.VirtualImageGenerator`` in
  favor of  :meth:`pyxem.signals.CommonDiffraction.get_virtual_image` (#1014)
- Several utility modules have been deprecated: utils.{pyfai,segement,virtual_images,background_utils,cluster_tools,signals, radial_utils} (#1030 & #1060, #1055).
- The following utils: ``insitu_utils``, ``correlations_utils`` and ``pixelated_stem_tools``, ``dask_tools`` are now private.
- ``utils.vector_utils``, prefer ``utils.vectors``
- ``utils.symmetric_vector_utils``, prefer ``utils.vectors``
- ``utils.labeled_vector_utils``, prefer ``utils.vectors``
- ``utils.expt_utils``, prefer ``utils.diffraction``

Deleted
-------
- Several expired methods of :class:`pyxem.signals.Diffraction2D` associated with radial integration (#998)
- The ``peak_find_lazy`` method of :class:`pyxem.signals.Diffraction2D`  (#1040)
- ``dummy_data`` that content is now available under ``data.dummy_data``

Changed
-------
- Subpixel refinement now a function of the :py:class:`~pyxem.signals.DiffractionVectors` class (#980)
- The :py:class:`~pyxem.generators.SubpixelrefinementGenerator` class has been deprecated (#980)



2024-01-05 - version 0.17.0
===========================
Added
-----
- LazyDiffractionVectors are now supported(#969)
- DiffractionVectors now support intensity(#969)
- Add Examples for vector_finding and determining_ellipticity(#969)
- Add slicing methods to DiffractionVectors using ``ivec`` (#972)
- :class:`~pyxem.signals.DiffractionVectors` now explicitly handles lazy signals (#972)
- Added html representation for non-lazy :class:`~pyxem.signals.DiffractionVectors` (#972)
- Added :class:`pyxem.signals.PolarVectors` for polar vectors (#981)
- Added clustering methods using :func:`pyxem.signals.DiffractionVectors.cluster` (#981)
- Added :class:`pyxem.signals.LabeledDiffractionVectors` for labeled diffraction vectors after clustering (#981)

Changed
-------
- Revised the pyxem logo banner and favicon (#988)

Fixed
-----
- Update pyxem to work with hyperspy 2.0.0 (#969)
- Fixed slow markers (#969)
- Removed parallel and max_workers keywords in favor of using dask (#969)
- :class:`~pyxem.signals.DiffractionVectors2D` now extends :class:`~pyxem.signals.DiffractionVectors`
  for a more consistent API (#972)
- Fix :meth:`~pyxem.data.zrnb_precipitate` dataset to point to proper dataset

Removed
-------
- Removed MIB reader (#979) in favor of https://github.com/hyperspy/rosettasciio/pull/174
- Support for Hyperspy 1.x.x is not supported.  Use pyxem 0.16.0 instead if using Hyperspy 1.x.x (#969)

2023-11-14 - version 0.16.0
===========================

Added
-----
- Added `magnitude_limits` to `DPCSignal2D` methods (#949)
- Added :py:class:`~pyxem.signals.LazyCorrelation1D` for lazy Correlation1D signals
- Improved the documentation and added Examples
- Add N-D filtering using :py:meth:`~pyxem.signals.Diffraction2D.filter`
- Add new :py:class:`pyxem.signals.InSituDiffraction2D` class for in-situ diffraction data

Fixed
-----
- Remove ``ipywidgets`` from requirements as it is not a dependency
- Set skimage != to version 0.21.0 because of regression
- Do not reverse the y-axis of diffraction patterns when template matching (#925)
- Fixed bug in :py:class:`pyxem.generators.indexation_generator.AcceleratedIndexationGenerator` when
  passing orientations as tuples.
- Fix bug in calculating strain (#958)




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
