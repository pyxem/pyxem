# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Getting and plot integrated intensity now support signals containing nan (#722)
- Add Symmetry1D signal class and symmetry analysis methods (#724)

### Removed
- lazy_* virtual imaging has been removed, use get_integrated_intensity (#722)

## 2021-01-13 - version 0.13.0
### Added
- Faster rotation indexing, using in place speeds up, added as AcceleratedIndexationGenerator (#673)
- get_direct_beam_position now supports lazy proccessing (#648)
- center_direct_beam now supports lazy processing (#658)
- Several functions for processing large datasets using dask (#648, #658)
- Methods to retrieve phase from DPC signal are added (#662)
- Add VirtualImageGenerator.set_ROI_mesh method to set mesh of CircleROI (#700)
- Added a setup.cfg

### Changed
- The importing of pyxem objects has been standardized (#704)
- get_direct_beam_position now has reversed order of the shifts [y, x] to [x, y] (#653)
- .apply_affine_transform now uses a default order of 1 (changed from 3)
- find_peaks is now provided by hyperspy, method 'xc' now called 'template_matching'
- virtual_annular_dark_field and virtual_bright_field renamed; now have a "lazy_" prefixing (#698)
- Plotting large, lazy, datasets will be much faster now (#655)
- Calibration workflow has been altered (see PR #640 for details)
- Azimuthal integration has been refactored (see PRs #625,#676 for details)

### Removed
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

## 2020-12-02 - version 0.12.3
### Changed
- CI is now provided by github actions
- Code now depends on hyperspy==1.6.1 and skimage>=0.17.0

## 2020-10-04 - version 0.12.2
### Added
- This project now keeps a Changelog

### Changed
- Slow tests now don't run by default
- Depend only on hyperspy-base and pyfai-base
