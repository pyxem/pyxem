# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Added a setup.cfg
- get_direct_beam_position now supports lazy proccessing (#648)
- center_direct_beam now supports lazy processing (#658)
- Several functions for processing large datasets using dask (#648, #658)

### Changed
- Calibration workflow has been altered (see PR #640 for details)
- Azimuthal integration has been refactored (see PR #625 for details)
- get_direct_beam_position now has reversed order of the shifts [y, x] to [x, y] (#653)
- Plotting large, lazy, datasets will be much faster now (#655)
- Methods to retrieve phase from DPC signal are added (#662)
- .apply_affine_transform now uses a default order of 1 (changed from 3)

### Removed
- The local_gaussian_method for subpixel refinement
- utils.plot, functionality now in signals.diffraction_vectors
- utils.subpixelrefinement_utils, functionality to subpxielrefinement_generator
- utils.dpc_tools, downstreamed to diffsims or up to differential_phase_contrast.py
- utils.diffraction_tools, downstreamed to diffsims
- utils.sim_utils, use the relevant diffsims functionality

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
