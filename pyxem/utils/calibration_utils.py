# -*- coding: utf-8 -*-
# Copyright 2016-2024 The pyXem developers
#
# This file is part of pyXem.
#
# pyXem is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyXem is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyXem.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import json

from diffsims.utils.sim_utils import get_electron_wavelength
import hyperspy.api as hs
from hyperspy.axes import UniformDataAxis

from pyxem.utils.indexation_utils import index_dataset_with_template_rotation
from pyxem.utils._azimuthal_utils import _get_control_points, _get_factors
from pyxem.utils._deprecated import deprecated


class Calibration:
    """
    This is a class to hold the calibration parameters for some dataset
    and the methods for calibrating the dataset.

    It is designed to hold information about the affine transformation
    and the mask to apply to the data for something like a beam stop.

    There are 2 ways to set the calibration:
    1. You can set the calibration with a known reciprocal space pixel size.  This will assume
       a flat ewald sphere.

    2. You can set the calibration directly with a known real pixel size, beam energy/wavelength,
       and detector distance. This is the most accurate method but requires detailed calibration
       information that is not trivial to acquire.  In most cases option 1 is sufficient.

    If you set the pixel size with `hyperSpy.axes_manager.signal_axes[0].scale = 0.1`
    it will default to the first method. The underlying data will be stored in the metadata
    of the signal to be used later.
    """

    def __init__(
        self,
        signal,
    ):
        """A class to hold the parameters for an Azimuthal Integrator.

        Parameters
        ----------
        affine: (3x3)
            The affine transformation to apply to the data
        mask:
            A boolean array to be added to the integrator.
        """
        self.signal = signal

    def __call__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def affine(self):
        """Set the affine transformation to apply to the data."""
        return self.signal.metadata.get_item("General.affine_transformation")

    @affine.setter
    def affine(self, affine):
        self.signal.metadata.set_item("General.affine_transformation", affine)

    @property
    def mask(self):
        """Set the mask to apply to the data."""
        return self.signal.metadata.get_item("General.mask")

    @mask.setter
    def mask(self, mask):
        self.signal.metadata.set_item("General.mask", mask)

    @property
    def scale(self):
        """Return the scale in pixels"""
        if self.flat_ewald:
            return [s.scale for s in self.signal.axes_manager.signal_axes]
        else:
            return None

    @scale.setter
    def scale(self, scale):
        """Set the scale in pixels"""
        if not self.flat_ewald:
            raise ValueError("Scale can only be set if the ewald sphere is flat")
        center = self.center  # save the center
        for ax, c in zip(self.signal.axes_manager.signal_axes, center):
            ax.scale = scale
            ax.offset = -c * scale

    @property
    def units(self):
        return [s.units for s in self.signal.axes_manager.signal_axes]

    @units.setter
    def units(self, units):
        # maybe consider converting the scales as well?
        for ax in self.signal.axes_manager.signal_axes:
            ax.units = units

    @property
    def wavelength(self):
        return self.signal.metadata.get_item("Acquisition_instrument.TEM.wavelength")

    @wavelength.setter
    def wavelength(self, wavelength):
        self.signal.metadata.set_item(
            "Acquisition_instrument.TEM.wavelength", wavelength
        )

    @property
    def beam_energy(self):
        try:
            return self.signal.metadata.get_item(
                "Acquisition_instrument.TEM.beam_energy"
            )
        except KeyError:
            return None

    @beam_energy.setter
    def beam_energy(self, beam_energy):
        self.signal.metadata.set_item(
            "Acquisition_instrument.TEM.beam_energy", beam_energy
        )
        self.signal.metadata.set_item(
            "Acquisition_instrument.TEM.wavelength",
            get_electron_wavelength(beam_energy),
        )

    def detector(
        self,
        pixel_size,
        detector_distance,
        beam_energy=None,
        wavelength=None,
        center=None,
        units="k_nm^-1",
    ):
        """
        Calibrate the signal with a known pixel size, detector distance, and beam energy/wavelength.


        This sets the signal axes to be a `~hyperspy.axes.DataAxis` instead of a UniformAxis to
        account for the Ewald sphere curvature. This is the most accurate method of calibration
        but requires a known beam energy/wavelength and detector distance or to calibrate the
        experimental configuration with a known standard.

        Parameters
        ----------
        pixel_size: float
            The pixel size in meter.
        detector_distance: float
            The detector distance in meter.
        beam_energy: float
            The beam energy in keV.
        wavelength: float
            The beam wavelength in nm^-1.
        center: list
            The center of the signal in pixels.
        unit: str
            The unit to calculate the radial extent with.
        """

        if beam_energy is None and wavelength is None:
            wavelength = self.wavelength
            if wavelength is None:
                raise ValueError("Must provide either beam_energy or wavelength")
        elif beam_energy is not None:
            wavelength = get_electron_wavelength(beam_energy)
            self.beam_energy = beam_energy
        self.wavelength = wavelength
        # We need these values to properly set the axes
        self.signal.metadata.set_item(
            "Acquisition_instrument.TEM.pixel_size", pixel_size
        )
        self.signal.metadata.set_item(
            "Acquisition_instrument.TEM.detector_distance", detector_distance
        )

        unit_factors = {
            "k_nm^-1": 1,
            "k_A^-1": 0.1,
            "q_nm^-1": 1 / (2 * np.pi),
            "q_A^-1": 0.1 / (2 * np.pi),
        }
        if units not in unit_factors.keys():
            raise ValueError(
                f"Unit {units} not recognized. Must be one of {unit_factors.keys()}"
            )
        if center is None:
            center = np.array(self.signal.axes_manager.signal_shape) / 2
        x_pixels = np.arange(-center[0], self.shape[0] - center[0])
        y_pixels = np.arange(-center[1], self.shape[1] - center[1])

        x_pixels *= pixel_size
        y_pixels *= pixel_size

        x_angles = np.arctan(x_pixels / detector_distance)
        y_angles = np.arctan(y_pixels / detector_distance)

        x_axes = np.sin(x_angles) * (1 / wavelength) * unit_factors[units]
        y_axes = np.sin(y_angles) * (1 / wavelength) * unit_factors[units]

        for ax, axis in zip(self.signal.axes_manager.signal_axes, [x_axes, y_axes]):
            if isinstance(ax, UniformDataAxis):
                ax.convert_to_non_uniform_axis()
            ax.axis = axis
        self.units = units

    @property
    def shape(self):
        return self.signal.axes_manager.signal_shape

    @property
    def flat_ewald(self):
        """If the ewald sphere is flat return True"""
        return isinstance(self.signal.axes_manager.signal_axes[0], UniformDataAxis)

    def __repr__(self):
        rep_str = f"Calibration for {self.signal}, Ewald sphere: "
        if self.flat_ewald:
            rep_str += "flat"
        else:
            rep_str += f"curved"
        rep_str += f", shape: {self.shape}, affine: {self.affine is not None},"
        rep_str += f" mask: {self.mask is not None}"
        return rep_str

    @property
    def axes(self):
        return [ax.axis for ax in self.signal.axes_manager.signal_axes]

    def get_slices2d(self, npt, npt_azim, radial_range=None):
        """Get the slices and factors for some image that can be used to
        slice the image for 2d integration.

        Parameters
        ----------
        npt: int
            The number of radial points
        npt_azim:
            The number of azimuthal points
        """
        # get the max radial range from the axes
        if radial_range is None:
            from itertools import combinations

            edges = np.reshape([[ax[0] ** 2, ax[-1] ** 2] for ax in self.axes], -1)
            max_range = np.max(
                np.power(np.sum(list(combinations(edges, 2)), axis=1), 0.5)
            )
            radial_range = (0, max_range)
        # Get the slices and factors for the integration
        slices, factors, factors_slice = self._get_slices_and_factors(
            npt, npt_azim, radial_range
        )
        return slices, factors, factors_slice, radial_range

    def _get_slices_and_factors(self, npt, npt_azim, radial_range):
        # get the points which bound each azimuthal pixel
        control_points = _get_control_points(npt, npt_azim, radial_range, self.affine)

        # get the min and max indices for each control point using the axes
        min_x = (
            np.min(
                np.searchsorted(self.axes[0], control_points[:, :, 0], side="left"),
                axis=1,
            ).astype(int)
            - 1
        )
        min_y = (
            np.min(
                np.searchsorted(self.axes[1], control_points[:, :, 1], side="left"),
                axis=1,
            ).astype(int)
            - 1
        )

        max_x = np.max(
            np.searchsorted(self.axes[0], control_points[:, :, 0], side="right"), axis=1
        ).astype(int)
        max_y = np.max(
            np.searchsorted(self.axes[1], control_points[:, :, 1], side="right"), axis=1
        ).astype(int)
        # Note that if a point is outside the range of the axes it will be set to the
        # maximum value of the axis+1 and 0 if it is below the minimum value of the axis.
        slices = np.array(
            [[mx, my, mxx, myy] for mx, my, mxx, myy in zip(min_x, min_y, max_x, max_y)]
        )
        max_y_ind = len(self.axes[1]) - 1
        max_x_ind = len(self.axes[0]) - 1

        # set the slices to be within the range of the axes.  If the entire slice is outside
        # the range of the axes then set the slice to be the maximum value of the axis
        slices[slices[:, 0] > max_x_ind, 0] = max_x_ind
        slices[slices[:, 1] > max_y_ind, 1] = max_y_ind
        slices[slices[:, 2] > max_x_ind, 2] = max_x_ind
        slices[slices[:, 3] > max_y_ind, 3] = max_y_ind

        slices[slices[:, 0] < 0, 0] = 0
        slices[slices[:, 1] < 0, 1] = 0
        slices[slices[:, 2] < 0, 2] = 0
        slices[slices[:, 3] < 0, 3] = 0

        factors, factors_slice = _get_factors(control_points, slices, self.axes)
        return slices, factors, factors_slice

    @property
    def center(self):
        """Return the center in pixels"""
        if self.flat_ewald:
            return [-s.offset / s.scale for s in self.signal.axes_manager.signal_axes]
        else:
            return [
                np.argmin(np.abs(s.axis)) for s in self.signal.axes_manager.signal_axes
            ]

    @center.setter
    def center(self, center=None):
        """Set the center in pixels"""
        if not self.flat_ewald:
            raise ValueError(
                "To set the center of a curved ewald sphere "
                "use the s.calibrate.detector method"
            )
        if center is None:
            for ax in self.signal.axes_manager.signal_axes:
                ax.offset = -ax.scale * (ax.size / 2)
        else:
            for ax, off in zip(self.signal.axes_manager.signal_axes, center):
                ax.offset = -off * ax.scale

    def to_pyfai(self):
        """
        Convert the calibration to a pyfai AzimuthalIntegrator.

        Returns
        -------
        """
        from pyFAI.detectors import Detector
        from pyxem.utils.pyfai_utils import _get_setup, get_azimuthal_integrator

        if self.flat_ewald:
            pixel_scale = self.scale
            if self.wavelength is None:
                raise ValueError(
                    "The wavelength must be set before converting to a pyfai AzimuthalIntegrator"
                )
            setup = _get_setup(
                wavelength=self.wavelength,
                pyxem_unit=self.units[0],
                pixel_scale=pixel_scale,
            )
            detector, dist, radial_range = setup
        else:
            pixel_size = self.signal.metadata.get_item(
                "Acquisition_instrument.TEM.pixel_size"
            )
            dist = self.signal.metadata.get_item(
                "Acquisition_instrument.TEM.detector_distance"
            )
            if pixel_size is None or dist is None:
                raise ValueError(
                    "The dector must be first initialized with the s.calibrate.detector method"
                )
            detector = Detector(
                pixel1=pixel_size,
                pixel2=pixel_size,
            )

        ai = get_azimuthal_integrator(
            detector=detector,
            detector_distance=dist,
            shape=self.shape,
            center=self.center,
            affine=self.affine,
            wavelength=self.wavelength,
        )
        return ai


@deprecated(
    since="0.18.0", removal="1.0.0", alternative="pyxem.signals.Diffraction2D.calibrate"
)
def find_diffraction_calibration(
    patterns,
    calibration_guess,
    library_phases,
    lib_gen,
    size,
    max_excitation_error=0.01,
    **kwargs,
):
    """Finds the diffraction calibration for a pattern or set of patterns by maximizing correlation scores.

    Parameters
    ----------
    patterns : hyperspy.signals.Signal2D
        Diffration patterns to be iteratively matched to find maximum correlation scores.
    calibration_guess : float
        Inital value for the diffraction calibration in inverse Angstoms per pixel
    library_phases : diffsims.libraries.StructureLibrary
        Dictionary of structures and associated orientations for which
        electron diffraction is to be simulated.
    lib_gen : diffsims.generators.DiffractionLibraryGenerator
        Computes a library of electron diffraction patterns for specified atomic
        structures and orientations.  Used to create the DiffractionLibrary.
    size : integer
        How many different steps to test for the first two iterations. These steps have a size of 1% of the calibration guess.
    max_excitation_error : float
        Maximum exacitation error.  Default is 0.01.
    kwargs
        Keyword arguments passed to :meth:`index_dataset_with_template_rotation`.

    Returns
    -------
    mean_cal : float
        Mean of calibrations found for each pattern.
    full_corrlines : numpy.ndarray
        Gives the explicit correlation vs calibration values. Shape:(size*2 + 20, 2 , number of patterns)
    found_cals : numpy.ndarray
        List of optimal calibration values for each pattern. Shape:(number of patterns)
    """

    images = patterns

    num_patterns = images.data.shape[0]
    found_cals = np.full((num_patterns,), calibration_guess)
    full_corrlines = np.zeros((0, 2, num_patterns))

    stepsize = 0.01 * calibration_guess
    # first set of checks
    corrlines = _calibration_iteration(
        images,
        calibration_guess,
        library_phases,
        lib_gen,
        stepsize,
        size,
        num_patterns,
        max_excitation_error,
        **kwargs,
    )
    full_corrlines = np.append(full_corrlines, corrlines, axis=0)

    # refined calibration checks
    calibration_guess = full_corrlines[
        full_corrlines[:, 1, :].argmax(axis=0), 0, 0
    ].mean()
    corrlines = _calibration_iteration(
        images,
        calibration_guess,
        library_phases,
        lib_gen,
        stepsize,
        size,
        num_patterns,
        max_excitation_error,
        **kwargs,
    )
    full_corrlines = np.append(full_corrlines, corrlines, axis=0)

    # more refined calibration checks with smaller step
    stepsize = 0.001 * calibration_guess
    size = 20
    calibration_guess = full_corrlines[
        full_corrlines[:, 1, :].argmax(axis=0), 0, 0
    ].mean()

    corrlines = _calibration_iteration(
        images,
        calibration_guess,
        library_phases,
        lib_gen,
        stepsize,
        size,
        num_patterns,
        max_excitation_error,
        **kwargs,
    )
    full_corrlines = np.append(full_corrlines, corrlines, axis=0)
    found_cals = full_corrlines[full_corrlines[:, 1, :].argmax(axis=0), 0, 0]

    mean_cal = found_cals.mean()
    return mean_cal, full_corrlines, found_cals


def _calibration_iteration(
    images,
    calibration_guess,
    library_phases,
    lib_gen,
    stepsize,
    size,
    num_patterns,
    max_excitation_error,
    **kwargs,
):
    """For use in find_diffraction_calibration.  Controls the iteration of _create_check_diflib over a set of steps.

    Parameters
    ----------
    images : hyperspy.signals.Signal2D
        Diffration patterns to be iteratively matched to find maximum correlation scores.
    calibration_guess : float
        Inital value for the diffraction calibration in inverse Angstoms per pixel
    library_phases : diffsims.libraries.StructureLibrary
        Dictionary of structures and associated orientations for which
        electron diffraction is to be simulated.
    lib_gen : diffsims.generators.DiffractionLibraryGenerator
        Computes a library of electron diffraction patterns for specified atomic
        structures and orientations.  Used to create the DiffractionLibrary.
    stepsize : float
        Stepsize of iteration.
    size : integer
        How many different steps to test.
    num_patterns : integer
        Number of patterns.
    max_excitation_error : float
        Maximum exacitation error.  Default is 0.01.
    kwargs
        Keyword arguments passed to :meth:`index_dataset_with_template_rotation`.

    Returns
    -------
    corrlines : numpy.ndarray
    """
    corrlines = np.zeros((0, 2, num_patterns))
    temp_line = np.zeros((1, 2, num_patterns))
    cal_guess_greater = calibration_guess
    cal_guess_lower = calibration_guess
    for i in range(size // 2):
        temp_line[0, 0, :] = cal_guess_lower
        temp_line[0, 1, :] = _create_check_diflib(
            images,
            cal_guess_lower,
            library_phases,
            lib_gen,
            max_excitation_error,
            **kwargs,
        )
        corrlines = np.append(corrlines, temp_line, axis=0)

        temp_line[0, 0, :] = cal_guess_greater
        temp_line[0, 1, :] = _create_check_diflib(
            images,
            cal_guess_greater,
            library_phases,
            lib_gen,
            max_excitation_error,
            **kwargs,
        )
        corrlines = np.append(corrlines, temp_line, axis=0)

        cal_guess_lower = cal_guess_lower - stepsize
        cal_guess_greater = cal_guess_greater + stepsize

    return corrlines


def _create_check_diflib(
    images, calibration_guess, library_phases, lib_gen, max_excitation_error, **kwargs
):
    """For use in find_diffraction_calibration via _calibration_iteration.  Creates a new DiffractionLibrary from the inputs and then matches it the images.

    Parameters
    ----------
    images : hyperspy.signals.Signal2D
        Diffration patterns to be iteratively matched to find maximum correlation scores.
    calibration_guess : float
        Inital value for the diffraction calibration in inverse Angstoms per pixel
    library_phases : diffsims.libraries.StructureLibrary
        Dictionary of structures and associated orientations for which
        electron diffraction is to be simulated.
    lib_gen : diffsims.generators.DiffractionLibraryGenerator
        Computes a library of electron diffraction patterns for specified atomic
        structures and orientations.  Used to create the DiffractionLibrary.
    max_excitation_error : float
        Maximum exacitation error.  Default is 0.01.
    kwargs
        Keyword arguments passed to :meth:`index_dataset_with_template_rotation`.

    Returns
    -------
    correlations : numpy.ndarray
    """

    half_shape = (images.data.shape[-2] // 2, images.data.shape[-1] // 2)
    reciprocal_r = np.sqrt(half_shape[0] ** 2 + half_shape[1] ** 2) * calibration_guess
    diff_lib = lib_gen.get_diffraction_library(
        library_phases,
        calibration=calibration_guess,
        reciprocal_radius=reciprocal_r,
        half_shape=half_shape,
        with_direct_beam=False,
        max_excitation_error=max_excitation_error,
    )

    result, phasedict = index_dataset_with_template_rotation(images, diff_lib, **kwargs)
    correlations = result["correlation"][:, :, 0].flatten()
    return correlations
