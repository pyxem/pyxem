# -*- coding: utf-8 -*-
# Copyright 2016-2020 The pyXem developers
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

"""
Signal class for two-dimensional diffraction data in Cartesian coordinates.
"""

import numpy as np
from warnings import warn

from hyperspy.signals import Signal2D, BaseSignal
from hyperspy._signals.lazy import LazySignal

from pyxem.signals.diffraction1d import Diffraction1D
from pyxem.signals.electron_diffraction1d import ElectronDiffraction1D
from pyxem.signals.polar_diffraction2d import PolarDiffraction2D
from pyxem.signals import transfer_navigation_axes, select_method_from_method_dict
from pyxem.signals.common_diffraction import CommonDiffraction
from pyxem.utils.pyfai_utils import (
    get_azimuthal_integrator,
    _get_radial_extent,
    _get_setup,
)

from pyxem.utils.expt_utils import (
    azimuthal_integrate1d_slow,
    azimuthal_integrate1d_fast,
    azimuthal_integrate2d_slow,
    azimuthal_integrate2d_fast,
    gain_normalise,
    remove_dead,
    regional_filter,
    subtract_background_dog,
    subtract_background_median,
    subtract_reference,
    circular_mask,
    find_beam_offset_cross_correlation,
    peaks_as_gvectors,
    convert_affine_to_transform,
    apply_transformation,
    find_beam_center_blur,
    find_beam_center_interpolate,
)

from pyxem.utils.peakfinders2D import (
    find_peaks_zaefferer,
    find_peaks_stat,
    find_peaks_dog,
    find_peaks_log,
    find_peaks_xc,
)


from pyxem.utils import peakfinder2D_gui

from skimage import filters
from skimage.morphology import square


class Diffraction2D(Signal2D, CommonDiffraction):
    _signal_type = "diffraction"

    def get_direct_beam_mask(self, radius):
        """Generate a signal mask for the direct beam.

        Parameters
        ----------
        radius : float
            Radius for the circular mask in pixel units.

        Return
        ------
        signal-mask : ndarray
            The mask of the direct beam
        """
        shape = self.axes_manager.signal_shape
        center = (shape[1] - 1) / 2, (shape[0] - 1) / 2

        signal_mask = Signal2D(circular_mask(shape=shape, radius=radius, center=center))

        return signal_mask

    def apply_affine_transformation(
        self, D, order=3, keep_dtype=False, inplace=True, *args, **kwargs
    ):
        """Correct geometric distortion by applying an affine transformation.

        Parameters
        ----------
        D : array or Signal2D of arrays
            3x3 np.array (or Signal2D thereof) specifying the affine transform
            to be applied.
        order : 1,2,3,4 or 5
            The order of interpolation on the transform. Default is 3.
        keep_dtype : bool
            If True dtype of returned ElectronDiffraction2D Signal is that of
            the input, if False, casting to higher precision may occur.
        inplace : bool
            If True (default), this signal is overwritten. Otherwise, returns a
            new signal.
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to be passed to map().

        Returns
        -------
            ElectronDiffraction2D Signal containing the affine Transformed
            diffraction patterns.

        """

        shape = self.axes_manager.signal_shape
        if isinstance(D, np.ndarray):
            transformation = convert_affine_to_transform(D, shape)
        else:
            transformation = D.map(
                convert_affine_to_transform, shape=shape, inplace=False
            )

        return self.map(
            apply_transformation,
            transformation=transformation,
            order=order,
            keep_dtype=keep_dtype,
            inplace=inplace,
            *args,
            **kwargs
        )

    def apply_gain_normalisation(
        self, dark_reference, bright_reference, inplace=True, *args, **kwargs
    ):
        """Apply gain normalization to experimentally acquired electron
        diffraction patterns.

        Parameters
        ----------
        dark_reference : ElectronDiffraction2D
            Dark reference image.
        bright_reference : DiffractionSignal
            Bright reference image.
        inplace : bool
            If True (default), this signal is overwritten. Otherwise, returns a
            new signal.
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to be passed to map().

        """
        return self.map(
            gain_normalise,
            dref=dark_reference,
            bref=bright_reference,
            inplace=inplace,
            *args,
            **kwargs
        )

    def remove_deadpixels(
        self,
        deadpixels,
        deadvalue="average",
        inplace=True,
        progress_bar=True,
        *args,
        **kwargs
    ):
        """Remove deadpixels from experimentally acquired diffraction patterns.

        Parameters
        ----------
        deadpixels : list
            List of deadpixels to be removed.
        deadvalue : str
            Specify how deadpixels should be treated. 'average' sets the dead
            pixel value to the average of adjacent pixels. 'nan' sets the dead
            pixel to nan
        inplace : bool
            If True (default), this signal is overwritten. Otherwise, returns a
            new signal.
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to be passed to map().

        """
        return self.map(
            remove_dead,
            deadpixels=deadpixels,
            deadvalue=deadvalue,
            inplace=inplace,
            show_progressbar=progress_bar,
            *args,
            **kwargs
        )

    def get_azimuthal_integral1d(
        self,
        npt_rad,
        center=None,
        affine=None,
        mask=None,
        radial_range=None,
        azimuth_range=None,
        wavelength=None,
        unit="pyxem",
        inplace=False,
        method="splitpixel",
        map_kwargs={},
        detector=None,
        detector_dist=None,
        correctSolidAngle=True,
        ai_kwargs={},
        integrate2d_kwargs={},
    ):
        """Creates a polar reprojection using pyFAI's azimuthal integrate 2d.

        This function is designed to be fairly flexible to account for 2 different cases:

        1 - If the unit is "pyxem" then it lets pyXEM take the lead. If wavelength is none in that case
        it doesn't account for the Ewald sphere.

        2 - If unit is any of the options from pyFAI then detector cannot be None and the handling of
        units is passed to pyxem and those units are used.

        Parameters
        ---------------
        npt_rad: int
            The number of radial points to calculate
        center: None or (x,y) or BaseSignal
            The center of the pattern in pixels to preform the integration around
        affine: 3x3 array or BaseSignal
            An affine transformation to apply during the transformation
             (creates a spline map that is used by pyFAI)
        mask:  boolean array or BaseSignal
            A boolean mask to apply to the data to exclude some points.
            If mask is a baseSignal then it is itereated over as well.
        radial_range: None or (float, float)
            The radial range over which to perform the integration. Default is
            the full frame
        azim_range:None or (float, float)
            The azimuthal range over which to perform the integration. Default is
            from -pi to pi
        wavelength: None or float
            The wavelength of for the microscope. Has to be in the same units as the pyxem units if you want
            it to properly work.
        unit: str
            The unit can be "pyxem" to use the pyxem units and “q_nm^-1”, “q_A^-1”, “2th_deg”, “2th_rad”, “r_mm”
            if pyFAI is used for unit handling
        inplace: bool
            If the signal is overwritten or copied to a new signal
        detector: pyFai.detector.Detector
            The detector set up to be used by the integrator
        detector_dist: float
            distance sample - detector plan (orthogonal distance, not along the beam), in meter.
        map_kwargs: dict
            Any other keyword arguments for hyperspys map function
        integrate2d_kwargs:dict
            Any keyword arguements for PyFAI's integrate2d function

        Returns
        ----------
        polar: PolarDiffraction2D
            A polar diffraction signal

        Examples
        ----------
        Basic case using "2th_deg" units (no wavelength needed)

        >>> ds.unit = "2th_deg"
        >>> ds.get_azimuthal_integral1d(npt_rad=100)

        Basic case using a curved Ewald Sphere approximation and pyXEM units
        (wavelength needed)

        >>> ds.unit = "k_nm^-1" # setting units
        >>> ds.get_azimuthal_integral1d(npt_rad=100, wavelength=2.5e-12)

        Using pyFAI to define a detector case using a curved Ewald Sphere approximation and pyXEM units

        >>> from pyFAI.detectors import Detector
        >>> det = Detector(pixel1=1e-4, pixel2=1e-4)
        >>> ds.get_azimuthal_integral1d(npt_rad=100, detector_dist=.2, detector= det, wavelength=2.508e-12)
        """
        pyxem_units = False
        sig_shape = self.axes_manager.signal_shape
        signal_type = self._signal_type

        if unit == "pyxem":  # Case 1
            pyxem_units = True
            pixel_scale = [
                self.axes_manager.signal_axes[0].scale,
                self.axes_manager.signal_axes[1].scale,
            ]
            if wavelength is None and self.unit not in ["2th_deg", "2th_rad"]:
                print(
                    'if the unit is not "2th_deg", "2th_rad"'
                    "then a wavelength must be given. "
                )
                return None
            setup = _get_setup(wavelength, self.unit, pixel_scale, radial_range)
            detector, detector_dist, radial_range, unit, scale_factor = setup
        use_iterate = any(
            [
                isinstance(mask, BaseSignal),
                isinstance(affine, BaseSignal),
                isinstance(center, BaseSignal),
            ]
        )
        if use_iterate:
            if radial_range is None:  # need consistent range
                if isinstance(center, BaseSignal):
                    ind = (0,) * len(self.axes_manager.navigation_shape)
                    cen = center.inav[ind].data
                else:
                    cen = center
                ai = get_azimuthal_integrator(
                    detector=detector,
                    detector_distance=detector_dist,
                    shape=sig_shape,
                    center=cen,
                    wavelength=wavelength,
                )  # take 1st center
                radial_range = _get_radial_extent(ai=ai, shape=sig_shape, unit=unit)
                radial_range[0] = 0
            integration = self.map(
                azimuthal_integrate1d_slow,
                detector=detector,
                center=center,
                mask=mask,
                affine=affine,
                detector_distance=detector_dist,
                npt_rad=npt_rad,
                wavelength=wavelength,
                radial_range=radial_range,
                azimuth_range=azimuth_range,
                inplace=inplace,
                unit=unit,
                method=method,
                correctSolidAngle=correctSolidAngle,
                **integrate2d_kwargs,
                **map_kwargs
            )  # Uses slow methodology

        else:  # much simpler and no changing integrator without using map iterate
            ai = get_azimuthal_integrator(
                detector=detector,
                detector_distance=detector_dist,
                shape=sig_shape,
                center=center,
                affine=affine,
                mask=mask,
                wavelength=wavelength,
                **ai_kwargs
            )
            if radial_range is None:
                radial_range = _get_radial_extent(ai=ai, shape=sig_shape, unit=unit)
                radial_range[0] = 0
            print(radial_range)

            integration = self.map(
                azimuthal_integrate1d_fast,
                azimuthal_integrator=ai,
                npt_rad=npt_rad,
                azimuth_range=azimuth_range,
                radial_range=radial_range,
                method=method,
                inplace=inplace,
                unit=unit,
                correctSolidAngle=correctSolidAngle,
                **integrate2d_kwargs,
                **map_kwargs
            )

        # Dealing with axis changes
        if inplace:
            k_axis = self.axes_manager.signal_axes[0]
            self.set_signal_type(signal_type)
        else:
            integration.set_signal_type(signal_type)
            transfer_navigation_axes(integration, self)
            k_axis = integration.axes_manager.signal_axes[0]
        k_axis.name = "Radius"
        if pyxem_units:
            k_axis.scale = (radial_range[1] - radial_range[0]) / npt_rad / scale_factor
            k_axis.offset = radial_range[0] / scale_factor
        else:
            k_axis.scale = (radial_range[1] - radial_range[0]) / npt_rad
            k_axis.units = unit
            k_axis.offset = radial_range[0]

        return integration

    def get_azimuthal_integral2d(
        self,
        npt_rad,
        npt_azim=360,
        center=None,
        affine=None,
        mask=None,
        radial_range=None,
        azimuth_range=None,
        wavelength=None,
        unit="pyxem",
        inplace=False,
        method="splitpixel",
        map_kwargs={},
        detector=None,
        detector_dist=None,
        correctSolidAngle=True,
        ai_kwargs={},
        integrate2d_kwargs={},
    ):
        """Creates a polar reprojection using pyFAI's azimuthal integrate 2d.

        This function is designed to be fairly flexible to account for 2 different cases:

        1 - If the unit is "pyxem" then it lets pyXEM take the lead. If wavelength is none in that case
        it doesn't account for the Ewald sphere.

        2 - If unit is any of the options from pyFAI then detector cannot be None and the handling of
        units is passed to pyxem and those units are used.

        Parameters
        ---------------
        npt_rad: int
            The number of radial points to calculate
        npt_azim: int
            The number of azimuthal points to calculate
        center: None or (x,y) or BaseSignal
            The center of the pattern in pixels to preform the integration around
        affine: 3x3 array or BaseSignal
            An affine transformation to apply during the transformation
             (creates a spline map that is used by pyFAI)
        mask:  boolean array or BaseSignal
            A boolean mask to apply to the data to exclude some points.
            If mask is a baseSignal then it is itereated over as well.
        radial_range: None or (float, float)
            The radial range over which to perform the integration. Default is
            the full frame
        azim_range:None or (float, float)
            The azimuthal range over which to perform the integration. Default is
            from -pi to pi
        wavelength: None or float
            The wavelength of for the microscope. Has to be in the same units as the pyxem units if you want
            it to properly work.
        unit: str
            The unit can be "pyxem" to use the pyxem units and “q_nm^-1”, “q_A^-1”, “2th_deg”, “2th_rad”, “r_mm”
            if pyFAI is used for unit handling
        inplace: bool
            If the signal is overwritten or copied to a new signal
        detector: pyFai.detector.Detector
            The detector set up to be used by the integrator
        detector_dist: float
            distance sample - detector plan (orthogonal distance, not along the beam), in meter.
        map_kwargs: dict
            Any other keyword arguments for hyperspys map function
        integrate2d_kwargs:dict
            Any keyword arguements for PyFAI's integrate2d function

        Returns
        ----------
        polar: PolarDiffraction2D
            A polar diffraction signal

        Examples
        ----------
        Basic case using "2th_deg" units (no wavelength needed)

        >>> ds.unit = "2th_deg"
        >>> ds.get_azimuthal_integral2d(npt_rad=100)

        Basic case using a curved Ewald Sphere approximation and pyXEM units
        (wavelength needed)

        >>> ds.unit = "k_nm^-1" # setting units
        >>> ds.get_azimuthal_integral1d(npt_rad=100, wavelength=2.5e-12)

        Using pyFAI to define a detector case using a curved Ewald Sphere approximation and pyXEM units

        >>> from pyFAI.detectors import Detector
        >>> det = Detector(pixel1=1e-4, pixel2=1e-4)
        >>> ds.get_azimuthal_integral1d(npt_rad=100, detector_dist=.2, detector= det, wavelength=2.508e-12)
        """
        pyxem_units = False
        sig_shape = self.axes_manager.signal_shape

        if unit == "pyxem":
            pyxem_units = True
            pixel_scale = [
                self.axes_manager.signal_axes[0].scale,
                self.axes_manager.signal_axes[1].scale,
            ]
            if wavelength is None and self.unit not in ["2th_deg", "2th_rad"]:
                print(
                    'if the unit is not "2th_deg", "2th_rad"'
                    "then a wavelength must be given. "
                )
                return
            setup = _get_setup(wavelength, self.unit, pixel_scale, radial_range)
            detector, detector_dist, radial_range, unit, scale_factor = setup
        use_iterate = any(
            [
                isinstance(mask, BaseSignal),
                isinstance(affine, BaseSignal),
                isinstance(center, BaseSignal),
            ]
        )
        if use_iterate:
            if radial_range is None:  # need consistent range
                if isinstance(center, BaseSignal):
                    ind = (0,) * len(self.axes_manager.navigation_shape)
                    cen = center.inav[ind].data
                else:
                    cen = center
                ai = get_azimuthal_integrator(
                    detector=detector,
                    detector_distance=detector_dist,
                    shape=sig_shape,
                    center=cen,
                    wavelength=wavelength,
                )  # take 1st center
                radial_range = _get_radial_extent(ai=ai, shape=sig_shape, unit=unit)
                radial_range[0] = 0
            integration = self.map(
                azimuthal_integrate2d_slow,
                npt_azim=npt_azim,
                detector=detector,
                center=center,
                mask=mask,
                affine=affine,
                detector_distance=detector_dist,
                npt_rad=npt_rad,
                wavelength=wavelength,
                radial_range=radial_range,
                azimuth_range=azimuth_range,
                inplace=inplace,
                unit=unit,
                method=method,
                correctSolidAngle=correctSolidAngle,
                **integrate2d_kwargs,
                **map_kwargs
            )  # Uses slow methodology

        else:  # much simpler and no changing integrator without using map iterate
            ai = get_azimuthal_integrator(
                detector=detector,
                detector_distance=detector_dist,
                shape=sig_shape,
                center=center,
                affine=affine,
                mask=mask,
                wavelength=wavelength,
                **ai_kwargs
            )
            if radial_range is None:
                radial_range = _get_radial_extent(ai=ai, shape=sig_shape, unit=unit)
                radial_range[0] = 0

            integration = self.map(
                azimuthal_integrate2d_fast,
                azimuthal_integrator=ai,
                npt_rad=npt_rad,
                npt_azim=npt_azim,
                azimuth_range=azimuth_range,
                radial_range=radial_range,
                method=method,
                inplace=inplace,
                unit=unit,
                correctSolidAngle=correctSolidAngle,
                **integrate2d_kwargs,
                **map_kwargs
            )

        # Dealing with axis changes
        if inplace:
            t_axis = self.axes_manager.signal_axes[0]
            k_axis = self.axes_manager.signal_axes[1]
            self.set_signal_type("polar_diffraction")
        else:
            transfer_navigation_axes(integration, self)
            integration.set_signal_type("polar_diffraction")
            t_axis = integration.axes_manager.signal_axes[0]
            k_axis = integration.axes_manager.signal_axes[1]
        t_axis.name = "Radians"
        if azimuth_range is None:
            t_axis.scale = np.pi * 2 / npt_azim
            t_axis.offset = -np.pi
        else:
            t_axis.scale = (azimuth_range[1] - azimuth_range[0]) / npt_rad
            t_axis.offset = azimuth_range[0]
        k_axis.name = "Radius"
        if pyxem_units:
            k_axis.scale = (radial_range[1] - radial_range[0]) / npt_rad / scale_factor
            k_axis.offset = radial_range[0] / scale_factor
        else:
            k_axis.scale = (radial_range[1] - radial_range[0]) / npt_rad
            k_axis.units = unit
            k_axis.offset = radial_range[0]

        return integration

    def get_direct_beam_position(self, method, **kwargs):
        """Estimate the direct beam position in each experimentally acquired
        electron diffraction pattern.

        Parameters
        ----------
        method : str,
            Must be one of "cross_correlate", "blur", "interpolate"
        **kwargs:
            Keyword arguments to be passed to map().

        Returns
        -------
        shifts : ndarray
            Array containing the shifts for each SED pattern.

        """
        signal_shape = self.axes_manager.signal_shape
        origin_coordinates = np.array(signal_shape) / 2

        method_dict = {
            "cross_correlate": find_beam_offset_cross_correlation,
            "blur": find_beam_center_blur,
            "interpolate": find_beam_center_interpolate,
        }

        method_function = select_method_from_method_dict(method, method_dict, **kwargs)

        if method == "cross_correlate":
            shifts = self.map(method_function, inplace=False, **kwargs)
        elif method == "blur" or method == "interpolate":
            centers = self.map(method_function, inplace=False, **kwargs)
            shifts = origin_coordinates - centers

        return shifts

    def center_direct_beam(
        self,
        method,
        half_square_width=None,
        return_shifts=False,
        align_kwargs={},
        *args,
        **kwargs
    ):
        """Estimate the direct beam position in each experimentally acquired
        electron diffraction pattern and translate it to the center of the
        image square.

        Parameters
        ----------
        method : str {'cross_correlate', 'blur', 'interpolate'}
            Method used to estimate the direct beam position
        half_square_width : int
            Half the side length of square that captures the direct beam in all
            scans. Means that the centering algorithm is stable against
            diffracted spots brighter than the direct beam.
        return_shifts : bool, default False
            If True, the values of applied shifts are returned
        align_kwargs : dict
            To be passed to .align2D() function
        **kwargs:
            To be passed to method function

        Returns
        -------
        centered : ElectronDiffraction2D
            The centered diffraction data.

        """
        nav_size = self.axes_manager.navigation_size
        signal_shape = self.axes_manager.signal_shape
        origin_coordinates = np.array(signal_shape) / 2

        if half_square_width is not None:
            min_index = np.int(origin_coordinates[0] - half_square_width)
            # fails if non-square dp
            max_index = np.int(origin_coordinates[0] + half_square_width)
            cropped = self.isig[min_index:max_index, min_index:max_index]
            shifts = cropped.get_direct_beam_position(method=method, **kwargs)
        else:
            shifts = self.get_direct_beam_position(method=method, **kwargs)

        shifts = -1 * shifts.data
        shifts = shifts.reshape(nav_size, 2)

        # Preserve existing behaviour by overriding
        # crop & fill_value
        align_kwargs.pop("crop", None)
        align_kwargs.pop("fill_value", None)

        self.align2D(shifts=shifts, crop=False, fill_value=0, **align_kwargs)

        if return_shifts:
            return shifts

    def remove_background(self, method, **kwargs):
        """Perform background subtraction via multiple methods.

        Parameters
        ----------
        method : str
            Specifies the method, from:
            {'h-dome','gaussian_difference','median','reference_pattern'}
        **kwargs:
            Keyword arguments to be passed to map(), including method specific ones,
            running a method with no kwargs will return help

        Returns
        -------
        bg_subtracted : :obj:`ElectronDiffraction2D`
            A copy of the data with the background subtracted. Be aware that
            this function will only return inplace.
        """
        method_dict = {
            "h-dome": regional_filter,
            "gaussian_difference": subtract_background_dog,
            "median": subtract_background_median,
            "reference_pattern": subtract_reference,
        }

        method_function = select_method_from_method_dict(method, method_dict, **kwargs)

        if method != "h-dome":
            bg_subtracted = self.map(method_function, inplace=False, **kwargs)
        elif method == "h-dome":
            scale = self.data.max()
            self.data = self.data / scale
            bg_subtracted = self.map(method_function, inplace=False, **kwargs)
            bg_subtracted.map(filters.rank.mean, selem=square(3))
            bg_subtracted.data = bg_subtracted.data / bg_subtracted.data.max()

        return bg_subtracted

    def find_peaks(self, method, *args, **kwargs):
        """Find the position of diffraction peaks.

        Function to locate the positive peaks in an image using various, user
        specified, methods. Returns a structured array containing the peak
        positions.

        Parameters
        ---------
        method : str
            Select peak finding algorithm to implement. Available methods are
            {'zaefferer', 'stat', 'laplacian_of_gaussians',
            'difference_of_gaussians', 'xc'}
        *args : arguments
            Arguments to be passed to the peak finders.
        **kwargs : arguments
            Keyword arguments to be passed to the peak finders.

        Returns
        -------
        peaks : DiffractionVectors
            A DiffractionVectors object with navigation dimensions identical to
            the original ElectronDiffraction2D object. Each signal is a BaseSignal
            object contiaining the diffraction vectors found at each navigation
            position, in calibrated units.

        Notes
        -----
        Peak finding methods are detailed as:

            * 'zaefferer' - based on gradient thresholding and refinement
              by local region of interest optimisation
            * 'stat' - statistical approach requiring no free params.
            * 'laplacian_of_gaussians' - a blob finder implemented in
              `scikit-image` which uses the laplacian of Gaussian matrices
              approach.
            * 'difference_of_gaussians' - a blob finder implemented in
              `scikit-image` which uses the difference of Gaussian matrices
              approach.
            * 'xc' - A cross correlation peakfinder

        """
        method_dict = {
            "zaefferer": find_peaks_zaefferer,
            "stat": find_peaks_stat,
            "laplacian_of_gaussians": find_peaks_log,
            "difference_of_gaussians": find_peaks_dog,
            "xc": find_peaks_xc,
        }
        if method in method_dict:
            method = method_dict[method]
        else:
            raise NotImplementedError(
                "The method `{}` is not implemented. "
                "See documentation for available "
                "implementations.".format(method)
            )

        peaks = self.map(method, *args, **kwargs, inplace=False, ragged=True)
        peaks.map(
            peaks_as_gvectors,
            center=np.array(self.axes_manager.signal_shape) / 2 - 0.5,
            calibration=self.axes_manager.signal_axes[0].scale,
        )
        peaks.set_signal_type("diffraction_vectors")

        # Set DiffractionVectors attributes
        peaks.pixel_calibration = self.axes_manager.signal_axes[0].scale
        peaks.detector_shape = self.axes_manager.signal_shape

        # Set calibration to same as signal
        x = peaks.axes_manager.navigation_axes[0]
        y = peaks.axes_manager.navigation_axes[1]

        x.name = "x"
        x.scale = self.axes_manager.navigation_axes[0].scale
        x.units = "nm"

        y.name = "y"
        y.scale = self.axes_manager.navigation_axes[1].scale
        y.units = "nm"

        return peaks

    def find_peaks_interactive(self, disc_image=None, imshow_kwargs={}):
        """Find peaks using an interactive tool.

        Parameters
        ----------
        disc_image : numpy.array
            See .utils.peakfinders2D.peak_finder_xc for details. If not
            given a warning will be raised.
        imshow_kwargs : arguments
            kwargs to be passed to internal imshow statements

        Notes
        -----
        Requires `ipywidgets` and `traitlets` to be installed.

        """
        if disc_image is None:
            warn(
                "You have not specified a disc image, as such you will not "
                "be able to use the xc method in this session"
            )

        peakfinder = peakfinder2D_gui.PeakFinderUIIPYW(
            disc_image=disc_image, imshow_kwargs=imshow_kwargs
        )
        peakfinder.interactive(self)


class LazyDiffraction2D(LazySignal, Diffraction2D):

    pass
