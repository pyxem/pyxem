# -*- coding: utf-8 -*-
# Copyright 2017-2020 The pyXem developers
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
Signal base class for two-dimensional diffraction data.
"""

import numpy as np
from warnings import warn

from hyperspy.api import interactive
from hyperspy.signals import Signal1D, Signal2D, BaseSignal
from hyperspy._signals.lazy import LazySignal

from pyxem.signals.diffraction1d import Diffraction1D
from pyxem.signals.electron_diffraction1d import ElectronDiffraction1D
from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.signals import push_metadata_through, transfer_navigation_axes, \
    select_method_from_method_dict

from pyxem.utils.expt_utils import _index_coords, _cart2polar, _polar2cart, \
    radial_average, azimuthal_integrate, azimuthal_integrate_fast, \
    gain_normalise, remove_dead, regional_filter, subtract_background_dog, \
    subtract_background_median, subtract_reference, circular_mask, \
    find_beam_offset_cross_correlation, peaks_as_gvectors, \
    convert_affine_to_transform, apply_transformation, find_beam_center_blur, \
    find_beam_center_interpolate

from pyxem.utils.peakfinders2D import find_peaks_zaefferer, find_peaks_stat, \
    find_peaks_dog, find_peaks_log, find_peaks_xc

from pyxem.utils import peakfinder2D_gui

from skimage import filters
from skimage import transform as tf
from skimage.morphology import square

from pyFAI.azimuthalIntegrator import AzimuthalIntegrator


class Diffraction2D(Signal2D):
    _signal_type = "diffraction2d"

    def __init__(self, *args, **kwargs):
        """
        Create an Diffraction2D object from a hs.Signal2D or np.array.

        Parameters
        ----------
        *args :
            Passed to the __init__ of Signal2D. The first arg should be
            either a numpy.ndarray or a Signal2D
        **kwargs :
            Passed to the __init__ of Signal2D
        """
        self, args, kwargs = push_metadata_through(self, *args, **kwargs)
        super().__init__(*args, **kwargs)

        self.decomposition.__func__.__doc__ = BaseSignal.decomposition.__doc__

    def plot_interactive_virtual_image(self, roi, **kwargs):
        """Plots an interactive virtual image formed with a specified and
        adjustable roi.

        Parameters
        ----------
        roi : :obj:`hyperspy.roi.BaseInteractiveROI`
            Any interactive ROI detailed in HyperSpy.
        **kwargs:
            Keyword arguments to be passed to `Diffraction2D.plot`

        Examples
        --------
        .. code-block:: python

            import hyperspy.api as hs
            roi = hs.roi.CircleROI(0, 0, 0.2)
            data.plot_interactive_virtual_image(roi)

        """
        self.plot(**kwargs)
        roi.add_widget(self, axes=self.axes_manager.signal_axes)
        # Add the ROI to the appropriate signal axes.
        dark_field = roi.interactive(self, navigation_signal='same')
        dark_field_placeholder = \
            BaseSignal(np.zeros(self.axes_manager.navigation_shape[::-1]))
        # Create an output signal for the virtual dark-field calculation.
        dark_field_sum = interactive(
            # Create an interactive signal
            dark_field.sum,
            # Formed from the sum of the pixels in the dark-field signal
            event=dark_field.axes_manager.events.any_axis_changed,
            # That updates whenever the widget is moved
            axis=dark_field.axes_manager.signal_axes,
            out=dark_field_placeholder,
            # And outputs into the prepared placeholder.
        )
        dark_field_sum.axes_manager.update_axes_attributes_from(
            self.axes_manager.navigation_axes,
            ['scale', 'offset', 'units', 'name'])
        dark_field_sum.metadata.General.title = "Virtual Dark Field"
        # Set the parameters
        dark_field_sum.plot()  # Plot the result

    def get_virtual_image(self, roi):
        """Obtains a virtual image associated with a specified ROI.

        Parameters
        ----------
        roi: :obj:`hyperspy.roi.BaseInteractiveROI`
            Any interactive ROI detailed in HyperSpy.

        Returns
        -------
        dark_field_sum : :obj:`hyperspy.signals.BaseSignal`
            The virtual image signal associated with the specified roi.

        Examples
        --------
        .. code-block:: python

            import hyperspy.api as hs
            roi = hs.roi.CircleROI(0, 0, 0.2)
            data.get_virtual_image(roi)

        """
        dark_field = roi(self, axes=self.axes_manager.signal_axes)
        dark_field_sum = dark_field.sum(
            axis=dark_field.axes_manager.signal_axes
        )
        dark_field_sum.metadata.General.title = "Virtual Dark Field"
        vdfim = dark_field_sum.as_signal2D((0, 1))

        return vdfim

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

        signal_mask = Signal2D(circular_mask(shape=shape,
                                             radius=radius,
                                             center=center))

        return signal_mask

    def apply_affine_transformation(self,
                                    D,
                                    order=3,
                                    keep_dtype=False,
                                    inplace=True,
                                    *args, **kwargs):
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
            transformation = D.map(convert_affine_to_transform, shape=shape, inplace=False)

        return self.map(apply_transformation,
                        transformation=transformation,
                        order=order,
                        keep_dtype=keep_dtype,
                        inplace=inplace,
                        *args, **kwargs)

    def apply_gain_normalisation(self,
                                 dark_reference,
                                 bright_reference,
                                 inplace=True,
                                 *args, **kwargs):
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
        return self.map(gain_normalise,
                        dref=dark_reference,
                        bref=bright_reference,
                        inplace=inplace,
                        *args, **kwargs)

    def remove_deadpixels(self,
                          deadpixels,
                          deadvalue='average',
                          inplace=True,
                          progress_bar=True,
                          *args, **kwargs):
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
        return self.map(remove_dead,
                        deadpixels=deadpixels,
                        deadvalue=deadvalue,
                        inplace=inplace,
                        show_progressbar=progress_bar,
                        *args, **kwargs)

    def get_azimuthal_integral(self, origin, detector, detector_distance,
                               wavelength, size_1d, unit='k_A^-1',
                               inplace=False,
                               kwargs_for_map={}, kwargs_for_integrator={},
                               kwargs_for_integrate1d={}):
        """
        Returns the azimuthal integral of the diffraction pattern as a
        Diffraction1D signal.

        Parameters
        ----------
        origin : np.array_like
            This parameter should either be a list or numpy.array with two
            coordinates ([x_origin,y_origin]), or an array of the same shape as
            the navigation axes, with an origin (with the shape
            [x_origin,y_origin]) at each navigation location.
        detector : pyFAI.detectors.Detector object
            A pyFAI detector used for the AzimuthalIntegrator.
        detector_distance : float
            Detector distance in meters passed to pyFAI AzimuthalIntegrator.
        wavelength : float
            The electron wavelength in meters. Used by pyFAI AzimuthalIntegrator
        size_1d : int
            The size of the returned 1D signal. (i.e. number of pixels in the
            1D azimuthal integral.)
        unit : str
            The unit for for PyFAI integrate1d. The default "k_A^-1" gives k in
            inverse Angstroms and is not natively in PyFAI. The other options
            are from PyFAI and are can be "q_nm^-1", "q_A^-1", "2th_deg",
            "2th_rad", and "r_mm".
        inplace : bool
            If True (default False), this signal is overwritten. Otherwise,
            returns anew signal.
        kwargs_for_map : dictionary
            Keyword arguments to be passed to self.map().
        kwargs_for_integrator : dictionary
            Keyword arguments to be passed to pyFAI AzimuthalIntegrator().
        kwargs_for_integrate1d : dictionary
            Keyword arguments to be passed to pyFAI ai.integrate1d().


        Returns
        -------
        radial_profile: :obj:`pyxem.signals.ElectronDiffraction1D`
            The radial average profile of each diffraction pattern in the
            ElectronDiffraction2D signal as an ElectronDiffraction1D.

        See also
        --------
        :func:`pyxem.utils.expt_utils.azimuthal_integrate`
        :func:`pyxem.utils.expt_utils.azimuthal_integrate_fast`
        """

        # Scaling factor is used to output the unit in k instead of q.
        # It multiplies the scale that comes out of pyFAI integrate1d
        scaling_factor = 1
        if unit == 'k_A^-1':
            scaling_factor = 1 / 2 / np.pi
            unit = 'q_A^-1'

        if np.array(origin).size == 2:
            # single origin
            # The AzimuthalIntegrator can be defined once and repeatedly used,
            # making for a fast integration
            # this uses azimuthal_integrate_fast

            p1, p2 = origin[0] * detector.pixel1, origin[1] * detector.pixel2
            ai = AzimuthalIntegrator(dist=detector_distance, poni1=p1, poni2=p2,
                                     detector=detector, wavelength=wavelength,
                                     **kwargs_for_integrator)

            azimuthal_integrals = self.map(azimuthal_integrate_fast,
                                           azimuthal_integrator=ai,
                                           size_1d=size_1d, unit=unit,
                                           inplace=inplace,
                                           kwargs_for_integrate1d=kwargs_for_integrate1d,
                                           **kwargs_for_map)

        else:
            # this time each centre is read in origin
            # origin is passed as a flattened array in the navigation dimensions
            azimuthal_integrals = self._map_iterate(azimuthal_integrate,
                                                    iterating_kwargs=(('origin',
                                                                       origin.reshape(-1, 2)),),
                                                    detector_distance=detector_distance,
                                                    detector=detector,
                                                    wavelength=wavelength,
                                                    size_1d=size_1d,
                                                    unit=unit,
                                                    inplace=inplace,
                                                    kwargs_for_integrator=kwargs_for_integrator,
                                                    kwargs_for_integrate1d=kwargs_for_integrate1d,
                                                    **kwargs_for_map)

        if len(azimuthal_integrals.data.shape) == 3:
            ap = Diffraction1D(azimuthal_integrals.data[:, 1, :])
            tth = azimuthal_integrals.data[0, 0, :]  # tth is the signal axis
        else:
            ap = Diffraction1D(azimuthal_integrals.data[:, :, 1, :])
            tth = azimuthal_integrals.data[0, 0, 0, :]  # tth is the signal axis
        scale = (tth[1] - tth[0]) * scaling_factor
        offset = tth[0] * scaling_factor
        ap.axes_manager.signal_axes[0].scale = scale
        ap.axes_manager.signal_axes[0].offset = offset
        ap.axes_manager.signal_axes[0].name = 'scattering'
        ap.axes_manager.signal_axes[0].units = unit

        transfer_navigation_axes(ap, self)
        push_metadata_through(ap, self)

        return ap

    def get_radial_profile(self, mask_array=None, inplace=False,
                           *args, **kwargs):
        """Return the radial profile of the diffraction pattern.

        Parameters
        ----------
        mask_array : numpy.array
            Optional array with the same dimensions as the signal axes.
            Consists of 0s for excluded pixels and 1s for non-excluded
            pixels. The 0-pixels are excluded from the radial average.
        inplace : bool
            If True (default), this signal is overwritten. Otherwise, returns a
            new signal.
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to be passed to map().

        Returns
        -------
        radial_profile: :obj:`pyxem.signals.ElectronDiffraction1D`
            The radial average profile of each diffraction pattern in the
            ElectronDiffraction2D signal as an ElectronDiffraction1D.

        See also
        --------
        :func:`pyxem.utils.expt_utils.radial_average`

        """
        radial_profiles = self.map(radial_average, mask=mask_array,
                                   inplace=inplace,
                                   *args, **kwargs)

        radial_profiles.axes_manager.signal_axes[0].offset = 0
        signal_axis = radial_profiles.axes_manager.signal_axes[0]

        rp = ElectronDiffraction1D(radial_profiles.as_signal1D(signal_axis))
        ax_old = self.axes_manager.navigation_axes
        rp.axes_manager.navigation_axes[0].scale = ax_old[0].scale
        rp.axes_manager.navigation_axes[0].units = ax_old[0].units
        rp.axes_manager.navigation_axes[0].name = ax_old[0].name
        if len(ax_old) > 1:
            rp.axes_manager.navigation_axes[1].scale = ax_old[1].scale
            rp.axes_manager.navigation_axes[1].units = ax_old[1].units
            rp.axes_manager.navigation_axes[1].name = ax_old[1].name
        rp_axis = rp.axes_manager.signal_axes[0]
        rp_axis.name = 'k'
        rp_axis.scale = self.axes_manager.signal_axes[0].scale
        rp_axis.units = '$A^{-1}$'

        return rp

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

        method_dict = {'cross_correlate': find_beam_offset_cross_correlation,
                       'blur': find_beam_center_blur,
                       'interpolate': find_beam_center_interpolate}

        method_function = select_method_from_method_dict(method, method_dict, **kwargs)

        if method == 'cross_correlate':
            shifts = self.map(method_function, inplace=False, **kwargs)
        elif method == 'blur' or method == 'interpolate':
            centers = self.map(method_function, inplace=False, **kwargs)
            shifts = origin_coordinates - centers

        return shifts

    def center_direct_beam(self,
                           method,
                           half_square_width=None,
                           return_shifts=False,
                           *args, **kwargs):
        """Estimate the direct beam position in each experimentally acquired
        electron diffraction pattern and translate it to the center of the
        image square.

        Parameters
        ----------
        method : str,
            Must be one of 'cross_correlate', 'blur', 'interpolate'
        half_square_width  : int
            Half the side length of square that captures the direct beam in all
            scans. Means that the centering algorithm is stable against
            diffracted spots brighter than the direct beam.
        return_shifts : bool
            If True, the values of applied shifts are returned
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

        self.align2D(shifts=shifts, crop=False, fill_value=0)

        if return_shifts:
            return shifts

    def remove_background(self, method,
                          **kwargs):
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
        method_dict = {'h-dome': regional_filter,
                       'gaussian_difference': subtract_background_dog,
                       'median': subtract_background_median,
                       'reference_pattern': subtract_reference, }

        method_function = select_method_from_method_dict(method, method_dict, **kwargs)

        if method != 'h-dome':
            bg_subtracted = self.map(method_function,
                                     inplace=False, **kwargs)
        elif method == 'h-dome':
            scale = self.data.max()
            self.data = self.data / scale
            bg_subtracted = self.map(method_function,
                                     inplace=False, **kwargs)
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
            'zaefferer': find_peaks_zaefferer,
            'stat': find_peaks_stat,
            'laplacian_of_gaussians': find_peaks_log,
            'difference_of_gaussians': find_peaks_dog,
            'xc': find_peaks_xc
        }
        if method in method_dict:
            method = method_dict[method]
        else:
            raise NotImplementedError("The method `{}` is not implemented. "
                                      "See documentation for available "
                                      "implementations.".format(method))

        peaks = self.map(method, *args, **kwargs, inplace=False, ragged=True)
        peaks.map(peaks_as_gvectors,
                  center=np.array(self.axes_manager.signal_shape) / 2 - 0.5,
                  calibration=self.axes_manager.signal_axes[0].scale)
        peaks = DiffractionVectors(peaks)
        peaks.axes_manager.set_signal_dimension(0)

        # Set DiffractionVectors attributes
        peaks.pixel_calibration = self.axes_manager.signal_axes[0].scale
        peaks.detector_shape = self.axes_manager.signal_shape

        # Set calibration to same as signal
        x = peaks.axes_manager.navigation_axes[0]
        y = peaks.axes_manager.navigation_axes[1]

        x.name = 'x'
        x.scale = self.axes_manager.navigation_axes[0].scale
        x.units = 'nm'

        y.name = 'y'
        y.scale = self.axes_manager.navigation_axes[1].scale
        y.units = 'nm'

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
            warn("You have not specified a disc image, as such you will not "
                 "be able to use the xc method in this session")

        peakfinder = peakfinder2D_gui.PeakFinderUIIPYW(
            disc_image=disc_image, imshow_kwargs=imshow_kwargs)
        peakfinder.interactive(self)

    def as_lazy(self, *args, **kwargs):
        """Create a copy of the Diffraction2D object as a
        :py:class:`~pyxem.signals.diffraction1d.LazyDiffraction2D`.

        Parameters
        ----------
        copy_variance : bool
            If True variance from the original Diffraction2D object is copied to
            the new LazyDiffraction2D object.

        Returns
        -------
        res : :py:class:`~pyxem.signals.diffraction1d.LazyDiffraction2D`.
            The lazy signal.
        """
        res = super().as_lazy(*args, **kwargs)
        res.__class__ = LazyDiffraction2D
        res.__init__(**res._to_dictionary())
        return res

    def decomposition(self, *args, **kwargs):
        super().decomposition(*args, **kwargs)
        self.__class__ = Diffraction2D


class LazyDiffraction2D(LazySignal, Diffraction2D):

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, *args, **kwargs):
        super().compute(*args, **kwargs)
        self.__class__ = Diffraction2D
        self.__init__(**self._to_dictionary())

    def decomposition(self, *args, **kwargs):
        super().decomposition(*args, **kwargs)
        self.__class__ = LazyDiffraction2D
