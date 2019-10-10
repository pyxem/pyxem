# -*- coding: utf-8 -*-
# Copyright 2017-2019 The pyXem developers
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

from pyxem.signals.electron_diffraction1d import ElectronDiffraction1D
from pyxem.signals.detector_coordinates2d import DetectorCoordinates2D
from pyxem.signals import push_metadata_through, select_method_from_method_dict

from pyxem.utils.expt_utils import _index_coords, _cart2polar, _polar2cart, \
    radial_average, gain_normalise, remove_dead,\
    regional_filter, subtract_background_dog, subtract_background_median, \
    subtract_reference, circular_mask, find_beam_offset_cross_correlation, \
    peaks_as_gvectors, convert_affine_to_transform, apply_transformation, \
    find_beam_center_blur, find_beam_center_interpolate

from pyxem.utils.peakfinders2D import find_peaks_zaefferer, find_peaks_stat, \
    find_peaks_dog, find_peaks_log, find_peaks_xc

from pyxem.utils import peakfinder2D_gui

from skimage import filters
from skimage import transform as tf
from skimage.morphology import square


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
                           square_width=None,
                           *args, **kwargs):
        """Estimate the direct beam position in each experimentally acquired
        electron diffraction pattern and translate it to the center of the
        image square.

        Parameters
        ----------
        method : str,
            Must be one of 'cross_correlate', 'blur', 'interpolate'

        square_width  : int
            Half the side length of square that captures the direct beam in all
            scans. Means that the centering algorithm is stable against
            diffracted spots brighter than the direct beam.

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

        if square_width is not None:
            min_index = np.int(origin_coordinates[0] - square_width)
            # fails if non-square dp
            max_index = np.int(origin_coordinates[0] + square_width)
            cropped = self.isig[min_index:max_index, min_index:max_index]
            shifts = cropped.get_direct_beam_position(method=method, **kwargs)
        else:
            shifts = self.get_direct_beam_position(method=method, **kwargs)

        shifts = -1 * shifts.data
        shifts = shifts.reshape(nav_size, 2)

        return self.align2D(shifts=shifts, crop=False, fill_value=0)

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

    def find_peaks(self, method,
                   *args, **kwargs):
        """Determine the coordinates in the detector plane of positive peaks
        using various user defined methods.

        Parameters
        ---------
        method : str
            Specifies the method, from:
            {'zaefferer', 'stat', 'laplacian_of_gaussians',
            'difference_of_gaussians', 'xc'}
        **kwargs :
            Method specific keyword arguments to be passed to map().
            If None, the method speicic kward documentation will be returned.

        Returns
        -------
        peak_coordinates : DetectorCoordinates2D
            The pixel coordinates of peaks found in the Diffraction2D signal,
            with navigation dimensions identical to the Diffraction2D object.

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
            'zaefferer':
            {'method': find_peaks_zaefferer,
             'params': ['grad_threshold', 'window_size', 'distance_cutoff']},
            'stat':
            {'method': find_peaks_stat,
             'params': ['alpha', 'window_raidus', 'convergence_ratio']},
            'difference_of_gaussians':
            {'method': find_peaks_dog,
             'params': ['min_sigma', 'max_sigma', 'sigma_ratio', 'threshold',
                        'overlap', 'exclude_border']},
            'laplacian_of_gaussians':
            {'method': find_peaks_log,
             'params': ['min_sigma', 'max_sigma', 'num_sigma', 'threshold',
                        'overlap', 'log_scale', 'exclude_border']},
            'xc':
            {'method': find_peaks_xc,
             'params': ['disc_image', 'min_distance', 'peak_threshold']},
        }

        if method not in method_dict:
            raise NotImplementedError("The method `{}` is not implemented. "
                                      "See documentation for available "
                                      "implementations.".format(method))
        if not kwargs:
            for kwarg in method_dict[method]['params']:
                print("You need the `{}` kwarg".format(kwarg))
            return None

        peak_coordinates = self.map(method_dict[method]['method'], **kwargs,
                                    inplace=False, ragged=True)
        # Set calibration to same as signal
        x = peak_coordinates.axes_manager.navigation_axes[0]
        y = peak_coordinates.axes_manager.navigation_axes[1]

        x.name = 'x'
        x.scale = self.axes_manager.navigation_axes[0].scale
        x.units = 'nm'

        y.name = 'y'
        y.scale = self.axes_manager.navigation_axes[1].scale
        y.units = 'nm'

        return peak_coordinates

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
