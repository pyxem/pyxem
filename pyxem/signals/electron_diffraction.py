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
"""Signal class for Electron Diffraction data.
"""
import numpy as np
from warnings import warn

from hyperspy.api import interactive
from hyperspy.signals import Signal1D, Signal2D, BaseSignal
from pyxem.signals.diffraction_profile import ElectronDiffractionProfile
from pyxem.signals.diffraction_vectors import DiffractionVectors

from pyxem.utils.expt_utils import _index_coords, _cart2polar, _polar2cart, \
    radial_average, gain_normalise, remove_dead, affine_transformation, \
    regional_filter, subtract_background_dog, subtract_background_median, \
    subtract_reference, circular_mask, find_beam_offset_cross_correlation, \
    peaks_as_gvectors

from pyxem.utils.peakfinders2D import find_peaks_zaefferer, find_peaks_stat, \
    find_peaks_dog, find_peaks_log, find_peaks_xc

from pyxem.utils import peakfinder2D_gui

from skimage import filters
from skimage import transform as tf
from skimage.morphology import square


class ElectronDiffraction(Signal2D):
    _signal_type = "electron_diffraction"

    def __init__(self, *args, **kwargs):
        Signal2D.__init__(self, *args, **kwargs)
        # Set default attributes
        if 'Acquisition_instrument' in self.metadata.as_dictionary():
            if 'SEM' in self.metadata.as_dictionary()['Acquisition_instrument']:
                self.metadata.set_item(
                    "Acquisition_instrument.TEM",
                    self.metadata.Acquisition_instrument.SEM)
                del self.metadata.Acquisition_instrument.SEM
        self.decomposition.__func__.__doc__ = BaseSignal.decomposition.__doc__

    def set_experimental_parameters(self,
                                    accelerating_voltage=None,
                                    camera_length=None,
                                    scan_rotation=None,
                                    convergence_angle=None,
                                    rocking_angle=None,
                                    rocking_frequency=None,
                                    exposure_time=None):
        """Set experimental parameters in metadata.

        Parameters
        ----------
        accelerating_voltage : float
            Accelerating voltage in kV
        camera_length: float
            Camera length in cm
        scan_rotation : float
            Scan rotation in degrees
        convergence_angle : float
            Convergence angle in mrad
        rocking_angle : float
            Beam rocking angle in mrad
        rocking_frequency : float
            Beam rocking frequency in Hz
        exposure_time : float
            Exposure time in ms.
        """
        md = self.metadata

        if accelerating_voltage is not None:
            md.set_item("Acquisition_instrument.TEM.accelerating_voltage",
                        accelerating_voltage)
        if camera_length is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.Diffraction.camera_length",
                camera_length)
        if scan_rotation is not None:
            md.set_item("Acquisition_instrument.TEM.scan_rotation",
                        scan_rotation)
        if convergence_angle is not None:
            md.set_item("Acquisition_instrument.TEM.convergence_angle",
                        convergence_angle)
        if rocking_angle is not None:
            md.set_item("Acquisition_instrument.TEM.rocking_angle",
                        rocking_angle)
        if rocking_frequency is not None:
            md.set_item("Acquisition_instrument.TEM.rocking_frequency",
                        rocking_frequency)
        if exposure_time is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.Diffraction.exposure_time",
                exposure_time)

    def set_diffraction_calibration(self, calibration, center=None):
        """Set diffraction pattern pixel size in reciprocal Angstroms and origin
        location.

        Parameters
        ----------
        calibration : float
            Diffraction pattern calibration in reciprocal Angstroms per pixel.
        center : tuple
            Position of the direct beam center, in pixels. If None the center of
            the data array is assumed to be the center of the pattern.
        """
        if center is None:
            center = np.array(self.axes_manager.signal_shape) / 2 * calibration

        dx = self.axes_manager.signal_axes[0]
        dy = self.axes_manager.signal_axes[1]

        dx.name = 'kx'
        dx.scale = calibration
        dx.offset = -center[0]
        dx.units = '$A^{-1}$'

        dy.name = 'ky'
        dy.scale = calibration
        dy.offset = -center[1]
        dy.units = '$A^{-1}$'

    def set_scan_calibration(self, calibration):
        """Set scan pixel size in nanometres.

        Parameters
        ----------
        calibration: float
            Scan calibration in nanometres per pixel.
        """
        x = self.axes_manager.navigation_axes[0]
        y = self.axes_manager.navigation_axes[1]

        x.name = 'x'
        x.scale = calibration
        x.units = 'nm'

        y.name = 'y'
        y.scale = calibration
        y.units = 'nm'

    def plot_interactive_virtual_image(self, roi, **kwargs):
        """Plots an interactive virtual image formed with a specified and
        adjustable roi.

        Parameters
        ----------
        roi : :obj:`hyperspy.roi.BaseInteractiveROI`
            Any interactive ROI detailed in HyperSpy.
        **kwargs:
            Keyword arguments to be passed to `ElectronDiffraction.plot`

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
                                    inplace=True,
                                    *args, **kwargs):
        """Correct geometric distortion by applying an affine transformation.

        Parameters
        ----------
        D : array
            3x3 np.array specifying the affine transform to be applied.
        inplace : bool
            If True (default), this signal is overwritten. Otherwise, returns a
            new signal.
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to be passed to map().

        Returns
        -------
            ElectronDiffraction Signal containing the affine Transformed
            diffraction patterns.

        """
        # Account for the transformation center not being (0,0)
        shape = self.axes_manager.signal_shape
        shift_x = (shape[1] - 1) / 2
        shift_y = (shape[0] - 1) / 2

        tf_shift = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
        tf_shift_inv = tf.SimilarityTransform(translation=[shift_x, shift_y])

        # This defines the transform you want to perform
        distortion = tf.AffineTransform(matrix=D)

        # skimage transforms can be added like this, does matrix multiplication,
        # hence the need for the brackets. (Note tf.warp takes the inverse)
        transformation = (tf_shift + (distortion + tf_shift_inv)).inverse

        return self.map(affine_transformation,
                        transformation=transformation,
                        order=order,
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
        dark_reference : ElectronDiffraction
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
        deadpixels : ElectronDiffraction
            List
        deadvalue : string
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
        radial_profile: :obj:`hyperspy.signals.Signal1D`
            The radial average profile of each diffraction pattern in the
            ElectronDiffraction signal as a Signal1D.

        See also
        --------
        :func:`pyxem.utils.expt_utils.radial_average`

        """
        radial_profiles = self.map(radial_average, mask=mask_array,
                                   inplace=inplace,
                                   *args, **kwargs)

        radial_profiles.axes_manager.signal_axes[0].offset = 0
        signal_axis = radial_profiles.axes_manager.signal_axes[0]

        rp = ElectronDiffractionProfile(radial_profiles.as_signal1D(signal_axis))
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

    def get_direct_beam_position(self, radius_start,
                                 radius_finish,
                                 *args, **kwargs):
        """Estimate the direct beam position in each experimentally acquired
        electron diffraction pattern.

        Parameters
        ----------
        radius_start : int
            The lower bound for the radius of the central disc to be used in the
            alignment.
        radius_finish : int
            The upper bounds for the radius of the central disc to be used in
            the alignment.
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to be passed to map().

        Returns
        -------
        centers : ndarray
            Array containing the centers for each SED pattern.

        """
        shifts = self.map(find_beam_offset_cross_correlation,
                          radius_start=radius_start,
                          radius_finish=radius_finish,
                          inplace=False, *args, **kwargs)
        return shifts

    def center_direct_beam(self,
                           radius_start, radius_finish,
                           square_width=None,
                           *args, **kwargs):
        """Estimate the direct beam position in each experimentally acquired
        electron diffraction pattern and translate it to the center of the
        image square.

        Parameters
        ----------
        radius_start : int
            The lower bound for the radius of the central disc to be used in the
            alignment.
        radius_finish : int
            The upper bounds for the radius of the central disc to be used in
            the alignment.
        square_width  : int
            Half the side length of square that captures the direct beam in all
            scans. Means that the centering algorithm is stable against
            diffracted spots brighter than the direct beam.
        *args:
            Arguments to be passed to align2D().
        **kwargs:
            Keyword arguments to be passed to align2D().

        Returns
        -------
        centered : ElectronDiffraction
            The centered diffraction data.

        """
        nav_shape_x = self.data.shape[0]
        nav_shape_y = self.data.shape[1]
        origin_coordinates = np.array((self.data.shape[2] / 2 - 0.5,
                                       self.data.shape[3] / 2 - 0.5))

        if square_width is not None:
            min_index = np.int(origin_coordinates[0] - (0.5 + square_width))
            # fails if non-square dp
            max_index = np.int(origin_coordinates[0] + (1.5 + square_width))
            shifts = self.isig[min_index:max_index, min_index:max_index].get_direct_beam_position(
                radius_start, radius_finish, *args, **kwargs)
        else:
            shifts = self.get_direct_beam_position(radius_start, radius_finish,
                                                   *args, **kwargs)

        shifts = -1 * shifts.data
        shifts = shifts.reshape(nav_shape_x * nav_shape_y, 2)

        return self.align2D(shifts=shifts, crop=False, fill_value=0,
                            *args, **kwargs)

    def remove_background(self, method,
                          *args, **kwargs):
        """Perform background subtraction via multiple methods.

        Parameters
        ----------
        method : string
            Specify the method used to determine the direct beam position.

            * 'h-dome' -
            * 'gaussian_difference' - Uses a difference between two gaussian
                                convolutions to determine where the peaks are,
                                and sets all other pixels to 0.
            * 'median' - Use a median filter for background removal
            * 'reference_pattern' - Subtract a user-defined reference patterns
                from every diffraction pattern.

        sigma_min : int, float
            Standard deviation for the minimum gaussian convolution
            (gaussian_difference only)
        sigma_max : int, float
            Standard deviation for the maximum gaussian convolution
            (gaussian_difference only)
        footprint : int
            Size of the window that is convoluted with the array to determine
            the median. Should be large enough that it is about 3x as big as the
            size of the peaks (median only).
        implementation : 'scipy' or 'skimage'
            (median only) see expt_utils.subtract_background_median
            for details, if not selected 'scipy' is used
        bg : array
            Background array extracted from vacuum. (subtract_reference only)
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to be passed to map().

        Returns
        -------
        bg_subtracted : :obj:`ElectronDiffraction`
            A copy of the data with the background subtracted. Be aware that
            this function will only return inplace.

        """
        if method == 'h-dome':
            scale = self.data.max()
            self.data = self.data / scale
            bg_subtracted = self.map(regional_filter,
                                     inplace=False, *args, **kwargs)
            bg_subtracted.map(filters.rank.mean, selem=square(3))
            bg_subtracted.data = bg_subtracted.data / bg_subtracted.data.max()

        elif method == 'gaussian_difference':
            bg_subtracted = self.map(subtract_background_dog,
                                     inplace=False, *args, **kwargs)

        elif method == 'median':
            if 'implementation' in kwargs.keys():
                if kwargs['implementation'] != 'scipy' and kwargs['implementation'] != 'skimage':
                    raise NotImplementedError(
                        "Unknown implementation `{}`".format(
                            kwargs['implementation']))

            bg_subtracted = self.map(subtract_background_median,
                                     inplace=False, *args, **kwargs)

        elif method == 'reference_pattern':
            bg_subtracted = self.map(subtract_reference, inplace=False,
                                     *args, **kwargs)

        else:
            raise NotImplementedError(
                "The method specified, '{}', is not implemented. See"
                "documentation for available implementations.".format(method))

        return bg_subtracted

    def decomposition(self, *args, **kwargs):
        """Decomposition with a choice of algorithms.

        Parameters
        ----------
        *args :
            Arguments to be passed to decomposition().
        **kwargs :
            Keyword arguments to be passed to decomposition().

        Returns
        -------
        The results are stored in self.learning_results. For a full description
        of parameters see :meth:`hyperspy.learn.mva.MVA.decomposition`

        """
        super(Signal2D, self).decomposition(*args, **kwargs)
        self.learning_results.loadings = np.nan_to_num(
            self.learning_results.loadings)

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
            the original ElectronDiffraction object. Each signal is a BaseSignal
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
