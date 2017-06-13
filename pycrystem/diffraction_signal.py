# -*- coding: utf-8 -*-
# Copyright 2017 The PyCrystEM developers
#
# This file is part of PyCrystEM.
#
# PyCrystEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyCrystEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyCrystEM.  If not, see <http://www.gnu.org/licenses/>.
"""Signal class for Electron Diffraction data

"""

from __future__ import division

from hyperspy.api import interactive, stack
from hyperspy.components1d import Voigt, Exponential, Polynomial
from hyperspy.signals import Signal2D, BaseSignal

from pycrystem.utils.expt_utils import *
from pycrystem.utils.peakfinders2D import *


class ElectronDiffraction(Signal2D):
    _signal_type = "electron_diffraction"

    def __init__(self, *args, **kwargs):
        Signal2D.__init__(self, *args, **kwargs)
        # Attributes defaults
        if 'Acquisition_instrument.TEM' not in self.metadata:
            if 'Acquisition_instrument.SEM' in self.metadata:
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
        """Set the experimental parameters in metadata.

        Parameters
        ----------
        accelerating_voltage: float
            Accelerating voltage in kV
        camera_length: float
            Camera length in cm
        scan_rotation: float
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
        if camera_length is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.Diffraction.camera_length",
                camera_length
            )
        if exposure_time is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.Diffraction.exposure_time",
                exposure_time
            )

    def set_calibration(self, calibration, center=None):
        """Set pixel size in reciprocal Angstroms and origin location.

        Parameters
        ----------
        calibration: float
            Calibration in reciprocal Angstroms per pixel
        center: tuple
            Position of the central beam, in pixels. If None the center of the
            frame is assumed to be the center of the pattern.
        """
        # TODO: extend to get calibration from a list of stored calibrations for
        # the camera length recorded in metadata.
        if center is None:
            center = np.array(self.axes_manager.signal_shape)/2 * calibration

        dx = self.axes_manager.signal_axes[0]
        dy = self.axes_manager.signal_axes[1]

        dx.name = 'dx'
        dx.scale = calibration
        dx.offset = -center[0]
        dx.units = '$A^{-1}$'

        dy.name = 'dy'
        dy.scale = calibration
        dy.offset = -center[1]
        dy.units = '$A^{-1}$'

    def plot_interactive_virtual_image(self, roi):
        """Plots an interactive virtual image formed with a specified and
        adjustable roi.

        Parameters
        ----------
        roi: :obj:`hyperspy.roi.BaseInteractiveROI`
            Any interactive ROI detailed in HyperSpy.

        Examples
        --------
        .. code-block:: python

            import hyperspy.api as hs
            roi = hs.roi.CircleROI(0, 0, 0.2)
            data.plot_interactive_virtual_image(roi)

        """
        self.plot()
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
        dark_field_sum: :obj:`hyperspy.signals.BaseSignal`
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
        # TODO: make outputs neat in obvious cases i.e. 2D for normal vdf
        return dark_field_sum

    def get_direct_beam_mask(self, radius, center=None):
        """Generate a signal mask for the direct beam.

        Parameters
        ----------
        radius : float
            Radius for the circular mask in pixel units.
        center : tuple, optional
            User specified (x, y) position of the diffraction pattern center.
            i.e. the direct beam position. If None (default) it is assumed that
            the direct beam is at the center of the diffraction pattern.

        Return
        ------
        signal-mask : ndarray
            The mask of the direct beam
        """
        shape = self.axes_manager.signal_shape
        if center is None:
            center = (shape[1] - 1) / 2, (shape[0] - 1) / 2

        signal_mask = Signal2D(circular_mask(shape=shape,
                                             radius=radius,
                                             center=center))

        return signal_mask

    def get_vacuum_mask(self, radius, threshold, center=None,
                        closing=True, opening=False):
        """Generate a navigation mask to exclude SED patterns acquired in vacuum.

        Vacuum regions are identified crudely based on searching for a peak
        value in each diffraction pattern, having masked the direct beam, above
        a user defined threshold value. Morphological opening or closing of the
        mask obtained is supported.

        Parameters
        ----------
        radius: float
            Radius of circular mask to exclude direct beam.
        threshold: float
            Minimum intensity required to consider a diffracted beam to be
            present.
        center: tuple, optional
            User specified position of the diffraction pattern center. If None
            it is assumed that the pattern center is the center of the image.
        closing: bool, optional
            Flag to perform morphological closing.
        opening: bool, optional
            Flag to perform morphological opening.

        Returns
        -------
        mask : Signal2D
            The mask of the region of interest. Vacuum regions to be masked are
            set True.

        See also
        --------
        get_direct_beam_mask
        """
        db = np.invert(self.get_direct_beam_mask(radius=radius, center=center))
        diff_only = db * self
        mask = (diff_only.max((-1, -2)) <= threshold)
        if closing:
            mask.data = ndi.morphology.binary_dilation(mask.data,
                                                       border_value=0)
            mask.data = ndi.morphology.binary_erosion(mask.data,
                                                      border_value=1)
        if opening:
            mask.data = ndi.morphology.binary_erosion(mask.data,
                                                      border_value=1)
            mask.data = ndi.morphology.binary_dilation(mask.data,
                                                       border_value=0)
        return mask

    def apply_affine_transformation(self, D, inplace=True):
        """Correct geometric distortion by applying an affine transformation.

        Parameters
        ----------
        D : 3x3 numpy array
            Specifies the affine transform to be applied.
        inplace : bool
            If True (default), this signal is overwritten. Otherwise, returns a
            new signal.

        """
        return self.map(
            affine_transformation, matrix=D, ragged=True, inplace=inplace
        )

    def gain_normalisation(self, dark_reference, bright_reference,
                           inplace=True):
        """Apply gain normalization to experimentally acquired electron
        diffraction patterns.

        Parameters
        ----------
        dark_reference : ElectronDiffraction
            Dark reference image.
        bright_reference : ElectronDiffraction
            Bright reference image.
        inplace : bool
            If True (default), this signal is overwritten. Otherwise, returns a
            new signal.

        """
        return self.map(gain_normalise, dref=dark_reference,
                        bref=bright_reference, inplace=inplace)

    def get_radial_profile(self, centers=None):
        """Return the radial profile of the diffraction pattern.

        Parameters
        ----------
        centers : array, optional
            Array of dimensions (navigation_shape, 2) containing the
            origin for the radial integration in each diffraction
            pattern. If None (default) the centers are calculated using
            :meth:`get_direct_beam_position`

        Returns
        -------
        radial_profile: :obj:`hyperspy.signals.Signal1D`
            The radial average profile of each diffraction pattern
            in the ElectronDiffraction signal as a Signal1D.

        See also
        --------
        :func:`~pycrystem.utils.expt_utils.radial_average`
        :meth:`get_direct_beam_position`

        """
        # TODO: make this work without averaging the centers
        # TODO: fix for case when data is singleton
        if centers is None:
            centers = self.get_direct_beam_position(radius=10)
        center = centers.mean(axis=(0, 1))
        radial_profiles = self.map(radial_average, center=center, inplace=False)
        radial_profiles.axes_manager.signal_axes[0].offset = 0
        signal_axis = radial_profiles.axes_manager.signal_axes[0]
        return radial_profiles.as_signal1D(signal_axis)

    def get_diffraction_variance(self):
        """Calculates the variance of associated with each diffraction pixel.

        Returns
        -------
        ElectronDiffraction
              A two dimensional signal containing the mean,
              mean squared, and variance.
        """
        mean = self.mean(axis=self.axes_manager.navigation_axes)
        square = np.square(self)
        meansquare = square.mean(axis=square.axes_manager.navigation_axes)
        variance = meansquare / np.square(mean) - 1
        return stack((mean, meansquare, variance))

    def get_direct_beam_position(self, radius):
        """Determine rigid shifts in the SED patterns based on the position of
        the direct beam and return the shifts required to center all patterns.

        Parameters
        ----------
        radius : int
            Defines the size of the circular region, in pixels, within which the
            direct beam position is refined.

        Returns
        -------
        shifts : ndarray
            Array containing the shift to be applied to each SED pattern to
            center it.

        """
        # sum images to produce image in which direct beam reinforced and take
        # the position of maximum intensity as the initial estimate of center.
        dp_sum = self.sum()
        maxes = np.asarray(np.where(dp_sum.data == dp_sum.data.max()))
        mref = np.rint([np.average(maxes[0]), np.average(maxes[1])])
        mref = mref.astype(int)
        # specify array of dims (nav_size, 2) in which to store centers and find
        # the center of each pattern by determining the direct beam position.
        arr_shape = (self.axes_manager.navigation_size, 2)
        c = np.zeros(arr_shape, dtype=int)
        for z, index in zip(self._iterate_signal(),
                            np.arange(0, self.axes_manager.navigation_size, 1)):
            c[index] = refine_beam_position(z, start=mref, radius=radius)
        # The arange function produces a linear set of centers that has to be
        # reshaped back to the original signal shape
        return c.reshape(self.axes_manager.navigation_shape[::-1] + (-1,))

    def remove_background(self, saturation_radius=0):
        """Perform background subtraction.

        The average radial profile of the data is used to model the background
        which is then subtracted from the data. The remaining noise is smoothed
        using an h-dome.

        Parameters
        ----------
        saturation_radius : int, optional
            The radius, in pixels, of the saturated data (if any) in the direct
            beam.

        Returns
        -------
        denoised : :obj:`ElectronDiffraction`
            A copy of the data with the background removed and the noise
            smoothed.

        See Also
        --------
        :meth:`get_background_model`

        """
        # TODO: separate into multiple methods
        # TODO: make an 'averaging' flag

        # Old method:
        # def mean_filter(h):
        #     self.data = self.data / self.data.max()
        #     self.map(regional_filter, h=h)
        #     self.map(filters.rank.mean, selem=square(3))
        #     self.data = self.data / self.data.max()
        bg = self.get_background_model(saturation_radius)

        bg_removed = np.clip(self - bg, 0, 255)

        denoised = ElectronDiffraction(
            bg_removed.map(regional_flattener, h=bg.data.min()-1, inplace=False)
        )
        denoised.axes_manager.update_axes_attributes_from(
            self.axes_manager.navigation_axes)
        denoised.axes_manager.update_axes_attributes_from(
            self.axes_manager.signal_axes)

        return denoised

    def get_background_model(self, saturation_radius):
        """Creates a model for the background of the signal.

        The mean radial profile is fitted with the following three components:

        * Voigt profile for the central beam
        * Exponential profile for the diffuse scatter
        * Linear profile for the background offset and to improve the fit

        Using the exponential profile and the linear profile, an
        ElectronDiffraction signal is produced representing the mean background
        of the signal. This may be used for background subtraction.

        Parameters
        ----------
        saturation_radius : int
            The radius of the region about the central beam in which pixels are
            saturated.

        Returns
        -------
        ElectronDiffraction
            The mean background of the signal.

        """
        # TODO: get this done without taking the mean

        profile = self.get_radial_profile().mean()
        model = profile.create_model()
        e1 = saturation_radius * profile.axes_manager.signal_axes[0].scale
        model.set_signal_range(e1)

        direct_beam = Voigt()
        direct_beam.centre.value = 0
        direct_beam.centre.free = False
        direct_beam.FWHM.value = 0.1
        direct_beam.area.bmin = 0
        model.append(direct_beam)

        diffuse_scatter = Exponential()
        diffuse_scatter.A.value = 0
        diffuse_scatter.A.bmin = 0
        diffuse_scatter.tau.value = 0
        diffuse_scatter.tau.bmin = 0
        model.append(diffuse_scatter)

        linear_decay = Polynomial(1)
        model.append(linear_decay)

        model.fit(bounded=True)

        x_axis = self.axes_manager.signal_axes[0].axis
        y_axis = self.axes_manager.signal_axes[1].axis
        xs, ys = np.meshgrid(x_axis, y_axis)
        rs = (xs ** 2 + ys ** 2) ** 0.5
        bg = ElectronDiffraction(
            diffuse_scatter.function(rs) + linear_decay.function(rs))
        for i in (0, 1):
            bg.axes_manager.signal_axes[i].update_from(
                self.axes_manager.signal_axes[i])
        return bg

    def decomposition(self, *args, **kwargs):
        """Decomposition with a choice of algorithms.

        The results are stored in self.learning_results. For a full description
        of parameters see :meth:`hyperspy.learn.mva.MVA.decomposition`

        """
        super(Signal2D, self).decomposition(*args, **kwargs)
        self.learning_results.loadings = np.nan_to_num(
            self.learning_results.loadings)

    def find_peaks(self, method='skimage', *args, **kwargs):
        """Find the position of diffraction peaks.

        Function to locate the positive peaks in an image using various, user
        specified, methods. Returns a structured array containing the peak
        positions.

        Parameters
        ---------
        method : str
            Select peak finding algorithm to implement. Available methods are:

            * 'max' - simple local maximum search
            * 'skimage' - call the peak finder implemented in scikit-image which
              uses a maximum filter
            * 'minmax' - finds peaks by comparing maximum filter results
              with minimum filter, calculates centers of mass
            * 'zaefferer' - based on gradient thresholding and refinement
              by local region of interest optimisation
            * 'stat' - statistical approach requiring no free params.
            * 'massiel' - finds peaks in each direction and compares the
              positions where these coincide.
            * 'laplacian_of_gaussians' - a blob finder implemented in
              `scikit-image` which uses the laplacian of Gaussian matrices
              approach.
            * 'difference_of_gaussians' - a blob finder implemented in
              `scikit-image` which uses the difference of Gaussian matrices
              approach.

        *args
            associated with above methods
        **kwargs
            associated with above methods.

        Returns
        -------
        peaks : structured array
            An array of shape _navigation_shape_in_array in which
            each cell contains an array with dimensions (npeaks, 2) that
            contains the x, y pixel coordinates of peaks found in each image.
        """
        method_dict = {
            'skimage': peak_local_max,
            'max': find_peaks_max,
            'minmax': find_peaks_minmax,
            'zaefferer': find_peaks_zaefferer,
            'stat': find_peaks_stat,
            'laplacian_of_gaussians':  find_peaks_log,
            'difference_of_gaussians': find_peaks_dog,
        }
        if method in method_dict:
            method = method_dict[method]
        else:
            raise NotImplementedError("The method `{}` is not implemented. "
                                      "See documentation for available "
                                      "implementations.".format(method))
        peaks = self.map(method, *args, **kwargs, inplace=False, ragged=True)
        # TODO: make return DiffractionVectors(peaks)
        return peaks

    def find_peaks_interactive(self):
        """Find peaks using an interactive tool.

        Requires `ipywidgets` and `traitlets` to be installed.

        """
        from pycrystem.utils import peakfinder2D_gui
        peakfinder = peakfinder2D_gui.PeakFinderUIIPYW()
        peakfinder.interactive(self)
