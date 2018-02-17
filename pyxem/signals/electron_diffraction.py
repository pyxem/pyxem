# -*- coding: utf-8 -*-
# Copyright 2017-2018 The pyXem developers
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
"""Signal class for Electron Diffraction data

"""

from hyperspy._signals.lazy import LazySignal
from hyperspy.api import interactive, stack
from hyperspy.components1d import Voigt, Exponential, Polynomial
from hyperspy.signals import Signal1D, Signal2D, BaseSignal
from pyxem.signals.diffraction_profile import DiffractionProfile
from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.utils.expt_utils import *
from pyxem.utils.peakfinders2D import *
from pyxem.utils import peakfinder2D_gui


class ElectronDiffraction(Signal2D):
    _signal_type = "electron_diffraction"

    def __init__(self, *args, **kwargs):
        Signal2D.__init__(self, *args, **kwargs)
        # Set default attributes
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

        Returns
        -------
            ElectronDiffraction Signal containing the affine Transformed
            diffraction patterns.

        """
        return self.map(affine_transformation,
                        matrix=D,
                        order=order,
                        inplace=inplace)

    def apply_gain_normalisation(self,
                                 dark_reference,
                                 bright_reference,
                                 inplace=True):
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

        """
        return self.map(gain_normalise,
                        dref=dark_reference,
                        bref=bright_reference,
                        inplace=inplace)

    def remove_deadpixels(self,
                          deadpixels,
                          deadvalue='average',
                          inplace=True):
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

        """
        return self.map(remove_dead,
                        deadpixels=deadpixels,
                        deadvalue=deadvalue,
                        inplace=inplace)

    def get_radial_profile(self, centers=None, cython=False):
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
        :func:`pyxem.utils.expt_utils.radial_average`
        :meth:`get_direct_beam_position`

        Examples
        --------
        .. code-block:: python

            centers = ed.get_direct_beam_position(method="blur")
            profiles = ed.get_radial_profile(centers)
            profiles.plot()
        """
        if centers is None:
            centers = self.get_direct_beam_position(radius=10)
        centers = Signal1D(centers)

        rp = self.map(radial_average, center=centers,
                      cython=cython, inplace=False)

        ragged = len(rp.data.shape) == 1

        if ragged:
            max_len = max(map(len, rp.data))
            rp = Signal1D([
                np.pad(row.reshape(-1,), (0, max_len-len(row)), mode="constant", constant_values=0)
                for row in rp.data])
        else:
            rp.axes_manager.signal_axes[0].offset = 0
            signal_axis = radial_profiles.axes_manager.signal_axes[0]
            radial_profiles.as_signal1D(signal_axis)

        return DiffractionProfile(rp)

    def reproject_as_polar(self, origin=None, jacobian=False, dr=1, dt=None):
        """Reproject the diffraction data into polar coordinates.

        Parameters
        ----------
        origin : tuple
            The coordinate (x0, y0) of the image center, relative to bottom-left.
            If 'None'defaults to the center of the pattern.
        Jacobian : boolean
            Include ``r`` intensity scaling in the coordinate transform.
            This should be included to account for the changing pixel size that
            occurs during the transform.
        dr : float
            Radial coordinate spacing for the grid interpolation
            tests show that there is not much point in going below 0.5
        dt : float
            Angular coordinate spacing (in radians)
            if ``dt=None``, dt will be set such that the number of theta values
            is equal to the maximum value between the height or the width of
            the image.

        Returns
        -------
        output : ElectronDiffraction
            The electron diffraction data in polar coordinates.

        """
        return self.map(reproject_polar,
                        origin=origin,
                        jacobian=jacobian,
                        dr=dr, dt=dt)

    # TODO: This method needs to keep track of what's what better, with labels
    # axes also need to track calibrations.
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

    def get_direct_beam_position(self,
                                 method='blur',
                                 sigma=30,
                                 *args, **kwargs):
        """Estimate the direct beam position in each experimentally acquired
        electron diffraction pattern.

        Parameters
        ----------
        method : string
            Specify the method used to determine the direct beam position.

            * 'blur' - Use gaussian filter to blur the image and take the
                pixel with the maximum intensity value as the center
            * 'refine_local' - Refine the position of the direct beam and
                hence an estimate for the position of the pattern center in
                each SED pattern.

        sigma : int
            Standard deviation for the gaussian convolution (only for
            'blur' method).

        Returns
        -------
        centers : ndarray
            Array containing the shift to be applied to each SED pattern to
            center it.

        """
        #TODO: add sub-pixel capabilities and model fitting methods.
        if method == 'blur':
            centers = self.map(find_beam_position_blur,
                               sigma=sigma, inplace=False)

        elif method == 'refine_local':
            if initial_center==None:
                initial_center = np.int(self.signal_axes.shape / 2)

            centers = self.map(refine_beam_position,
                               initial_center=initial_center,
                               radius=radius,
                               inplace=False)

        else:
            raise NotImplementedError("The method specified is not implemented. "
                                      "See documentation for available "
                                      "implementations.")
        return centers

    def remove_background(self, method='model', *args, **kwargs):
        """Perform background subtraction via multiple methods.

        Parameters
        ----------
        method : string
            Specify the method used to determine the direct beam position.

            * 'h-dome' -
            * 'model' - fit a model to the radial profile of the average
                diffraction pattern and then smooth remaining noise using
                an h-dome method.
            * 'gaussian_difference' - Uses a difference between two gaussian
				convolutions to determine where the peaks are, and sets
				all other pixels to 0.
            * 'median' - Use a median filter for background removal
            * 'reference_pattern' - Subtract a user-defined reference patterns
                from every diffraction pattern.

        saturation_radius : int, optional
            The radius, in pixels, of the saturated data (if any) in the direct
            beam if the model method is used (h-dome / model only).
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
        bg : array
            Background array extracted from vacuum. (subtract_reference only)

        Returns
        -------
        bg_subtracted : :obj:`ElectronDiffraction`
            A copy of the data with the background subtracted.

        See Also
        --------
        :meth:`get_background_model`

        """
        if method == 'h-dome':
            scale = self.data.max()
            self.data = self.data / scale
            bg_subtracted = self.map(regional_filter,
                                     inplace=False, *args, **kwargs)
            bg_subtracted.map(filters.rank.mean, selem=square(3))
            bg_subtracted.data = bg_subtracted.data / bg_subtracted.data.max()

        elif method == 'model':
            bg = self.get_background_model(*args, **kwargs)

            bg_removed = np.clip(self - bg, self.min(), self.max())

            h = max(bg.data.min(), 1e-6)
            bg_subtracted = ElectronDiffraction(
                bg_removed.map(regional_flattener, h=h, inplace=False))
            bg_subtracted.axes_manager.update_axes_attributes_from(
                self.axes_manager.navigation_axes)
            bg_subtracted.axes_manager.update_axes_attributes_from(
                self.axes_manager.signal_axes)

        elif method == 'gaussian_difference':
            bg_subtracted = self.map(subtract_background_dog,
                                     inplace=False, *args, **kwargs)

        elif method == 'median':
            bg_subtracted = self.map(subtract_background_median,
                                     inplace=False, *args, **kwargs)

        elif method == 'reference_pattern':
            bg_subtracted = self.map(subtract_reference, *args, **kwargs)

        else:
            raise NotImplementedError(
                "The method specified, '{}', is not implemented. See"
                "documentation for available implementations.".format(method))

        return bg_subtracted

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

    def get_no_diffraction_mask(self, *args, **kwargs):
        """Identify electron diffraction patterns containing no diffraction
        peaks to remove from further processing.

        Parameters
        ----------
        method : string
            Choice of method

        Returns
        -------
        mask : Signal
            Signal object containing the mask.
        """
        #TODO: Make this actually work.
        if method == 'shapiro-wilk':
            shapiro_values = self.map(stats.shapiro)
            mask = shapiro_values > threshold

        elif method == 'threshold':
            mask = self.sum((2,3)) > threshold

        else:
            raise NotImplementedError("The method specified is not implemented. "
                                      "See documentation for available "
                                      "implementations.")

        return mask

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
            * 'laplacian_of_gaussians' - a blob finder implemented in
              `scikit-image` which uses the laplacian of Gaussian matrices
              approach.
            * 'difference_of_gaussians' - a blob finder implemented in
              `scikit-image` which uses the difference of Gaussian matrices
              approach.
            * 'regionprops' - Uses regionprops to find islands of connected
               pixels representing a peak

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
            'regionprops': find_peaks_regionprops,
        }
        if method in method_dict:
            method = method_dict[method]
        else:
            raise NotImplementedError("The method `{}` is not implemented. "
                                      "See documentation for available "
                                      "implementations.".format(method))
        peaks = self.map(method, *args, **kwargs, inplace=False, ragged=True)
        peaks.map(peaks_as_gvectors,
                  center=np.array(self.axes_manager.signal_shape)/2,
                  calibration=self.axes_manager.signal_axes[0].scale)
        peaks = DiffractionVectors(peaks)
        peaks.axes_manager.set_signal_dimension(0)
        if peaks.axes_manager.navigation_dimension != self.axes_manager.navigation_dimension:
            #ToDo Remove this hardcore
            peaks = peaks.transpose(navigation_axes=2)
        if peaks.axes_manager.navigation_dimension != self.axes_manager.navigation_dimension:
            raise RuntimeWarning('You do not have the same size navigation axes \
            for your Diffraction pattern and your peaks')
        return peaks

    def find_peaks_interactive(self, imshow_kwargs={}):
        """Find peaks using an interactive tool.

        Requires `ipywidgets` and `traitlets` to be installed.

        """
        peakfinder = peakfinder2D_gui.PeakFinderUIIPYW(imshow_kwargs=imshow_kwargs)
        peakfinder.interactive(self)

    def enhance(self,sigma_blur=1.6, sigma_enhance=0.5,
                threshold=6.5, k=0.01, window_size=11,
                *args, **kwargs):
        """Enhances peaks in the diffraction patterns.

        A gaussian filter is applied and the blurred image subtracted from the
        original, thresholding removes low intensities, local Sauvola
        thresholding creates a mask of peaks, final Gaussian blurring.

        Parameters:
        ------------
        sigma_blur : float
            Sigma of the gaussian filter used for initial blur.

        sigma_enhance : float
            Sigma of the gaussian filter used for the ehnancement of the peaks

        threshold : float
            Value of the small threshold for removal of low intensity fake peaks

        k : float
            Parameter for Sauvola thresholding

        window_size : int
            Size of the window considered for each pixel when calculating local
            Sauvola threshold. Has to be odd and >=3.

        See Also
        --------
        http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_niblack_sauvola.html#id2
        J. Sauvola and M. Pietikainen, “Adaptive document image binarization,”
        Pattern Recognition 33(2), pp. 225-236, 2000.
        DOI:10.1016/S0031-3203(99)00055-2
        """
        return self.map(enhance_gauss_sauvola,
                        sigma_blur=sigma_blur,
                        sigma_enhance=sigma_enhance,
                        threshold=threshold,
                        window_size=window_size,
                        k=k, *args, **kwargs)


class LazyElectronDiffraction(LazySignal, ElectronDiffraction):

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
