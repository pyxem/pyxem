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

from __future__ import division

import tqdm
from hyperspy.api import interactive, roi, stack
from hyperspy.components1d import Voigt, Exponential, Polynomial
from hyperspy.signals import Signal2D, Signal1D, BaseSignal

from pycrystem.utils.expt_utils import *
from pycrystem.utils.peakfinders2D import *
from .library_generator import DiffractionLibrary
from .indexation_generator import IndexationGenerator

"""
Signal class for Electron Diffraction Data
"""


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
                        precession_angle)
        if rocking_frequency is not None:
            md.set_item("Acquisition_instrument.TEM.rocking_frequency",
                        precession_frequency)
        if camera_length is not None:
            md.set_item("Acquisition_instrument.TEM.Detector.Diffraction.camera_length",
                        camera_length)
        if exposure_time is not None:
            md.set_item("Acquisition_instrument.TEM.Detector.Diffraction.exposure_time",
                        exposure_time)

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
        if center==None:
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
        """Obtains a virtual image associated with a specified roi.

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
        dark_field_sum = dark_field.sum(axis=dark_field.axes_manager.signal_axes)
        dark_field_sum.metadata.General.title = "Virtual Dark Field"
        #TODO:make outputs neat in obvious cases i.e. 2D for normal vdf
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
        mask : array
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
        mask : signal
            The mask of the region of interest. Vacuum regions to be masked are
            set True.

        See also
        --------
        get_direct_beam_mask
        """
        db = np.invert(self.get_direct_beam_mask(radius=radius, center=center))
        diff_only = self * db
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

    def apply_affine_transformation(self, D):
        """Correct geometric distortion by applying an affine transformation.

        Parameters
        ----------
        D : 3x3 numpy array
            Specifies the affine transform to be applied.

        """
        #TODO:Make output optional so may or may not overwrite.
        self.map(affine_transformation, matrix=D, ragged=True)

    def gain_normalisation(self, dref, bref):
        """Apply gain normalization to experimentally acquired electron
        diffraction patterns.

        Parameters
        ----------
        dref : ElectronDiffraction
            Dark reference image.

        bref : ElectronDiffraction
            Bright reference image.
        """
        #TODO:Make output optional so may or may not overwrite.
        self.map(gain_normalise, dref=dref, bref=bref)

    def get_radial_profile(self, centers=None):
        """Return the radial profile of the diffraction pattern.

        Parameters
        ----------
        centers : array
            Array of dimensions (navigation_shape, 2) containing the
            origin for the radial integration in each diffraction
            pattern.

        Returns
        -------
        radial_profile: :obj:`hyperspy.signals.Signal1D`
            The radial average profile of each diffraction pattern
            in the ElectronDiffraction signal as a Signal1D.

        See also
        --------
        radial_average
        get_direct_beam_position

        """
        #TODO: make this work without averaging the centers
        #TODO: fix for case when data is singleton
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

        vardps : Signal2D
              A two dimensional Signal class object containing the mean DP,
              mean squared DP, and variance DP.
        """
        mean_dp = dp.mean((0,1))
        meansq_dp = Signal2D(np.square(dp.data)).mean((0,1))
        var_dp = Signal2D(((meansq_dp.data / np.square(mean_dp.data)) - 1.))
        return stack((mean_dp, meansq_dp, var_dp))

    def remove_background(self, saturation_radius=0):
        """Perform background subtraction.

        Parameters
        ----------
        saturation_radius: int, optional
            The radius, in pixels, of the saturated data (if any) in the direct
            beam.
        method : String
            'h-dome', 'profile_fit'

        Returns
        -------
        denoised: `ElectronDiffraction`
            A copy of the data with the background removed and the noise
            smoothed.

        """
        # TODO: separate into multiple methods
        # TODO: make an 'averaging' flag

        # Old method:
        # def mean_filter(h):
        #     self.data = self.data / self.data.max()
        #     self.map(regional_filter, h=h)
        #     self.map(filters.rank.mean, selem=square(3))
        #     self.data = self.data / self.data.max()

        profile = self.get_radial_profile().mean()
        bg = self.get_background_model(profile, saturation_radius)

        bg_removed = np.clip(self - bg, 0, 255)

        denoised = ElectronDiffraction(
            bg_removed.map(regional_flattener, h=bg.data.min()-1, inplace=False)
        )
        denoised.axes_manager.update_axes_attributes_from(
            self.axes_manager.navigation_axes)
        denoised.axes_manager.update_axes_attributes_from(
            self.axes_manager.signal_axes)

        return denoised

    def get_background_model(self, profile, saturation_radius):
        """Create a background model from a radial profile.

        The current model contains a Voigt model to fit the direct beam and an
        Exponential combined with a Linear profile to model the background. The
        exponential and linear model are used to create a background "image".
        This might be used for background subtraction.

        Parameters
        ----------
        profile : :obj:`hyperspy.signals.Signal1D`
            A 1-d signal describing the radial profile of the data.
        saturation_radius : int
            The approximate radius of the region of saturation of the direct
            beam.

        Returns
        -------
        background : :obj:`ElectronDiffraction`
            A 2-d signal estimating the background of the signal.

        """
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
        background = ElectronDiffraction(
            diffuse_scatter.function(rs) + linear_decay.function(rs))
        for i in (0, 1):
            background.axes_manager.signal_axes[i].update_from(
                self.axes_manager.signal_axes[i])
        return background

    def decomposition(self,
                      normalize_poissonian_noise=True,
                      signal_mask=None,
                      center=None,
                      navigation_mask=None,
                      threshold=None,
                      closing=True,
                      *args,
                      **kwargs):
        """Decomposition with a choice of algorithms.

        The results are stored in self.learning_results

        Parameters
        ----------
        normalize_poissonian_noise : bool
            If True, scale the SI to normalize Poissonian noise
        direct_beam_mask : None or float or boolean numpy array
            The navigation locations marked as True are not used in the
            decompostion. If float is given the direct_beam_mask method is used
            to generate a mask with the float value as radius.
        closing: bool
            If true, applied a morphologic closing to the maks obtained by
            vacuum_mask.
        algorithm : 'svd' | 'fast_svd' | 'mlpca' | 'fast_mlpca' | 'nmf' |
            'sparse_pca' | 'mini_batch_sparse_pca'
        output_dimension : None or int
            number of components to keep/calculate
        centre : None | 'variables' | 'trials'
            If None no centring is applied. If 'variable' the centring will be
            performed in the variable axis. If 'trials', the centring will be
            performed in the 'trials' axis. It only has effect when using the
            svd or fast_svd algorithms
        auto_transpose : bool
            If True, automatically transposes the data to boost performance.
            Only has effect when using the svd of fast_svd algorithms.
        signal_mask : boolean numpy array
            The signal locations marked as True are not used in the
            decomposition.
        var_array : numpy array
            Array of variance for the maximum likelihood PCA algorithm
        var_func : function or numpy array
            If function, it will apply it to the dataset to obtain the
            var_array. Alternatively, it can a an array with the coefficients
            of a polynomial.
        polyfit :
        reproject : None | signal | navigation | both
            If not None, the results of the decomposition will be projected in
            the selected masked area.

        Examples
        --------
        >>> dp = hs.datasets.example_signals.electron_diffraction()
        >>> dps = hs.stack([dp]*3)
        >>> dps.change_dtype(float)
        >>> dps.decomposition()

        See also
        --------
        get_direct_beam_mask
        get_vacuum_mask
        """
        if isinstance(signal_mask, float):
            signal_mask = self.direct_beam_mask(signal_mask, center)
        if isinstance(navigation_mask, float):
            navigation_mask = self.vacuum_mask(navigation_mask,
                                               center, threshold).data
        super(Signal2D, self).decomposition(
            normalize_poissonian_noise=normalize_poissonian_noise,
            signal_mask=signal_mask, navigation_mask=navigation_mask,
            *args, **kwargs)
        self.learning_results.loadings = np.nan_to_num(
            self.learning_results.loadings)

    def find_peaks2D(self, method='skimage', *args, **kwargs):
        """Find peaks in a 2D signal/image.
        Function to locate the positive peaks in an image using various, user
        specified, methods. Returns a structured array containing the peak
        positions.
        Parameters
        ---------
        method : str
                 Select peak finding algorithm to implement. Available methods
                 are:
                     'max' - simple local maximum search
                     'skimage' - call the peak finder implemented in
                                 scikit-image which uses a maximum filter
                     'minmax' - finds peaks by comparing maximum filter results
                                with minimum filter, calculates centers of mass
                     'zaefferer' - based on gradient thresholding and refinement
                                   by local region of interest optimisation
                     'stat' - statistical approach requiring no free params.
                     'massiel' - finds peaks in each direction and compares the
                                 positions where these coincide.
                     'laplacian_of_gaussians' - a blob finder implemented in
                                                `scikit-image` which uses the
                                                laplacian of Gaussian matrices
                                                approach.
                     'difference_of_gaussians' - a blob finder implemented in
                                                 `scikit-image` which uses
                                                 the difference of Gaussian
                                                 matrices approach.
        *args : associated with above methods
        **kwargs : associated with above methods.
        Returns
        -------
        peaks: structured array of shape _navigation_shape_in_array in which
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
        #TODO: make return DiffractionVectors(peaks)
        return peaks

    def find_peaks2D_interactive(self):
        from pycrystem.utils import peakfinder2D_gui
        """
        Find peaks using an interactive tool.

        Notes
        -----
        Requires `ipywidgets` and `traitlets` to be installed.

        """
        peakfinder = peakfinder2D_gui.PeakFinderUIIPYW()
        peakfinder.interactive(self)
