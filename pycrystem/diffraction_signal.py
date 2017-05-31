# -*- coding: utf-8 -*-
# Copyright 2016 The PyCrystEM developers
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
from hyperspy.api import interactive
from hyperspy.api import roi
from hyperspy.components1d import Voigt, Exponential, Polynomial
from hyperspy.signals import Signal2D, Signal1D, BaseSignal
from hyperspy._lazy_signals import LazySignal2D
from scipy.ndimage import variance

from scipy.ndimage import variance
from pycrystem.utils.expt_utils import *
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
        """Set the experimental parameters.

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

    def set_calibration(self, calibration, offset=None):
        """Set pixel size in reciprocal Angstroms and origin location.

        Parameters
        ----------
        calibration: float
            Calibration in reciprocal Angstroms per pixel
        offset: tuple
            Position of the central beam, in pixels
        """
        # TODO: extend to get calibration from a list of stored calibrations for
        # the camera length recorded in metadata.
        if offset==None:
            offset = np.array(self.axes_manager.signal_shape)/2 * calibration

        dx = self.axes_manager.signal_axes[0]
        dy = self.axes_manager.signal_axes[1]

        dx.name = 'dx'
        dx.scale = calibration
        dx.offset = -offset[0]
        dx.units = '$A^{-1}$'

        dy.name = 'dy'
        dy.scale = calibration
        dy.offset = -offset[1]
        dy.units = '$A^{-1}$'

    def plot_interactive_virtual_image(self, roi):
        """Plots an interactive virtual image formed with a specified but
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
            BaseSignal(np.zeros(self.axes_manager.navigation_shape))
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
        return dark_field_sum

    # TODO: this appears to be broken in HyperSpy
    # def plot_line_profile(self, x1, y1, x2, y2, width):
    #     """Plots an interactive line profile."""
    #     self.plot()
    #     lin = roi.Line2DROI(x1=x1, y1=y1, x2=x2, y2=y2, linewidth=width)
    #     lin.add_widget(self, axes=self.axes_manager.signal_axes, color='red')
    #     lin.interactive(self, navigation_signal='same').plot()

    def get_variance_image(self, roi):
        """Form a variance image for a specified region of interest in the
        diffraction signal.

        The variance image plots the variance of values within a specified
        set of pixels in the diffraction signal as a function of probe position.

        Parameters
        ----------
        roi: :obj:`hyperspy.roi.BaseInteractiveROI`
            Any interactive ROI detailed in HyperSpy.

        Returns
        -------
        :obj:`hyperspy.signals.Signal2D`
            The variance image as a HyperSpy Signal2D.
        """
        # Crop the data using the roi
        annulus = roi(self, axes=self.axes_manager.signal_axes)
        annulus.change_dtype('float')
        # Create an empty array to contain the image.
        arr_shape = (annulus.axes_manager._navigation_shape_in_array
                     if annulus.axes_manager.navigation_size > 0
                     else [1, ])
        var_image = np.zeros(arr_shape)
        # Calculate the variance within the roi for each DP.
        for i in annulus.axes_manager:
            it = (i[1], i[0])
            var_image[it] = variance(annulus.data[it]) / (np.mean(annulus.data[it]) * np.mean(annulus.data[it]))

        return Signal2D(var_image)

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

    def get_direct_beam_position(self, radius):
        """
        Determine rigid shifts in the SED patterns based on the position of the
        direct beam and return the shifts required to center all patterns.

        Parameters
        ----------
        radius : int
            Defines the size of the circular region within which the direct beam
            position is refined.

        subpixel : bool
            If True the direct beam position is refined to sub-pixel precision
            via calculation of the intensity center_of_mass.

        Returns
        -------
        shifts : array
            Array containing the shift to be applied to each SED pattern to
            center it.

        See also
        --------
        _get_direct_beam_position

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

    def get_direct_beam_shifts(self, centers=None, radius=None):
        """Determine rigid shifts in the SED patterns based on the position of
        the direct beam and return the shifts required to center all patterns.

        Parameters
        ----------
        centers : array, None
            Array of dimension (navigation_size, 2) containing the position of
            the diffraction pattern
            If None, an array containing center positions is obtained using the
            `get_direct_beam_position` method.
        radius : int
            Defines the size of the circular region within which the direct beam
            position is refined.
        subpixel : bool
            If True the direct beam position is refined to sub-pixel precision
            via calculation of the intensity center_of_mass.

        Returns
        -------
        shifts : array
            Array containing the shift to be applied to each SED pattern to
            center it.

        See also
        --------
        get_direct_beam_position
        """
        if centers == None:
            centers = self.get_direct_beam_position(radius=radius)
        if centers != None:
            if centers.shape != (self.axes_manager.navigation_size, 2):
                raise ValueError("The number of center positions provided "
                                 "must match the navigation_size")

        # calculate shifts to align all patterns to the reference position
        shifts = centers - [(self.axes_manager.signal_shape[0] - 1) / 2,
                            (self.axes_manager.signal_shape[1] - 1) / 2]

        return shifts

    def correct_geometric_distortion(self, D):
        """Correct geometric distortion by applying an affine transformation.

        Parameters
        ----------


        Returns
        -------


        """
        #TODO:Add automatic method based on power spectrum optimisation as
        #presented in Vigouroux et al...
        self.map(affine_transformation, matrix=D, ragged=True)

    def rotate_patterns(self, angle):
        """Rotate the diffraction patterns in a clockwise direction.

        Parameters
        ----------

        Returns
        -------

        """
        #TODO: Preserve knowledge of basis - and eventually remove when the
        # version in devlopment for HyperSpy is completed.
        a = angle * np.pi/180.0
        t = np.array([[math.cos(a), math.sin(a), 0.],
                      [-math.sin(a), math.cos(a), 0.],
                      [0., 0., 1.]])

        self.map(affine_transformation, matrix=t)

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
        # TODO: make this work without averaging the centers
        if centers is None:
            centers = self.get_direct_beam_position(radius=10)
        center = centers.mean(axis=(0, 1))
        radial_profiles = self.map(radial_average, center=center, inplace=False)
        radial_profiles.axes_manager.signal_axes[0].offset = 0
        signal_axis = radial_profiles.axes_manager.signal_axes[0]
        return radial_profiles.as_signal1D(signal_axis)

    def remove_background(self, saturation_radius=0):
        """Perform background subtraction.

        Parameters
        ----------
        saturation_radius: int, optional
            The radius, in pixels, of the saturated data (if any) in the direct
            beam.

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
        model = profile.create_model()
        e1 = saturation_radius * profile.axes_manager.signal_axes[0].scale
        model.set_signal_range(e1)

        direct_beam = Voigt()
        direct_beam.centre.value = 0
        direct_beam.centre.free = False
        direct_beam.area.value = 40
        model.append(direct_beam)

        diffuse_scatter = Exponential()
        diffuse_scatter.tau.value = 0.5
        model.append(diffuse_scatter)

        linear_decay = Polynomial(1)
        model.append(linear_decay)

        model.fit()

        x_axis = self.axes_manager.signal_axes[0].axis
        y_axis = self.axes_manager.signal_axes[1].axis

        xs, ys = np.meshgrid(x_axis, y_axis)
        rs = (xs ** 2 + ys ** 2) ** 0.5

        bg = ElectronDiffraction(
            diffuse_scatter.function(rs) + linear_decay.function(rs))

        for i in (0, 1):
            bg.axes_manager.signal_axes[i].update_from(
                self.axes_manager.signal_axes[i])

        bg_removed = np.clip(self - bg, 0, 255)

        denoised = ElectronDiffraction(
            bg_removed.map(regional_flattener, h=bg.data.min(), inplace=False)
        )
        denoised.axes_manager.update_axes_attributes_from(
            self.axes_manager.navigation_axes)
        denoised.axes_manager.update_axes_attributes_from(
            self.axes_manager.signal_axes)

        return denoised

    def get_data_movie_frames(self, image, indices, save_path):
        """

        Parameters
        ----------

        Returns
        -------
        """
        import matplotlib.pyplot as plt
        j=0
        for i in indices:
            j=j+1
            fig, axes = plt.subplots(ncols=2)
            ax1, ax2 = axes.ravel()

            ax1.imshow(image)
            ax2.imshow(self.inav[i].data)

            ax1.set_title('Scan Position')
            ax2.set_title('Diffraction Pattern')

            ax1.plot(i[0],i[1],'.')

            ax1.set_xlim(0, image.shape[1])
            ax1.set_ylim(image.shape[0],0)

            ax1.set_axis_off()
            ax2.set_axis_off()

            plt.xlabel('')
            plt.ylabel('')
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,
                                wspace=0.02)

            plt.savefig(save_path +'{}.png'.format(j))
            plt.close()

    def get_gvector_magnitudes(self, peaks):
        """Obtain the magnitude of g-vectors in calibrated units
        from a structured array containing peaks in array units.

        Parameters
        ----------

        Returns
        -------

        """
        # Allocate an empty structured array in which to store the gvector
        # magnitudes.
        arr_shape = (self.axes_manager._navigation_shape_in_array
                     if self.axes_manager.navigation_size > 0
                     else [1, ])
        gvectors = np.zeros(arr_shape, dtype=object)
        #
        for i in self.axes_manager:
            it = (i[1], i[0])
            res = []
            centered = peaks[it] - [256,256]
            for j in np.arange(len(centered)):
                res.append(np.linalg.norm(centered[j]))

            cent = peaks[it][np.where(res == min(res))[0]][0]
            vectors = peaks[it] - cent

            mags = []
            for k in np.arange(len(vectors)):
                mags.append(np.linalg.norm(vectors[k]))
            maga = np.asarray(mags)
            gvectors[it] = maga * self.axes_manager.signal_axes[0].scale

        return gvectors

    def get_reflection_intensities(self, indexed_reflections):
        """

        Parameters
        ----------

        Returns
        -------
        """
        # TODO: For each peak in a structured array of peaks obtain the
        # intensity of the reflection associated with it. Multiple methods
        # should be implemented including just summing or fitting a Gaussian.
        pass

    def get_gvector_indexation(self, glengths, calc_peaks, threshold):
        """Index the magnitude of g-vectors in calibrated units
        from a structured array containing gvector magnitudes.

        Parameters
        ----------

        glengths : A structured array containing the

        calc_peaks : A structured array

        threshold : Float indicating the maximum allowed deviation from the
            theoretical value.

        Returns
        -------

        gindex : Structured array containing possible indexation results
            consistent with the data.

        """
        # TODO: Make it so that the threshold can be specified as a fraction of
        # the g-vector magnitude.
        arr_shape = (self.axes_manager._navigation_shape_in_array
                     if self.axes_manager.navigation_size > 0
                     else [1, ])
        gindex = np.zeros(arr_shape, dtype=object)

        for i in self.axes_manager:
            it = (i[1], i[0])
            res = []
            for j in np.arange(len(glengths[it])):
                peak_diff = (calc_peaks.T[1] - glengths[it][j]) * (calc_peaks.T[1] - glengths[it][j])
                res.append((calc_peaks[np.where(peak_diff < threshold)],
                            peak_diff[np.where(peak_diff < threshold)]))
            gindex[it] = res

        return gindex

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

