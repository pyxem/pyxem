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
"""Signal class for Electron Diffraction radial profiles

"""

import numpy as np

from hyperspy.api import interactive
from hyperspy.signals import Signal1D, BaseSignal
from hyperspy.roi import SpanROI

from pyxem.signals.diffraction_peaks import DiffractionPeaks
from pyxem.utils.peak_utils import mapping_indeces_dictionary


class ElectronDiffractionProfile(Signal1D):
    _signal_type = "diffraction_profile"

    def __init__(self, *args, **kwargs):
        Signal1D.__init__(self, *args, **kwargs)

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

    def set_diffraction_calibration(self, calibration):
        """Set diffraction profile channel size in reciprocal Angstroms.

        Parameters
        ----------
        calibration : float
            Diffraction profile calibration in reciprocal Angstroms per pixel.
        """
        dx = self.axes_manager.signal_axes[0]

        dx.name = 'k'
        dx.scale = calibration
        dx.units = '$A^{-1}$'

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

    def plot_interactive_virtual_image(self, left, right, **kwargs):
        """Plots an interactive virtual image formed by integrating scatterered
        intensity over a specified range.

        Parameters
        ----------
        left : float
            Lower bound of the data range to be plotted.
        right : float
            Upper bound of the data range to be plotted.
        **kwargs:
            Keyword arguments to be passed to `ElectronDiffractionProfile.plot`

        Examples
        --------
        .. code-block:: python

            rp.plot_interactive_virtual_image(left=0.5, right=0.7)

        """
        # Define ROI
        roi = SpanROI(left=left, right=right)
        # Plot signal
        self.plot(**kwargs)
        # Add the ROI to the appropriate signal axes.
        roi.add_widget(self, axes=self.axes_manager.signal_axes)
        # Create an output signal for the virtual dark-field calculation.
        dark_field = roi.interactive(self, navigation_signal='same')
        dark_field_placeholder = \
            BaseSignal(np.zeros(self.axes_manager.navigation_shape[::-1]))
        # Create an interactive signal
        dark_field_sum = interactive(
            # Formed from the sum of the pixels in the dark-field signal
            dark_field.sum,
            # That updates whenever the widget is moved
            event=dark_field.axes_manager.events.any_axis_changed,
            axis=dark_field.axes_manager.signal_axes,
            # And outputs into the prepared placeholder.
            out=dark_field_placeholder,
        )
        # Set the parameters
        dark_field_sum.axes_manager.update_axes_attributes_from(
            self.axes_manager.navigation_axes,
            ['scale', 'offset', 'units', 'name'])
        dark_field_sum.metadata.General.title = "Virtual Dark Field"
        # Plot the result
        dark_field_sum.plot()

    def get_virtual_image(self, left, right):
        """Obtains a virtual image associated with a specified scattering range.

        Parameters
        ----------
        left : float
            Lower bound of the data range to be plotted.
        right : float
            Upper bound of the data range to be plotted.

        Returns
        -------
        dark_field_sum : :obj:`hyperspy.signals.BaseSignal`
            The virtual image signal associated with the specified scattering
            range.

        Examples
        --------
        .. code-block:: python

            rp.get_virtual_image(left=0.5, right=0.7)

        """
        # Define ROI
        roi = SpanROI(left=left, right=right)
        dark_field = roi(self, axes=self.axes_manager.signal_axes)
        dark_field_sum = dark_field.sum(
            axis=dark_field.axes_manager.signal_axes
        )
        dark_field_sum.metadata.General.title = "Virtual Dark Field"
        vdfim = dark_field_sum.as_signal2D((0, 1))

        return vdfim

    def detector_px_to_1D_kspace(self, wavelength, det2sample_len, px_size):
        """Converts the detector 1D coordinates in px, to the respective 1D coordinates in the kspace, in Angstom^-1, using purely geometrical arguments. 
        Only use for DiffractionProfile class. 2D version of the detector_to_3D_kspace from the class DiffractionVectors.
        Args:
        -------
            wavelength (float):
                Wavelength of the scanning beam, in Amstrong.
            det2sample_len (float): 
                Distance from detector to sample, in Amstrong. IMPORTANT: Distance obtained from the calibration file.
            px_size (float):
                Length of the pixel in the detector, in micrometres
        Returns:
        -------
            self:
                DiffractionProfile has been modified from px cordinates to angstrom^-1
        """
        #Extract the pixel-coordinates as an array
        px = np.array(self.axes_manager['k'].axis)
        #Convert each pixel to the actual disctance in Angstrom
        x = px*1e4*px_size
        # Get the two_theta angle (in 1D Edwald circunference it is arctan(px/d)) for each x:
        two_theta = np.arctan2(x,det2sample_len)
        # #Convert each x to the respective gx value:
        # gx = (1/wavelength)*np.sin(two_theta)
        # #Convert each x to the respective gy value:
        # gy = (1/wavelength)*(1-np.cos(two_theta))
        # #Get the diffraction vector g magnitude:
        # g = np.sqrt(gx**2 + gy**2)

        g = 2 * (1/wavelength) * np.sin(two_theta/2)

        # Replace pixel coordinates in the ElectronDiffractionProfile to the kx values:
        self.axes_manager['k'].axis = g
        self.axes_manager['k'].name = '$\mid g \mid$'

    def find_peaks_1D(self, *args, **kwargs):
        """Find peaks in a DiffractionProfile using the O-Haver function (Hyperspy). It returns the DiffractionPeaks class.
        Parameters
        ----------
            slope threshold : float 
                Higher values will neglect broader features
            amp/intensity threshold: float
                Intensity below which peaks are ignored
            *args
                Inherited parameters from Hyperspy find_peaks1D_ohaver
        Returns
        ----------
            peaks: DiffractionPeaks
                A DiffractionPeaks object with navigation dimensions identical to the original ElectronDiffraction object.
                Each datapoint is an array of the position/magnitude of the diffraction peaks. The intensity of each peak is stored in the "intensity" attribute, with navigation dimensions identical to the original object.
        """
        #Find peaks using the o'Haver function from Hyperspy. It returns a BaseSignal object, in which each data point is a dictionary containing "position" and "height" of each peak.
        peaks = self.find_peaks1D_ohaver(*args, **kwargs)
        #Create a DiffractionPeaks object.
        peaks = DiffractionPeaks(peaks)
        peaks.axes_manager.set_signal_dimension(0)

        #Extract the intensity and store it as an attribute.
        peaks.intensity = peaks.map(mapping_indeces_dictionary, key='height', inplace=False)
        #Extract the peak position. Replace each data point (a dictionary) for a peak position/magnitude array.
        peaks.map(mapping_indeces_dictionary, key='position', inplace=True)

        # For diffraction profiles with navigation axes, transfer them to the DiffractionPeaks object:
        if self.axes_manager.navigation_axes != ():
            x = peaks.axes_manager.navigation_axes[0]
            y = peaks.axes_manager.navigation_axes[1]

            x.name = 'x'
            x.scale = self.axes_manager.navigation_axes[0].scale
            x.units = 'nm'

            y.name = 'y'
            y.scale = self.axes_manager.navigation_axes[1].scale
            y.units = 'nm'
        
        return peaks
