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
