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
"""Signal class for two-dimensional X-ray diffraction data.
"""

import numpy as np
from hyperspy.signals import BaseSignal
from hyperspy._signals.lazy import LazySignal

from pyxem.signals.diffraction2d import Diffraction2D


class XrayDiffraction2D(Diffraction2D):
    _signal_type = "xray_diffraction"

    def set_experimental_parameters(
        self, beam_energy=None, camera_length=None, exposure_time=None
    ):
        """Set experimental parameters in metadata.

        Parameters
        ----------
        beam_energy : float
            Beam energy in kV
        camera_length: float
            Camera length in cm
        exposure_time : float
            Exposure time in ms.
        """
        md = self.metadata

        if beam_energy is not None:
            md.set_item("Acquisition_instrument.I14.beam_energy", beam_energy)
        if camera_length is not None:
            md.set_item(
                "Acquisition_instrument.I14.Detector.Diffraction.camera_length",
                camera_length,
            )
        if exposure_time is not None:
            md.set_item(
                "Acquisition_instrument.I14.Detector.Diffraction.exposure_time",
                exposure_time,
            )

    def set_scan_calibration(self, calibration):
        """Set scan pixel size in nanometres.

        Parameters
        ----------
        calibration: float
            Scan calibration in nanometres per pixel.
        """
        x = self.axes_manager.navigation_axes[0]
        y = self.axes_manager.navigation_axes[1]

        x.name = "x"
        x.scale = calibration
        x.units = "nm"

        y.name = "y"
        y.scale = calibration
        y.units = "nm"


class LazyXrayDiffraction2D(LazySignal, XrayDiffraction2D):

    pass
