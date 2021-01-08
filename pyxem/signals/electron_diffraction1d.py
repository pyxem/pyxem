# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
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
"""Signal class for Electron Diffraction radial profiles."""
from hyperspy._signals.lazy import LazySignal

from pyxem.signals.diffraction1d import Diffraction1D


class ElectronDiffraction1D(Diffraction1D):
    _signal_type = "electron_diffraction"

    def set_experimental_parameters(
        self,
        accelerating_voltage=None,
        camera_length=None,
        scan_rotation=None,
        convergence_angle=None,
        rocking_angle=None,
        rocking_frequency=None,
        exposure_time=None,
    ):
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
            md.set_item(
                "Acquisition_instrument.TEM.accelerating_voltage", accelerating_voltage
            )
        if camera_length is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.Diffraction.camera_length",
                camera_length,
            )
        if scan_rotation is not None:
            md.set_item("Acquisition_instrument.TEM.scan_rotation", scan_rotation)
        if convergence_angle is not None:
            md.set_item(
                "Acquisition_instrument.TEM.convergence_angle", convergence_angle
            )
        if rocking_angle is not None:
            md.set_item("Acquisition_instrument.TEM.rocking_angle", rocking_angle)
        if rocking_frequency is not None:
            md.set_item(
                "Acquisition_instrument.TEM.rocking_frequency", rocking_frequency
            )
        if exposure_time is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.Diffraction.exposure_time",
                exposure_time,
            )

    def set_diffraction_calibration(self, calibration):
        """Set diffraction profile channel size in reciprocal Angstroms.

        Parameters
        ----------
        calibration : float
            Diffraction profile calibration in reciprocal Angstroms per pixel.
        """
        dx = self.axes_manager.signal_axes[0]

        dx.name = "k"
        dx.scale = calibration
        dx.units = "$A^{-1}$"

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


class LazyElectronDiffraction1D(LazySignal, ElectronDiffraction1D):

    pass
