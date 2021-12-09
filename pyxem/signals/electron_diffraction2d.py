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
"""
Signal class for two-dimensional electron diffraction data in Cartesian
coordinates.
"""

import numpy as np

from pyxem.signals.diffraction2d import Diffraction2D, LazyDiffraction2D
from diffsims.utils.sim_utils import get_electron_wavelength


class ElectronDiffraction2D(Diffraction2D):
    _signal_type = "electron_diffraction"

    def __init__(self, *args, **kwargs):
        """
        Create an ElectronDiffraction2D object from numpy.ndarray.

        Parameters
        ----------
        *args :
            Passed to the __init__ of Diffraction2D. The first arg should be
            numpy.ndarray
        **kwargs :
            Passed to the __init__ of Diffraction2D
        """
        super().__init__(*args, **kwargs)

        # Set default attributes
        if "Acquisition_instrument" in self.metadata.as_dictionary():
            if "SEM" in self.metadata.as_dictionary()["Acquisition_instrument"]:
                self.metadata.set_item(
                    "Acquisition_instrument.TEM",
                    self.metadata.Acquisition_instrument.SEM,
                )
                del self.metadata.Acquisition_instrument.SEM

    def set_ai(
        self, center=None, energy=None, affine=None, radial_range=None, **kwargs
    ):
        if energy is None and self.beam_energy is not None:
            energy = self.beam_energy
        if energy is not None:
            wavelength = get_electron_wavelength(energy) * 1e-10
        else:
            wavelength = None
        ai = super().set_ai(
            center=center,
            wavelength=wavelength,
            affine=affine,
            radial_range=radial_range,
            **kwargs
        )
        return ai

    @property
    def beam_energy(self):
        try:
            return self.metadata.Acquisition_instrument.TEM["beam_energy"]
        except (AttributeError):
            return None

    @beam_energy.setter
    def beam_energy(self, energy):
        self.metadata.set_item("Acquisition_instrument.TEM.beam_energy", energy)

    @property
    def camera_length(self):
        try:
            return self.metadata.Acquisition_instrument.TEM["camera_length"]
        except (AttributeError):
            return None

    @camera_length.setter
    def camera_length(self, length):
        self.metadata.set_item("Acquisition_instrument.TEM.camera_length", length)

    @property
    def diffraction_calibration(self):
        return self.axes_manager.signal_axes[0].scale

    @diffraction_calibration.setter
    def diffraction_calibration(self, calibration):
        self.axes_manager.signal_axes[0].scale = calibration
        self.axes_manager.signal_axes[1].scale = calibration

    @property
    def scan_calibration(self):
        return self.axes_manager.navigation_axes[0].scale

    @scan_calibration.setter
    def scan_calibration(self, calibration):
        self.axes_manager.navigation_axes[0].scale = calibration
        self.axes_manager.navigation_axes[1].scale = calibration

    def set_experimental_parameters(
        self,
        beam_energy=None,
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
        beam_energy : float
            Beam energy in keV
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

        if beam_energy is not None:
            md.set_item("Acquisition_instrument.TEM.beam_energy", beam_energy)
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

        dx.name = "kx"
        dx.scale = calibration
        dx.offset = -center[0]
        dx.units = "$A^{-1}$"

        dy.name = "ky"
        dy.scale = calibration
        dy.offset = -center[1]
        dy.units = "$A^{-1}$"

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


class LazyElectronDiffraction2D(LazyDiffraction2D, ElectronDiffraction2D):

    pass
