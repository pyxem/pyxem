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
"""Signal class for two-dimensional electron diffraction data.
"""

import numpy as np
from hyperspy.signals import BaseSignal
from hyperspy._signals.lazy import LazySignal

from pyxem.signals import push_metadata_through
from pyxem.signals.diffraction2d import Diffraction2D


class XrayDiffraction2D(Diffraction2D):
    _signal_type = "xray_diffraction2d"

    def __init__(self, *args, **kwargs):
        """
        Create an XrayDiffraction2D object from a hs.Signal2D or np.array.

        Parameters
        ----------
        *args :
            Passed to the __init__ of Diffraction2D. The first arg should be
            either a numpy.ndarray or a Signal2D
        **kwargs :
            Passed to the __init__ of Diffraction2D
        """
        self, args, kwargs = push_metadata_through(self, *args, **kwargs)
        super().__init__(*args, **kwargs)

        # Set default attributes
        if 'Acquisition_instrument' in self.metadata.as_dictionary():
            if 'SEM' in self.metadata.as_dictionary()['Acquisition_instrument']:
                self.metadata.set_item(
                    "Acquisition_instrument.I14",
                    self.metadata.Acquisition_instrument.SEM)
                del self.metadata.Acquisition_instrument.SEM
            if 'REM' in self.metadata.as_dictionary()['Acquisition_instrument']:
                self.metadata.set_item(
                    "Acquisition_instrument.I14",
                    self.metadata.Acquisition_instrument.TEM)
                del self.metadata.Acquisition_instrument.TEM
        self.decomposition.__func__.__doc__ = BaseSignal.decomposition.__doc__

    def set_experimental_parameters(self,
                                    beam_energy=None,
                                    exposure_time=None):
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
            md.set_item("Acquisition_instrument.I14.beam_energy",
                        accelerating_voltage)
        if camera_length is not None:
            md.set_item(
                "Acquisition_instrument.I14.Detector.Diffraction.camera_length",
                camera_length)
        if exposure_time is not None:
            md.set_item(
                "Acquisition_instrument.I14.Detector.Diffraction.exposure_time",
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
        pass

    def set_scan_calibration(self, calibration):
        """Set scan pixel size in nanometres.

        Parameters
        ----------
        calibration: float
            Scan calibration in nanometres per pixel.
        """
        pass

    def as_lazy(self, *args, **kwargs):
        """Create a copy of the XrayDiffraction2D object as a
        :py:class:`~pyxem.signals.xray_diffraction2d.LazyXrayDiffraction2D`.

        Parameters
        ----------
        copy_variance : bool
            If True variance from the original XrayDiffraction2D object is
            copied to the new LazyXrayDiffraction2D object.

        Returns
        -------
        res : :py:class:`~pyxem.signals.xray_diffraction2d.LazyXrayDiffraction2D`.
            The lazy signal.
        """
        res = super().as_lazy(*args, **kwargs)
        res.__class__ = LazyXrayDiffraction2D
        res.__init__(**res._to_dictionary())
        return res

    def decomposition(self, *args, **kwargs):
        super().decomposition(*args, **kwargs)
        self.__class__ = XrayDiffraction2D


class LazyXrayDiffraction2D(LazySignal, XrayDiffraction2D):

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, *args, **kwargs):
        super().compute(*args, **kwargs)
        self.__class__ = XrayDiffraction2D
        self.__init__(**self._to_dictionary())

    def decomposition(self, *args, **kwargs):
        super().decomposition(*args, **kwargs)
        self.__class__ = LazyXrayDiffraction2D
