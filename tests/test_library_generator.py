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
import numpy as np
import nose.tools as nt

from pycrystem.diffraction_signal import ElectronDiffraction
from hyperspy.signals import Signal1D, Signal2D


class Test_metadata:

    def setUp(self):
        # Create an empty diffraction pattern
        dp = ElectronDiffraction(np.ones((2, 2, 2, 2)))
        dp.axes_manager.signal_axes[0].scale = 1e-3
        dp.metadata.Acquisition_instrument.TEM.accelerating_voltage = 200
        dp.metadata.Acquisition_instrument.TEM.convergence_angle = 15.0
        dp.metadata.Acquisition_instrument.TEM.rocking_angle = 18.0
        dp.metadata.Acquisition_instrument.TEM.rocking_frequency = 63
        dp.metadata.Acquisition_instrument.TEM.Detector.Diffraction.exposure_time = 35
        self.signal = dp

    def test_default_param(self):
        dp = self.signal
        md = dp.metadata
        nt.assert_equal(md.Acquisition_instrument.TEM.rocking_angle,
                        preferences.ElectronDiffraction.ed_precession_angle)
