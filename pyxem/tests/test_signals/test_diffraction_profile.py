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

import numpy as np
import pytest
from pyxem.signals.diffraction_profile import ElectronDiffractionProfile


class TestSimpleHyperspy:
    # Tests functions that assign to hyperspy metadata

    def test_set_experimental_parameters(self, diffraction_profile):
        diffraction_profile.set_experimental_parameters(accelerating_voltage=3,
                                                        camera_length=3,
                                                        scan_rotation=1,
                                                        convergence_angle=1,
                                                        rocking_angle=1,
                                                        rocking_frequency=1,
                                                        exposure_time=1)
        assert isinstance(diffraction_profile, ElectronDiffractionProfile)

    def test_set_scan_calibration(self, diffraction_profile):
        diffraction_profile.set_scan_calibration(19)
        assert isinstance(diffraction_profile, ElectronDiffractionProfile)

    @pytest.mark.parametrize('calibration', [1, 0.017, 0.5, ])
    def test_set_diffraction_calibration(self,
                                         diffraction_profile,
                                         calibration):
        diffraction_profile.set_diffraction_calibration(calibration)
        dx = diffraction_profile.axes_manager.signal_axes[0]
        assert dx.scale == calibration


class TestVirtualImaging:
    # Tests that virtual imaging runs without failure

    def test_plot_interactive_virtual_image(self, diffraction_profile):
        diffraction_profile.plot_interactive_virtual_image(left=1., right=2.)

    def test_get_virtual_image(self, diffraction_profile):
        diffraction_profile.get_virtual_image(left=1., right=2.)
