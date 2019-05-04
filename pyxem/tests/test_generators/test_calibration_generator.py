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

import pytest
import numpy as np

from pyxem.generators.calibration_generator import CalibrationGenerator

from pyxem.libraries.calibration_library import CalibrationDataLibrary
from pyxem.signals.electron_diffraction import ElectronDiffraction


@pytest.fixture
def input_parameters():
    x0 = [95, 1200, 2.8, 450, 1.5, 10]
    return x0

@pytest.fixture
def calibration_data(input_parameters):
    x0 = input_parameters
    ring_data = generate_ring_pattern(image_size=256,
                                      mask=True,
                                      mask_radius=10,
                                      scale=x0[0],
                                      amplitude=x0[1],
                                      spread=x0[2],
                                      direct_beam_amplitude=x0[3],
                                      asymmetry=x0[4],
                                      rotation=x0[5])

    return ElectronDiffraction(ring_data)

@pytest.fixture
def calibration_library(request, calibration_data):
    return CalibrationDataLibrary(au_x_grating_dp=calibration_data)

#class TestCalibrationGenerator:
#
#    def test_init(self, diffraction_calculator: DiffractionGenerator):
#        assert diffraction_calculator.debye_waller_factors == {}
