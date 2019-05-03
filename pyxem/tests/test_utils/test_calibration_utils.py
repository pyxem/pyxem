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

from pyxem.signals.electron_diffraction import ElectronDiffraction

from pyxem.utils.calibration_utils import call_ring_pattern, \
                                          calc_radius_with_distortion, \
                                          generate_ring_pattern

@pytest.fixture
def input_parameters():
    x0 = [95, 1200, 2.8, 450, 1.5, 10]
    return x0

@pytest.mark.parametrize('known_values', [
    (np.array(
        [124.05909278, 25.85258647, 39.09906246,
         173.75469207, 79.48046629, 533.72925614,
         36.23521052, 29.58603406, 21.83270633,
         75.89239623, 40.04732689, 14.52041808,
         35.82637996, 75.33666451, 21.21751965,
         38.97731538, 19.64631964, 161.72783637,
         23.6894442, 282.3126376]
    ))])

@pytest.mark.parametrize('reference_indices', [
    (np.array(
        [[205, 158],
         [197, 1],
         [105, 239],
         [64, 148],
         [61, 84],
         [136, 155],
         [37, 85],
         [21, 94],
         [247, 31],
         [171, 195],
         [202, 39],
         [225, 255],
         [233, 128],
         [56, 107],
         [22, 51],
         [28, 119],
         [20, 45],
         [164, 65],
         [235, 188],
         [75, 186]]
    ))])

def test_generate_ring_pattern(input_parameters,
                               known_values, reference_indices):
    x0 = input_parameters
    rings = generate_ring_pattern(image_size=256,
                                  mask=True,
                                  mask_radius=10,
                                  scale=x0[0],
                                  amplitude=x0[1],
                                  spread=x0[2],
                                  direct_beam_amplitude=x0[3],
                                  asymmetry=x0[4],
                                  rotation=x0[5])
    assert np.allclose(known_values,
                       rings[reference_indices[:, 0], reference_indices[:, 1]])
