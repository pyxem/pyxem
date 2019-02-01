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


@pytest.fixture
def fit_parameters(self):
    x0 = [95, 1200, 2.8, 450, 1.5, 10]
    return x0

@pytest.fixture
def generate_parameters(self):
    x0 = [95, 256, 1200, 2.8, 450, 1.5, 10]
    return x0

@pytest.fixture
def calibration_generator(pattern_for_fit_ring):
    return CalibrationGenerator(ElectronDiffraction(np.zeros((256, 256))))

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
def test_generate_ring_pattern(self, calibration_generator, input_parameters,
                               known_values, reference_indices):
    x0 = generate_parameters
    rings = calibration_generator.generate_ring_pattern(mask=True,
                                                        mask_radius=10,
                                                        scale=x0[0],
                                                        size=x0[1],
                                                        amplitude=x0[2],
                                                        spread=x0[3],
                                                        direct_beam_amplitude=x0[4],
                                                        asymmetry=x0[5],
                                                        rotation=x0[6])
    assert np.allclose(known_values,
                       rings[reference_indices[:, 0], reference_indices[:, 1]])


@pytest.fixture
def calibration_generator(pattern_for_fit_ring):
    return CalibrationGenerator(ElectronDiffraction(np.zeros((256, 256))))


@pytest.fixture
def cal_generator_wt_data(self, calibration_generator, input_parameters):
    x0 = input_parameters
    ring_data = calibration_generator.generate_ring_pattern(mask=True,
                                                            mask_radius=10,
                                                            scale=x0[0],
                                                            amplitude=x0[1],
                                                            spread=x0[2],
                                                            direct_beam_amplitude=x0[3],
                                                            asymmetry=x0[4],
                                                            rotation=x0[5])
    dp = ElectronDiffraction(data)
    return CalibrationGenerator(dp)

def test_fit_ring_pattern(self, cal_generator_wt_data,
                          fit_parameters):
    x0 = input_parameters
    xf = pattern_for_fit_ring.fit_ring_pattern(10)
    # Need to re-phase the rotation angle
    mod0 = x0[-1] % (2 * np.pi)
    modf = xf[-1] % (2 * np.pi)
    if mod0 > 3 * np.pi / 2:
        x0[-1] = 2 * np.pi - mod0
    elif mod0 > np.pi / 2:
        x0[-1] = mod0 - np.pi
    else:
        x0[-1] = mod0
    if modf > 3 * np.pi / 2:
        xf[-1] = 2 * np.pi - modf
    elif modf > np.pi / 2:
        xf[-1] = modf - np.pi
    else:
        xf[-1] = modf
    assert np.allclose(x0, xf)
