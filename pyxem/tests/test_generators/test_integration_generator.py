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

from pyxem.generators.integration_generator import IntegrationGenerator
from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D
from hyperspy.signals import BaseSignal


# @pytest.fixture
# def pixel_positions():
#     positions = np.array([[0, 0], [15, -15], [-15, 15]])
#     return positions

# @pytest.fixture
# def diffraction_pattern():
#     pattern = np.zeros((50, 50))
#     i,j = pixel_positions.T
#     pattern[i, j] = 1
#     return pattern

@pytest.mark.parametrize("radius, offset", [[1, 0], [2, 1], [3, 2]])
# @pytest.mark.parametrize("offset", [0, 1, 2])
def test_integration_generator(radius, offset):
    pixel_positions = np.array([[0, 0], [15, -15], [-15, 15]])
    pattern = np.zeros((50, 50))
    center = np.array(pattern.shape) / 2
    i,j = (pixel_positions + center + offset).T.astype(int)
    pattern[i, j] = 1

    dv = DiffractionVectors(pixel_positions)
    dp = ElectronDiffraction2D(pattern)
    ig = IntegrationGenerator(dp, dv)
    assert isinstance(ig, IntegrationGenerator)
    
    inties = ig.extract_intensities(radius=radius)
    assert isinstance(inties, BaseSignal)
    
    assert np.allclose(inties.data, [1, 1, 1])