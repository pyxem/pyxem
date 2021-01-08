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

import pytest
import numpy as np

from scipy.ndimage import gaussian_filter

from hyperspy.signals import BaseSignal

from pyxem.generators import IntegrationGenerator
from pyxem.signals import DiffractionVectors, ElectronDiffraction2D


@pytest.mark.parametrize("radius, offset", [[1, 0], [2, 1], [3, 2]])
def test_integration_generator(radius, offset):
    pixel_positions = np.array([[0, 0], [15, -15], [-15, 15]])
    pattern = np.zeros((50, 50))
    center = np.array(pattern.shape) / 2
    i, j = (pixel_positions + center + offset).T.astype(int)
    pattern[i, j] = 1

    dv = DiffractionVectors(pixel_positions)
    dp = ElectronDiffraction2D(pattern)
    ig = IntegrationGenerator(dp, dv)
    assert isinstance(ig, IntegrationGenerator)

    inties = ig.extract_intensities(radius=radius)
    assert isinstance(inties, BaseSignal)

    assert np.allclose(inties.data, [1, 1, 1])


def test_integration_generator_summation_method():
    pixel_positions = np.array([[0, 0], [25, -25], [-25, 25]])
    pattern = np.zeros((100, 100))
    center = np.array(pattern.shape) / 2
    i, j = (pixel_positions + center).T.astype(int)
    pattern[i, j] = 1.0
    pattern = gaussian_filter(pattern, 2)

    dv = DiffractionVectors(pixel_positions)
    dp = ElectronDiffraction2D(pattern)
    ig = IntegrationGenerator(dp, dv)

    assert isinstance(ig, IntegrationGenerator)

    vectors = ig.extract_intensities_summation_method()

    assert np.allclose(pixel_positions, vectors.data, atol=0.05)
    assert np.allclose(vectors.data, pixel_positions, atol=0.05)
    assert np.allclose(vectors.intensities.data[0], 1.0, atol=0.05)
    assert np.allclose(vectors.sigma.data[0], 0.0, atol=0.05)
    assert isinstance(vectors, DiffractionVectors)
