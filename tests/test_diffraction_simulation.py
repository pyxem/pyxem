# -*- coding: utf-8 -*-
# Copyright 2018 The pyXem developers
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
from pyxem.signals.diffraction_simulation import DiffractionSimulation as DiffractionSimulation


@pytest.fixture
def coords_intensity_simulation():
    return DiffractionSimulation(coordinates = np.asarray([[0.3,0.7,0],[0.1,0.8,1],[0.2,1.2,2]]), intensities = np.ones(3))

@pytest.fixture
def as_signal_size_sigma_max_r():
    return [144,0.03,1.5]

@pytest.fixture
def out_of_plane_only():
    return DiffractionSimulation(coordinates = np.asarray([[0.3,0.7,1]]), intensities = np.ones(1))

@pytest.fixture
def get_signal():
    size  = as_signal_size_sigma_max_r()[0]
    sigma = as_signal_size_sigma_max_r()[1]
    max_r = as_signal_size_sigma_max_r()[2]
    return coords_intensity_simulation().as_signal(size,sigma,max_r)

def test_shape_as_expected():
    assert get_signal().data.shape == (as_signal_size_sigma_max_r()[0],as_signal_size_sigma_max_r()[0])
        
def test_out_of_plane_removal():
    assert np.all(out_of_plane_only().as_signal(144,0.03,1.5).data == np.zeros((144,144)))
    
# ToDo - Test low and high sigma 