# -*- coding: utf-8 -*-
# Copyright 2017-2018 The pyXem developers
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
from pyxem.generators.indexation_generator import IndexationGenerator
from pyxem import ElectronDiffraction


@pytest.fixture
def coords_intensity_simulation():
    return DiffractionSimulation(coordinates = np.asarray([[0.3,1.2,0]]), intensities = np.ones(1))

@pytest.fixture
def get_signal():
    size  = 144
    sigma = 0.03
    max_r = 1.5
    return coords_intensity_simulation().as_signal(size,sigma,max_r)

def test_typing():
    assert type(get_signal()) is ElectronDiffraction

def test_correct_quadrant_np():
    A = get_signal().data
    assert (np.sum(A[:72,:72]) == 0)
    assert (np.sum(A[72:,:72]) == 0)
    assert (np.sum(A[:72,72:]) == 0)
    assert (np.sum(A[72:,72:])  > 0)
