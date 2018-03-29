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

import pytest
from pyxem.utils.orientation_utils import build_linear_grid_in_euler

@pytest.mark.skip(reason="function not currently implemented")
def test_DiffractionLibraryIO():
    # Create a library
    # Save it
    # Load it
    # Assert loaded is the same as saved
    return None

def test_build_linear_grid_in_euler():
    alpha,beta,gamma = 5,5,5
    width = 5
    resolution = 2
    
    # Thus we expect 5,7 and 9 from each of the three, 27 items
    grid = build_linear_grid_in_euler(alpha,beta,gamma,width,resolution)
    assert len(grid) == 27
    return None
