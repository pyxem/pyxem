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
from pyxem.generators.subpixelrefinement_generator import SubpixelrefinementGenerator

@pytest.fixture
def SPR_generator(diffraction_pattern,vect = [[1,1]]):
    return SubpixelrefinementGenerator(diffraction_pattern,vect)

def test_conventional_xc(SPR_generator):
    diff_vect = SPR_generator.conventional_xc()
    assert True

def test_sobel_filtered_xc(SPR_generator):
    diff_vect SPR_generator.sobel_filtered_xc()
    assert True
