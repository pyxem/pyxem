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
import pyxem as pxm
import os




@pytest.fixture()
def make_saved_dp(diffraction_pattern):
    """
    This fixture handles the creation and destruction of a saved electron_diffraction
    pattern.
    #Lifted from stackoverflow question #22627659
    """
    diffraction_pattern.save('dp_temp')
    yield
    os.remove('dp_temp.hspy')

def test_load_ElectronDiffraction(diffraction_pattern,make_saved_dp):
    """
    This tests that our load function keeps .data, metadata and instance
    """
    dp = pxm.load('dp_temp.hspy')
    assert np.allclose(dp.data,diffraction_pattern.data)
    assert isinstance(dp, pxm.ElectronDiffraction)
    assert diffraction_pattern.metadata == dp.metadata
