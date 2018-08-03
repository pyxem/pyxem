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
from pyxem.signals.indexation_results import * #import both objects

@pytest.fixture
def sp_match_result():
    row_1 = np.array([0,2,3,4,0.7])
    row_2 = np.array([0,2,3,5,0.6])
    # note we require (correlation of row_1 > correlation row_2)
    return np.vstack((row_1,row_2))

@pytest.fixture
def dp_match_result():
    row_1 = np.array([0,2,3,4,0.7])
    row_2 = np.array([0,2,3,5,0.8])
    row_3 = np.array([1,2,3,4,0.5])
    row_4 = np.array([1,2,3,5,0.3])
    return np.vstack((row_1,row_2,row_3,row_4))

def test_crystal_from_matching_results_sp(sp_match_result):
    #branch single phase
    cmap = crystal_from_matching_results(sp_match_result)
    assert np.allclose(cmap,np.array([0,2,3,4,0.7,100*(1-(0.6/0.7))]))

def test_crystal_from_matching_results_dp(dp_match_result):
    # branch double phase
    cmap = crystal_from_matching_results(dp_match_result)
    r_or = 100*(1-(0.7/0.8))
    r_ph = 100*(1-(0.5/0.8))
    assert np.allclose(cmap,np.array([0,2,3,5,0.8,r_or,r_ph]))

def test_get_crystalographic_map(dp_match_result,sp_match_result):
    #Assertion free test, as the tests above do the heavy lifting
    results = np.vstack((dp_match_result,sp_match_result))
    results = IndexationResults(results)
    results.get_crystallographic_map()
    return 0
