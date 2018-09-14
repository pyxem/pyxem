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
import numpy as np
from pyxem.utils.peakfinders2D import *

"""
A philosophical note:
    It's very dificult to prove that a peakfinder with many parameters couldn't
    find the peaks you have 'defined' if you just picked better parameters.
    These tests are designed to show that when the peakfinders find 0,1,x (x>1)
    peaks the results are of the style - if not the value - that we expect.
"""

# see https://stackoverflow.com/questions/9205081/
dispatcher = {'log': find_peaks_log, 'dog': find_peaks_dog, 'zaf':find_peaks_zaefferer,'stat':find_peaks_stat}

@pytest.fixture
def single_peak():
    pattern = np.zeros((128,128))
    pattern[40:43,40:43] = 1 #index 40,41,42 are greater than zero
    return pattern

@pytest.fixture
def many_peak():
    pattern = np.zeros((128,128))
    pattern[40:43,40:43] = 1 #index 40,41,42 are greater than zero
    pattern[70,21]    = 1 #index 70 and 21 are greater than zero
    pattern[10:13,41:43] = 1
    pattern[100:113,100:105] = 2
    return pattern

@pytest.fixture
def no_peak():
    pattern = np.ones((128,128))*0.5
    return pattern


methods = ['zaf']
# dog and log have no safe way of returning for an empty peak array
# stat throws an error while running
@pytest.mark.parametrize('method', methods)
def test_no_peak_case(no_peak,method):
    peaks = dispatcher[method](no_peak)
    assert np.isnan(peaks[0,0,0])
    assert np.isnan(peaks[0,0,1])

methods = ['zaf','log', 'dog','stat']
@pytest.mark.parametrize('method', methods)
def test_one_peak_case(single_peak,method):
    peaks = dispatcher[method](single_peak)
    assert peaks[0,0] > 39.5
    assert peaks[0,0] < 42.5
    assert peaks[0,0] == peaks[0,1]


methods = ['zaf','log','stat']
@pytest.mark.parametrize('method', methods)
def test_many_peak_case(many_peak,method):
    peaks = dispatcher[method](many_peak)
    assert np.max(peaks) > 2

class TestUncoveredCodePaths:
    def test_zaf_continue(self,many_peak):
        peaks = find_peaks_zaefferer(many_peak,distance_cutoff=1e-5)
