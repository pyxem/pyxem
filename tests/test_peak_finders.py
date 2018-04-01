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
import pyxem as pxm
from pyxem.signals.diffraction_vectors import DiffractionVectors

@pytest.fixture
def create_blank_background():
    return np.zeros((2,2,128,128))

@pytest.fixture
def create_single_peak(background):
    background[:,:,40:43,40:43] = 2
    return pxm.ElectronDiffraction(background)

@pytest.fixture
def create_seperated_double_peak(background):
    background[:,:,40:43,40:43] = 2
    background[:,:,70:73,30:33] = 1.8
    return pxm.ElectronDiffraction(background)

@pytest.fixture
def create_airey_double_peak():
    pass
 
methods = ['skimage', 'max', 'minmax', 'zaefferer', 'stat', 'laplacian_of_gaussians', 'difference_of_gaussians']
samples = [create_single_peak(create_blank_background()),create_seperated_double_peak(create_blank_background())]

@pytest.mark.parametrize("sample",samples)
def test_argless_run(sample):
    assert sample.find_peaks()

@pytest.mark.parametrize("sample",samples)
@pytest.mark.parametrize("method",methods)
def test_findpeaks_runs_without_error(sample,method):
    assert type(sample.find_peaks(method)) == DiffractionVectors

@pytest.mark.parametrize("sample",create_single_peak(create_blank_background())) #single peak
@pytest.mark.parametrize("method",methods)

def test_findpeaks_single(sample,method):
    output = (sample.find_peaks(method)).inav[0,0] #should be <2,2|stuff>
    assert output.data.shape[0] == 1        #  correct number of peaks
    assert output.data[0] == output.data[1] #  sym peaks
    assert output.data[0] == -22.5          #  peak as expected
    