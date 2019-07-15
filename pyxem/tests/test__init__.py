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

from hyperspy.signals import Signal2D
from pyxem.signals.indexation_results import TemplateMatchingResults

@pytest.fixture()
def make_saved_Signal2D():
    z = np.zeros((2,2,2,2))
    s = Signal2D(z)
    s.metadata.Signal.tracker = 'make_save_Signal2D'
    s.save('S2D_temp')
    yield
    os.remove('S2D_temp.hspy')

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

@pytest.fixture()
def make_saved_TMR():
    """
    This fixture handles the creation and destruction of a saved electron_diffraction
    pattern.
    #Lifted from stackoverflow question #22627659
    """
    TMR = TemplateMatchingResults(np.zeros((2,2,2,2)))
    TMR.metadata.Signal.tracker = 'make_saved_TMR'
    TMR.save('TMR_temp')
    yield
    os.remove('TMR_temp.hspy')

TemplateMatchingResults

@pytest.mark.filterwarnings('ignore::UserWarning') #this warning is by design (A)
def test_load_Signal2D(make_saved_Signal2D):
    """
    This tests that we can load a Signal2D with pxm.load and that we can cast
    safetly into ElectronDiffraction
    """
    s = pxm.load('S2D_temp.hspy') #(A)
    assert s.metadata.Signal.tracker == 'make_save_Signal2D'
    dp = pxm.ElectronDiffraction(s)
    assert dp.metadata.Signal.signal_type == 'electron_diffraction'
    assert dp.metadata.Signal.tracker == 'make_save_Signal2D'

def test_load_ElectronDiffraction(diffraction_pattern,make_saved_dp):
    """
    This tests that our load function keeps .data, instance and metadata
    """
    dp = pxm.load('dp_temp.hspy')
    assert np.allclose(dp.data,diffraction_pattern.data)
    assert isinstance(dp, pxm.ElectronDiffraction)
    assert diffraction_pattern.metadata.Signal.found_from == dp.metadata.Signal.found_from

def test_load_TMR(make_saved_TMR):
    """
    This tests our load function keeps TemplateMatchingResults metadata
    """
    TMR = pxm.load('TMR_temp.hspy')
    assert isinstance(TMR, TemplateMatchingResults)
    assert TMR.metadata.Signal.tracker == 'make_saved_TMR'
