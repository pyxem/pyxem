# -*- coding: utf-8 -*-
# Copyright 2017-2020 The pyXem developers
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

from pyxem.signals.crystallographic_map import CrystallographicMap
from pyxem.signals.electron_diffraction1d import ElectronDiffraction1D
from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D
from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.signals.indexation_results import TemplateMatchingResults
from pyxem.signals.vdf_image import VDFImage


@pytest.mark.parametrize("class_to_test,meta_string", [(ElectronDiffraction2D, 'string1'),
                                                       (TemplateMatchingResults, 'string2'),
                                                       (DiffractionVectors, 'string3'),
                                                       (CrystallographicMap, 'string4'),
                                                       (ElectronDiffraction1D, 'string5'),
                                                       (VDFImage, 'string6')])
def test_load_function_core(class_to_test, meta_string):
    """
    Test the core; which is load a previously saved pyxem object.
    """
    to_save = class_to_test(np.zeros((2, 2, 2, 2)))
    to_save.metadata.Signal.tracker = meta_string
    to_save.save('tempfile_for_load_and_save.hspy')
    from_save = pxm.load('tempfile_for_load_and_save.hspy')
    assert isinstance(from_save, class_to_test)
    assert from_save.metadata.Signal.tracker == meta_string
    assert np.allclose(to_save.data, from_save.data)
    os.remove('tempfile_for_load_and_save.hspy')


@pytest.fixture()
def make_saved_Signal2D():
    """
    #Lifted from stackoverflow question #22627659
    """
    s = Signal2D(np.zeros((2, 2, 2, 2)))
    s.metadata.Signal.tracker = 'make_save_Signal2D'
    s.save('S2D_temp')
    s.save('badfilesuffix.emd')
    yield
    os.remove('S2D_temp.hspy')
    os.remove('badfilesuffix.emd')  # for case 3 of the edgecases


@pytest.mark.xfail(raises=ValueError)
def test_load_Signal2D(make_saved_Signal2D):
    """
    This tests that we can "load a Signal2D" with pxm.load and that we auto cast
    safetly into ElectronDiffraction2D
    """
    dp = pxm.load('S2D_temp.hspy')


def test_load_hspy_Signal2D(make_saved_Signal2D):
    """
    This tests that we can "load a Signal2D" with pxm.load and that we auto cast
    safetly into ElectronDiffraction2D
    """
    dp = pxm.load_hspy('S2D_temp.hspy', assign_to='electron_diffraction2d')
    assert dp.metadata.Signal.signal_type == 'electron_diffraction2d'
    assert dp.metadata.Signal.tracker == 'make_save_Signal2D'


@pytest.mark.xfail(raises=ValueError)
def test_load_hspy_Signal2D_not_pyxem(make_saved_Signal2D):
    """
    This tests that we can "load a Signal2D" with pxm.load and that we auto cast
    safetly into ElectronDiffraction2D
    """
    dp = pxm.load_hspy('S2D_temp.hspy', assign_to='not_pyxem_signal')


@pytest.fixture()
def make_saved_dp(diffraction_pattern):
    """
    This makes use of conftest
    """
    diffraction_pattern.save('dp_temp')
    yield
    os.remove('dp_temp.hspy')


def test_load_ElectronDiffraction2D(diffraction_pattern, make_saved_dp):
    """
    This tests that our load function keeps .data, instance and metadata
    """
    dp = pxm.load('dp_temp.hspy')
    assert np.allclose(dp.data, diffraction_pattern.data)
    assert isinstance(dp, ElectronDiffraction2D)
    assert diffraction_pattern.metadata.Signal.found_from == dp.metadata.Signal.found_from
