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
import hyperspy.api as hs

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
    Test the core functionality which is
    loading a previously saved pyxem object
    """
    to_save = class_to_test(np.zeros((2, 2, 2, 2)))
    to_save.metadata.Signal.tracker = meta_string
    to_save.save('tempfile.hspy')
    from_save = hs.load('tempfile.hspy')
    assert isinstance(from_save, class_to_test)
    assert from_save.metadata.Signal.tracker == meta_string
    assert np.allclose(to_save.data, from_save.data)
    os.remove('tempfile.hspy')
