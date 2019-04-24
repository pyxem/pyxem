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

from pyxem.components.scalable_reference_pattern import ScalableReferencePattern
from pyxem.signals.electron_diffraction import ElectronDiffraction
from pyxem.utils.expt_utils import _index_coords


@pytest.fixture(params=[
    np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 1., 0., 0., 0., 0., 0.],
              [0., 1., 2., 1., 0., 0., 0., 0.],
              [0., 0., 1., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 1., 0., 0.],
              [0., 0., 0., 0., 1., 2., 1., 0.],
              [0., 0., 0., 0., 0., 1., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0.]])
])
def diffraction_pattern(request):
    return ElectronDiffraction(request.param)


def test_scalable_reference_pattern_init(diffraction_pattern):
    ref = ScalableReferencePattern(diffraction_pattern)
    assert isinstance(ref, ScalableReferencePattern)


def test_function(diffraction_pattern):
    ref = ScalableReferencePattern(diffraction_pattern)
    x, y = _index_coords(diffraction_pattern.data)
    func = ref.function(x=x, y=y)
    np.testing.assert_almost_equal(func, diffraction_pattern.data)
