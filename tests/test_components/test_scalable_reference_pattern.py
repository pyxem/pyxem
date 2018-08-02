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

from pyxem.components.scalable_reference_pattern import ScalableReferencePattern

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
