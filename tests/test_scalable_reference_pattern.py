# -*- coding: utf-8 -*-
# Copyright 2017 The PyCrystEM developers
#
# This file is part of PyCrystEM.
#
# PyCrystEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyCrystEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyCrystEM.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import nose.tools as nt

from pycrystem.scalable_reference_pattern import ScalableReferencePattern
from pycrystem.tensor_field2d import TensorField2D

class TestScalableReferencePattern:

    def setUp(self):
        ref = ScalableReferencePattern(np.ones((2, 2, 2, 2)))
        self.signal = ref

    def test_function(self):
        ref = self.signal
        md = dp.metadata
        nt.assert_equal(md.Acquisition_instrument.TEM.rocking_angle,
                        preferences.ElectronDiffraction.ed_precession_angle)

    def test_construct_displacement_gradient_tensor_type(self):
        ref = self.signal
        ref.construct_displacement_gradient()
        #Assert is TensorField2D

    def test_construct_displacement_gradient_tensor_contents(self):
        ref = self.signal
        ref.construct_displacement_gradient()
        #Check values are in the right place
