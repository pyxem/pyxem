# -*- coding: utf-8 -*-
# Copyright 2016 The PyCrystEM developers
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

from pycrystem.tensor_field import TensorField

class TestTensorField:

    def setUp(self):
        # Create an empty diffraction pattern
        tf = TensorField(np.ones((2, 2, 2, 2)))
        self.signal = tf

    def test_polar_decomposition(self):
        tf = self.signal
        md = dp.metadata
        nt.assert_equal(md.Acquisition_instrument.TEM.rocking_angle,
                        preferences.ElectronDiffraction.ed_precession_angle)

    def test_change_basis(self):
        pass
