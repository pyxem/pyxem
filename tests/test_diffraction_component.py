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

from pycrystem.diffraction_component import ElectronDiffractionForwardModel
from pycrystem.electron_diffraction_calculator import ElectronDiffractionCalculator
from pymatgen import Structure

class TestDiffractionComponent:

    def setUp(self):
        #Specify Au Crystal structure as test case
        au = Structure()
        #Define electron diffraction calculator
        ed = ElectronDiffractionCalculator(300, 3, 1)
        #Assign component
        diff = ElectronDiffractionForwardModel(electron_diffraction_calculator=ed,
                                               structure=au,
                                               D11=1., D12=0., D13=0.,
                                               D21=0., D22=1., D23=0.,
                                               D31=0., D32=0., D33=1.)
        self.signal = diff

    def test_simulate(self):
        diff = self.signal
        sim = diff.simulate()
        #Assert is somthing known....
