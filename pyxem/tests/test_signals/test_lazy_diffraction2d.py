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

from pyxem.signals.diffraction2d import Diffraction2D, LazyDiffraction2D

<<<<<<< HEAD:tests/test_signals/test_diffraction_profile.py
from hyperspy.signals import Signal1D
from pyxem.signals.diffraction_profile import ElectronDiffractionProfile

class TestDiffractionProfile:
    
    def test_get_electron_diffraction_profile(self,
                        ):
        rad_signal = Signal1D(np.array([0,4,3,5,1,4,6,2]))
        difprof = ElectronDiffractionProfile(rad_signal)
        assert isinstance(difprof,ElectronDiffractionProfile)
=======

class TestDecomposition:
    def test_decomposition_class_assignment(self, diffraction_pattern):
        s = Diffraction2D(diffraction_pattern)
        s = s.as_lazy()
        s.decomposition()
        assert isinstance(s, LazyDiffraction2D)
>>>>>>> master:pyxem/tests/test_signals/test_lazy_diffraction2d.py
