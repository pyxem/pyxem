# -*- coding: utf-8 -*-
# Copyright 2017 The pyXem developers
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

from pyxem.diffraction_signal import ElectronDiffraction
from pyxem.library_generator import DiffractionLibraryGenerator
from hyperspy.signals import Signal1D, Signal2D


@pytest.fixture
def generator():
    return DiffractionLibraryGenerator()

