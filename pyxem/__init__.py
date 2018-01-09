# -*- coding: utf-8 -*-
# Copyright 2018 The pyXem developers
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

from hyperspy.api import load as hsload
from hyperspy.api import roi

from pymatgen import Lattice, Structure

from .diffraction_signal import ElectronDiffraction
from .diffraction_generator import ElectronDiffractionCalculator
from .library_generator import DiffractionLibraryGenerator
from .diffraction_component import ElectronDiffractionForwardModel
from .scalable_reference_pattern import ScalableReferencePattern

def load(*args, **kwargs):
    signal = hsload(*args, **kwargs)
    return ElectronDiffraction(**signal._to_dictionary())
