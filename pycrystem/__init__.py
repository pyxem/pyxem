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

from hyperspy.api import load as hsload
from hyperspy.api import roi

from pymatgen import Lattice, Structure

from .diffraction_signal import ElectronDiffraction
from .diffraction_generator import ElectronDiffractionCalculator
from .library_generator import DiffractionLibraryGenerator
from .diffraction_component import ElectronDiffractionForwardModel
from .scalable_reference_pattern import ScalableReferencePattern

__author__ = "Duncan Johnstone"
__copyright__ = "Copyright 2016, Python Crystallographic Electron Microscopy"
__credits__ = ["Duncan Johnstone", "Ben Martineau"]
__license__ = "GPL"
__version__ = "0.4"
__maintainer__ = "Duncan Johnstone"
__email__ = "dnj23@cam.ac.uk"
__status__ = "Development"


def load(*args, **kwargs):
    signal = hsload(*args, **kwargs)
    return ElectronDiffraction(**signal._to_dictionary())
