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
import pymatgen as pmg
import numpy as np

from pyxem.generators.diffraction_generator import DiffractionGenerator
from pyxem.components.diffraction_component import ElectronDiffractionForwardModel

@pytest.fixture(params=[
    (300, 0.02, None),
])
def diffraction_calculator(request):
    return DiffractionGenerator(*request.param)

@pytest.fixture(params=[
    "Si",
])
def element(request):
    return pmg.Element(request.param)

@pytest.fixture(params=[
    5.431
])
def lattice(request):
    return pmg.Lattice.cubic(request.param)

@pytest.fixture(params=[
    "Fd-3m"
])
def structure(request, lattice, element):
    return pmg.Structure.from_spacegroup(request.param, lattice, [element], [[0, 0, 0]])

def test_electron_diffraction_component_init(diffraction_calculator,
                                             structure):
    ref = ElectronDiffractionForwardModel(diffraction_calculator,
                                          structure,
                                          reciprocal_radius=1.,
                                          calibration=0.01)
    assert isinstance(ref, ElectronDiffractionForwardModel)

def test_function(diffraction_calculator, structure):
    ref = ElectronDiffractionForwardModel(diffraction_calculator,
                                          structure,
                                          reciprocal_radius=1.,
                                          calibration=0.01)
    func = ref.function()
    np.testing.assert_almost_equal(func, 1)

def test_simulate(diffraction_calculator, structure):
    ref = ElectronDiffractionForwardModel(diffraction_calculator,
                                          structure,
                                          reciprocal_radius=1.,
                                          calibration=0.01)
    sim = ref.simulate()
    assert isinstance(ref, DiffractionSimulation)
