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

import numpy as np
import pytest
from pyxem.signals.diffraction_simulation import DiffractionSimulation
from pyxem.generators.diffraction_generator import DiffractionGenerator
import pymatgen as pmg



""" These are .as_signal() tests and should/could be wrapped in a class"""

@pytest.fixture
def coords_intensity_simulation():
    return DiffractionSimulation(coordinates = np.asarray([[0.3,0.7,0],[0.1,0.8,1],[0.2,1.2,2]]), intensities = np.ones(3))

@pytest.fixture
def as_signal_size_sigma_max_r():
    return [144,0.03,1.5]

@pytest.fixture
def get_signal():
    size  = as_signal_size_sigma_max_r()[0]
    sigma = as_signal_size_sigma_max_r()[1]
    max_r = as_signal_size_sigma_max_r()[2]
    return coords_intensity_simulation().as_signal(size,sigma,max_r)

def test_shape_as_expected():
    assert get_signal().data.shape == (as_signal_size_sigma_max_r()[0],as_signal_size_sigma_max_r()[0])
        
# ToDo - Test low and high sigma

""" These test that our kinematic simulation behaves as we would expect it to """

# Generate Cubic I both ways and test ==

Ga = pmg.Element("Ga")
cubic_lattice = pmg.Lattice.cubic(5)
Mscope = DiffractionGenerator(300, 5e-2) #a 300kev EM

@pytest.fixture
def formal_Cubic_I():
    # Formal is using the correct space group
    return pmg.Structure.from_spacegroup("I23",cubic_lattice, [Ga], [[0, 0, 0]])

@pytest.fixture
def casual_Cubic_I():
    # Casual is dropping a motif onto a primitive lattice
    return pmg.Structure.from_spacegroup(195,cubic_lattice, [Ga,Ga], [[0, 0, 0],[0.5,0.5,0.5]])

@pytest.fixture
def formal_pattern():
    return Mscope.calculate_ed_data(formal_Cubic_I(),1)

@pytest.fixture
def casual_pattern():
    return Mscope.calculate_ed_data(casual_Cubic_I(),1)
    
def test_casual_formal():
    # Checks that Pymatgen understands that these are the same structure
    assert formal_Cubic_I() == casual_Cubic_I()

def test_casual_formal_in_simulation():
    ## Checks that are simulations also realise that
    assert np.allclose(formal_pattern().coordinates,casual_pattern().coordinates)
    assert np.allclose(formal_pattern().intensities,casual_pattern().intensities)
    assert np.allclose(formal_pattern().indices,casual_pattern().indices)
    
def test_systematic_absence():
    ## Cubic I thus each peak must have indicies that sum to an even number
    assert np.all(np.sum(formal_pattern().indices,axis=1) % 2 == 0)
    assert np.all(np.sum(casual_pattern().indices,axis=1) % 2 == 0)

#ToDo Generate an A centered and test the sys condition is satisfied 
#ToDo Check obvious thing like doubling lattice size and changing voltages
