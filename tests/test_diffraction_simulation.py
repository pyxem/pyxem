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
#XXX


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


Cl = pmg.Element("Cl")
Ar = pmg.Element("Ar")
cubic_lattice = pmg.Lattice.cubic(5)
Mscope = DiffractionGenerator(300, 5e-2) #a 300kev EM

formal_cubic_I = pmg.Structure.from_spacegroup("I23",cubic_lattice, [Cl], [[0, 0, 0]])
casual_cubic_I = pmg.Structure.from_spacegroup(195,cubic_lattice, [Cl,Cl], [[0, 0, 0],[0.5,0.5,0.5]])
fake_cubic_I   = pmg.Structure.from_spacegroup(195,cubic_lattice, [Cl,Ar], [[0, 0, 0],[0.5,0.5,0.5]])

larger_cubic_I = pmg.Structure.from_spacegroup("I23",cubic_lattice, [Cl], [[0, 0, 0]])
larger_cubic_I.make_supercell([2,4,2])

def get_pattern(microscope,structure):
    return microscope.calculate_ed_data(structure,1)

def check_pattern_equivilance(p1,p2,coords_only=False):
    assert np.allclose(p1.coordinates,p2.coordinates)
    if not coords_only:
        assert np.allclose(p1.indices,p2.indices)
        assert np.allclose(p1.intensities,p2.intensities) 
    
def test_casual_formal():
    # Checks that Pymatgen understands that these are the same structure
    assert formal_cubic_I == casual_cubic_I

formal_pattern = get_pattern(Mscope,formal_cubic_I)
casual_pattern = get_pattern(Mscope,casual_cubic_I)
fake_pattern   = get_pattern(Mscope,fake_cubic_I)
larger_pattern = get_pattern(Mscope,larger_cubic_I)

def test_casual_formal_in_simulation():
    ## Checks that are simulations also realise that
    check_pattern_equivilance(formal_pattern,casual_pattern)
    
def test_systematic_absence():
    ## Cubic I thus each peak must have indicies that sum to an even number
    assert np.all(np.sum(formal_pattern.indices,axis=1) % 2 == 0)
    assert np.all(np.sum(casual_pattern.indices,axis=1) % 2 == 0)
    ## This isn't actually cubic I, so we expect a (100) type
    assert np.any(fake_pattern.indices == np.array([1,0,0]))

def test_scaling():
    check_pattern_equivilance(formal_pattern,larger_pattern,coords_only=True)
    
#ToDo Generate an A centered and test the sys condition is satisfied 
#ToDo Check obvious thing like doubling lattice size and changing voltages
