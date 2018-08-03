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
from pyxem.signals.diffraction_simulation import ProfileSimulation
from pyxem.generators.diffraction_generator import DiffractionGenerator
from pyxem import ElectronDiffraction
import pymatgen as pmg


@pytest.fixture
def coords_intensity_simulation():
    return DiffractionSimulation(coordinates = np.asarray([[0.3,1.2,0]]),
                                 intensities = np.ones(1))

@pytest.mark.xfail(raises=ValueError)
def test_wrong_calibration_setting():
    DiffractionSimulation(coordinates = np.asarray([[0.3,1.2,0]]),
                          intensities = np.ones(1),
                          calibration = [1,2,5])

@pytest.fixture
def get_signal():
    size  = 144
    sigma = 0.03
    max_r = 1.5
    return coords_intensity_simulation().as_signal(size,sigma,max_r)

def test_typing():
    assert type(get_signal()) is ElectronDiffraction

def test_correct_quadrant_np():
    A = get_signal().data
    assert (np.sum(A[:72,:72]) == 0)
    assert (np.sum(A[72:,:72]) == 0)
    assert (np.sum(A[:72,72:]) == 0)
    assert (np.sum(A[72:,72:])  > 0)

def test_correct_quadrant_hs():
    S = get_signal()
    assert (np.sum(S.isig[:72,:72].data) == 0)
    assert (np.sum(S.isig[72:,:72].data) == 0)
    assert (np.sum(S.isig[:72,72:].data) == 0)
    assert (np.sum(S.isig[72:,72:].data)  > 0)

# These test that our kinematic simulation behaves physically

def get_pattern(microscope,structure):
    return microscope.calculate_ed_data(structure,1)

def check_pattern_equivilance(p1,p2,coords_only=False):
    assert np.allclose(p1.coordinates,p2.coordinates)
    if not coords_only:
        assert np.allclose(p1.indices,p2.indices)
        assert np.allclose(p1.intensities,p2.intensities)

# Becuase of the slight differences between each of the structures, the
# explictily named, ugly pathway has been taken

Cl = pmg.Element("Cl")
Ar = pmg.Element("Ar")
cubic_lattice = pmg.Lattice.cubic(5)
Mscope = DiffractionGenerator(300, 5e-2) #a 300kev EM

formal_cubic_I = pmg.Structure.from_spacegroup("I23", cubic_lattice,
                                               [Cl], [[0, 0, 0]])
casual_cubic_I = pmg.Structure.from_spacegroup(195, cubic_lattice,
                                               [Cl,Cl], [[0, 0, 0],
                                               [0.5,0.5,0.5]])
fake_cubic_I   = pmg.Structure.from_spacegroup(195, cubic_lattice,
                                               [Cl,Ar], [[0, 0, 0],
                                               [0.5,0.5,0.5]])
larger_cubic_I = pmg.Structure.from_spacegroup("I23", cubic_lattice,
                                               [Cl], [[0, 0, 0]])
larger_cubic_I.make_supercell([2,4,2])

formal_pattern = get_pattern(Mscope,formal_cubic_I)
casual_pattern = get_pattern(Mscope,casual_cubic_I)
fake_pattern   = get_pattern(Mscope,fake_cubic_I)
larger_pattern = get_pattern(Mscope,larger_cubic_I)


def test_casual_formal():
    # Checks that Pymatgen understands that these are the same structure
    assert formal_cubic_I == casual_cubic_I

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

# Test Profile Simulation

@pytest.fixture(params=[
    (300, 0.02, None),
])
def diffraction_calculator(request):
    return DiffractionGenerator(*request.param)

@pytest.fixture
def profile_simulation():
    return ProfileSimulation(magnitudes=[0.31891931643691351,
                                         0.52079306292509475,
                                         0.6106839974876449,
                                         0.73651261277849378,
                                         0.80259601243613932,
                                         0.9020400452156796,
                                         0.95675794931074043,
                                         1.0415861258501895,
                                         1.0893168446141808,
                                         1.1645286909108374,
                                         1.2074090451670043,
                                         1.2756772657476541],
                             intensities = np.array([100.,
                                                     99.34619104,
                                                     64.1846346,
                                                     18.57137199,
                                                     28.84307971,
                                                     41.31084268,
                                                     23.42104951,
                                                     13.996264,
                                                     24.87559364,
                                                     20.85636003,
                                                     9.46737774,
                                                     5.43222307]),
                             hkls = [{(1, 1, 1): 8},
                                     {(2, 2, 0): 12},
                                     {(3, 1, 1): 24},
                                     {(4, 0, 0): 6},
                                     {(3, 3, 1): 24},
                                     {(4, 2, 2): 24},
                                     {(3, 3, 3): 8, (5, 1, 1): 24},
                                     {(4, 4, 0): 12},
                                     {(5, 3, 1): 48},
                                     {(6, 2, 0): 24},
                                     {(5, 3, 3): 24},
                                     {(4, 4, 4): 8}])

def test_plot_profile_simulation(profile_simulation):
    profile_simulation.get_plot(g_max=1)
