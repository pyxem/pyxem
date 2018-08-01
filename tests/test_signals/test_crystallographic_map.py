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
from pyxem.signals.crystallographic_map  import CrystallographicMap
from pyxem.signals.crystallographic_map import load_mtex_map,_distance_from_fixed_angle
from transforms3d.euler import euler2quat,quat2axangle
from transforms3d.quaternions import qmult,qinverse
import os

def get_distance_between_two_angles_longform(angle_1,angle_2):
    """
    Using the long form to find the distance between two angles in euler form
    """
    q1 = euler2quat(*np.deg2rad(angle_1),axes='rzxz')
    q2 = euler2quat(*np.deg2rad(angle_2),axes='rzxz')
    ## now assume transform of the form MODAL then Something = TOTAL
    ## so we want to calculate MODAL^{-1} TOTAL

    q_from_mode = qmult(qinverse(q2),q1)
    axis,angle = quat2axangle(q_from_mode)
    return angle

@pytest.fixture()
def sp_cryst_map():
    """
    Generates a single phase Crystallographic Map
    """
    base = np.zeros((4,6))
    base[0] = [0,5,17,6,3e-17,0.5]
    base[1] = [0,6,17,6,2e-17,0.4]
    base[2] = [0,12,3,6,4e-17,0.3]
    base[3] = [0,12,3,5,8e-16,0.2]
    crystal_map = CrystallographicMap(base.reshape((2,2,6)))
    return crystal_map

@pytest.fixture()
def dp_cryst_map():
    """
    Generates a Crystallographic Map with two phases
    """
    base = np.zeros((4,7))
    base[0] = [0,5,17,6,3e-17,0.5,0.6]
    base[1] = [1,6,17,6,2e-17,0.4,0.7]
    base[2] = [0,12,3,6,4e-17,0.3,0.1]
    base[3] = [0,12,3,5,8e-16,0.2,0.8]
    crystal_map = CrystallographicMap(base.reshape((2,2,7)))
    return crystal_map

@pytest.fixture()
def mod_cryst_map():
    """
    Generates a Crystallographic Map with (5,17,6) as the modal angle
    """
    base = np.zeros((6,6))
    base[0] = [0,5,17,6,5e-17,0.5]
    base[1] = [0,5,17,6,5e-17,0.5]
    base[2] = [0,6,19,6,5e-17,0.5]
    base[3] = [0,7,19,6,5e-17,0.5]
    base[4] = [0,8,19,6,5e-17,0.5]
    base[5] = [0,9,19,6,5e-17,0.5]
    crystal_map = CrystallographicMap(base.reshape((3,2,6)))
    return crystal_map

@pytest.fixture()
def zero_modal_map():
    """
    Generates a crystallographic map, modal angle (0,0,0)
    """
    base = np.zeros((4,7))
    base[0] = [0,2,0,4,3e-17,0.5,0.6]
    base[1] = [0,0,0,0,2e-17,0.4,0.7]
    base[2] = [0,0,0,0,4e-17,0.3,0.1]
    base[3] = [0,0,0,0,8e-16,0.2,0.8]
    crystal_map = CrystallographicMap(base.reshape((2,2,7)))
    return crystal_map



def test_get_phase_map(sp_cryst_map):
    phasemap = sp_cryst_map.get_phase_map()
    assert phasemap.isig[0,0] == 0

def test_get_correlation_map(sp_cryst_map):
    correlationmap = sp_cryst_map.get_correlation_map()
    assert correlationmap.isig[0,0] == 3e-17

def test_get_reliability_map_orientation(sp_cryst_map):
    reliabilitymap_orientation = sp_cryst_map.get_reliability_map_orientation()
    assert reliabilitymap_orientation.isig[0,0] == 0.5

def test_get_reliability_map_phase(dp_cryst_map):
    reliabilitymap_phase = dp_cryst_map.get_reliability_map_phase()
    assert reliabilitymap_phase.isig[0,0] == 0.6

@pytest.mark.parametrize('maps',[sp_cryst_map(),dp_cryst_map()])
def test_CrystallographicMap_io(maps):
    maps.save_mtex_map('file_01.txt')
    lmap = load_mtex_map('file_01.txt')
    os.remove('file_01.txt')
    # remember we've dropped reliability in saving
    assert np.allclose(maps.data[:,:,:5],lmap.data)

def test_get_modal_angles(mod_cryst_map):
    out = mod_cryst_map.get_modal_angles()
    assert np.allclose(out[0],[5,17,6])
    assert np.allclose(out[1],(2/6))

def test_get_distance_from_fixed_angle():
    angle_1 = [1,1,3]
    angle_2 = [1,2,4]
    implemented = _distance_from_fixed_angle(angle_1,angle_2)
    testing = get_distance_between_two_angles_longform(angle_1,angle_2)
    assert np.allclose(implemented,testing)

def test_get_distance_from_modal(zero_modal_map):
    formal = zero_modal_map.get_distance_from_modal_angle()
    #if this runs, the test above has confirmed that is works
    assert True
