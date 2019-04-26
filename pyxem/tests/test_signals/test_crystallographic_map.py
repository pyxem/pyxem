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
from pyxem.signals.crystallographic_map import CrystallographicMap
from pyxem.signals.crystallographic_map import load_mtex_map
from pyxem.signals.crystallographic_map import _distance_from_fixed_angle
from transforms3d.euler import euler2quat, quat2axangle
from transforms3d.quaternions import qmult, qinverse
import os


def worker_for_test_CrystallographicMap_io(mapp):
    mapp.save_mtex_map('file_01.txt')
    lmap = load_mtex_map('file_01.txt')
    os.remove('file_01.txt')
    return mapp, lmap


def get_distance_between_two_angles_longform(angle_1, angle_2):
    """
    Using the long form to find the distance between two angles in euler form
    """
    q1 = euler2quat(*np.deg2rad(angle_1), axes='rzxz')
    q2 = euler2quat(*np.deg2rad(angle_2), axes='rzxz')
    # now assume transform of the form MODAL then Something = TOTAL
    # so we want to calculate MODAL^{-1} TOTAL

    q_from_mode = qmult(qinverse(q2), q1)
    axis, angle = quat2axangle(q_from_mode)
    return np.rad2deg(angle)


@pytest.fixture()
def sp_cryst_map():
    """
    Generates a single phase Crystallographic Map
    """
    base = np.empty((4, 3), dtype='object')
    base[0] = [0, np.array([5, 17, 6]), {'correlation': 3e-17, 'orientation_reliability': 0.5}]
    base[1] = [0, np.array([6, 17, 6]), {'correlation': 2e-17, 'orientation_reliability': 0.4}]
    base[2] = [0, np.array([12, 3, 6]), {'correlation': 4e-17, 'orientation_reliability': 0.3}]
    base[3] = [0, np.array([12, 3, 5]), {'correlation': 8e-16, 'orientation_reliability': 0.2}]
    crystal_map = CrystallographicMap(base.reshape((2, 2, 3)))
    crystal_map.method = 'template_matching'
    return crystal_map


@pytest.fixture()
def dp_cryst_map():
    """
    Generates a Crystallographic Map with two phases
    """
    base = np.empty((4, 3), dtype='object')
    base[0] = [0, np.array([5, 17, 6]), {'correlation': 3e-17, 'orientation_reliability': 0.5, 'phase_reliability': 0.6}]
    base[1] = [1, np.array([6, 17, 6]), {'correlation': 2e-17, 'orientation_reliability': 0.4, 'phase_reliability': 0.7}]
    base[2] = [0, np.array([12, 3, 6]), {'correlation': 4e-17, 'orientation_reliability': 0.3, 'phase_reliability': 0.1}]
    base[3] = [0, np.array([12, 3, 5]), {'correlation': 8e-16, 'orientation_reliability': 0.2, 'phase_reliability': 0.8}]
    crystal_map = CrystallographicMap(base.reshape((2, 2, 3)))
    crystal_map.method = 'template_matching'
    return crystal_map


@pytest.fixture()
def mod_cryst_map():
    """
    Generates a Crystallographic Map with (5,17,6) as the modal angle
    """
    base = np.empty((6, 3), dtype='object')
    base[0] = [0, np.array([5, 17, 6]), {'correlation': 5e-17, 'orientation_reliability': 0.5}]
    base[1] = [0, np.array([5, 17, 6]), {'correlation': 5e-17, 'orientation_reliability': 0.5}]
    base[2] = [0, np.array([6, 19, 6]), {'correlation': 5e-17, 'orientation_reliability': 0.5}]
    base[3] = [0, np.array([7, 19, 6]), {'correlation': 5e-17, 'orientation_reliability': 0.5}]
    base[4] = [0, np.array([8, 19, 6]), {'correlation': 5e-17, 'orientation_reliability': 0.5}]
    base[5] = [0, np.array([9, 19, 6]), {'correlation': 5e-17, 'orientation_reliability': 0.5}]
    crystal_map = CrystallographicMap(base.reshape((3, 2, 3)))
    crystal_map.method = 'template_matching'
    return crystal_map


@pytest.fixture()
def dp_cryst_map_vector():
    """
    Generates a Crystallographic Map with two phases from vector matching
    """
    base = np.empty((4, 3), dtype='object')
    base[0] = [0, np.array([5, 17, 6]), {
        'match_rate': 0.5, 'ehkls': np.array([0.1, 0.05, 0.2]),
        'total_error': 0.1, 'orientation_reliability': 13.2,
        'phase_reliability': 42.0}]
    base[1] = [1, np.array([6, 17, 6]), {
        'match_rate': 0.5, 'ehkls': np.array([0.1, 0.05, 0.2]),
        'total_error': 0.1, 'orientation_reliability': 13.2,
        'phase_reliability': 42.0}]
    base[2] = [0, np.array([12, 3, 6]), {
        'match_rate': 0.5, 'ehkls': np.array([0.1, 0.05, 0.2]),
        'total_error': 0.1, 'orientation_reliability': 13.2,
        'phase_reliability': 42.0}]
    base[3] = [0, np.array([12, 3, 5]), {
        'match_rate': 0.5, 'ehkls': np.array([0.1, 0.05, 0.2]),
        'total_error': 0.1, 'orientation_reliability': 13.2,
        'phase_reliability': 42.0}]
    crystal_map = CrystallographicMap(base.reshape((2, 2, 3)))
    crystal_map.method = 'vector_matching'
    return crystal_map


class TestMapCreation:

    def test_get_phase_map(self, sp_cryst_map):
        phasemap = sp_cryst_map.get_phase_map()
        assert phasemap.isig[0, 0] == 0

    def test_get_orientation_map(self, sp_cryst_map):
        orimap = sp_cryst_map.get_orientation_map()
        assert orimap.isig[0, 0] == 0

    @pytest.mark.parametrize('metric, value', [
        ('correlation', 3e-17),
        ('orientation_reliability', 0.5),
        ('phase_reliability', 0.6)
    ])
    def test_get_metric_map_template_match(self, dp_cryst_map, metric, value):
        metric_map = dp_cryst_map.get_metric_map(metric)
        assert metric_map.isig[0, 0] == value

    @pytest.mark.parametrize('metric, value', [
        ('match_rate', 0.5),
        ('ehkls', np.array([0.1, 0.05, 0.2])),
        ('total_error', 0.1),
        ('orientation_reliability', 13.2),
        ('phase_reliability', 42.0)
    ])
    def test_get_metric_map_vector_match(self, dp_cryst_map_vector, metric, value):
        metric_map = dp_cryst_map_vector.get_metric_map(metric)
        assert np.allclose(metric_map.isig[0, 0], value)

    @pytest.mark.xfail(raises=ValueError)
    def test_get_metric_map_template_match_bad_metric(self, sp_cryst_map):
        metric_map = sp_cryst_map.get_metric_map('no metric')

    @pytest.mark.xfail(raises=ValueError)
    def test_get_metric_map_vector_match_bad_metric(self, dp_cryst_map_vector):
        metric_map = dp_cryst_map_vector.get_metric_map('no metric')

    @pytest.mark.xfail(raises=ValueError)
    def test_get_metric_map_no_method(self):
        crystal_map = CrystallographicMap(np.array([[1]]))
        metric_map = crystal_map.get_metric_map('no metric')


class TestMTEXIO:
    def test_Crystallographic_Map_io_sp(self, sp_cryst_map):
        saved, loaded = worker_for_test_CrystallographicMap_io(sp_cryst_map)
        saved.method = 'template_matching'
        loaded.method = 'template_matching'
        assert np.allclose(saved.isig[0].data.astype('int'), loaded.isig[0].data.astype('int'))
        assert np.allclose(np.array(saved.isig[1].data.tolist()), np.array(loaded.isig[1].data.tolist()))
        assert np.allclose(saved.get_metric_map('correlation').data, loaded.get_metric_map('correlation').data)

    def test_Crystallographic_Map_io_dp(self, dp_cryst_map):
        saved, loaded = worker_for_test_CrystallographicMap_io(dp_cryst_map)
        saved.method = 'template_matching'
        loaded.method = 'template_matching'
        assert np.allclose(saved.isig[0].data.astype('int'), loaded.isig[0].data.astype('int'))
        assert np.allclose(np.array(saved.isig[1].data.tolist()), np.array(loaded.isig[1].data.tolist()))
        assert np.allclose(saved.get_metric_map('correlation').data, loaded.get_metric_map('correlation').data)


class TestModalAngularFunctionality:

    def test_get_distance_from_modal(self, mod_cryst_map):
        # function runs without error
        formal = mod_cryst_map.get_distance_from_modal_angle()
        assert True

    def test_get_modal_angles(self, mod_cryst_map):
        # modal angle is found correctly
        out = mod_cryst_map.get_modal_angles()
        assert np.allclose(out[0], [5, 17, 6])
        assert np.allclose(out[1], (2 / 6))

    def test_get_distance_from_fixed_angle(self):
        # distance between two angles is found correctly
        angle_1 = [1, 1, 3]
        angle_2 = [1, 1, 4]
        implemented = _distance_from_fixed_angle([angle_1], angle_2)
        testing = get_distance_between_two_angles_longform(angle_1, angle_2)
        assert np.allclose(implemented, testing)
        assert np.allclose(implemented, 1)
