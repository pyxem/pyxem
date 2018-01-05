# -*- coding: utf-8 -*-
# Copyright 2017 The PyCrystEM developers
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

import pytest
import numpy as np

from pycrystem.diffraction_signal import ElectronDiffraction
from hyperspy.signals import Signal1D, Signal2D


@pytest.fixture
def electron_diffraction(request):
    return ElectronDiffraction(request.param)


@pytest.mark.parametrize('electron_diffraction', [
    (np.ones((2, 2, 2, 2)),),
], indirect=['electron_diffraction'])
def test_default_params(electron_diffraction):
    # TODO: preferences in pycrystem
    a = electron_diffraction.metadata.Acquisition_instrument.TEM.rocking_angle
    pass


@pytest.mark.incremental
class TestDirectBeamMethods:

    @pytest.fixture
    def diffraction_pattern(self):
        dp = ElectronDiffraction(np.zeros((4, 8, 8)))
        dp.data[0]= np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 1., 0., 0., 0., 0.],
                              [0., 0., 1., 2., 1., 0., 0., 0.],
                              [0., 0., 0., 1., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.]])

        dp.data[1]= np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 1., 0., 0., 0.],
                              [0., 0., 0., 1., 2., 1., 0., 0.],
                              [0., 0., 0., 0., 1., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.]])

        dp.data[2]= np.array([[0., 0., 0., 0., 0., 0., 0., 2.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 1., 0., 0., 0., 0.],
                              [0., 0., 1., 2., 1., 0., 0., 0.],
                              [0., 0., 0., 1., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.]])

        dp.data[3]= np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 2., 0., 0., 0.],
                              [0., 0., 0., 2., 2., 2., 0., 0.],
                              [0., 0., 0., 0., 2., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.]])
        return dp

    def test_get_direct_beam_position(self, diffraction_pattern):
        c = diffraction_pattern.get_direct_beam_position(radius=2)
        positions = np.array([[3, 3], [4, 4], [3, 3], [4, 4]])
        assert np.allclose(c, positions)

    @pytest.mark.skip(reason="`get_direct_beam_shifts` not implemented")
    @pytest.mark.parametrize('centers, shifts_expected', [
        (None, np.array([[-0.5, -0.5], [0.5, 0.5], [-0.5, -0.5], [0.5, 0.5]]),),
        (
                np.array([[3, 3], [4, 4], [3, 3], [4, 4]]),
                np.array([[-0.5, -0.5], [0.5, 0.5], [-0.5, -0.5], [0.5, 0.5]]),
        ),
        pytest.param(
            np.array([[3, 3], [4, 4], [3, 3]]),
            np.array([[-0.5, -0.5], [0.5, 0.5], [-0.5, -0.5], [0.5, 0.5]]),
            marks=pytest.mark.xfail(raises=ValueError)
        )
    ])
    def test_get_direct_beam_shifts(self, diffraction_pattern, centers, shifts_expected):
        shifts_calculated = diffraction_pattern.get_direct_beam_shifts(radius=2, centers=centers)
        assert np.allclose(shifts_calculated, shifts_expected)

    @pytest.mark.parametrize('center, mask_expected', [
        (None, np.array([
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False,  True,  True, False, False, False],
            [False, False,  True,  True,  True,  True, False, False],
            [False, False,  True,  True,  True,  True, False, False],
            [False, False, False,  True,  True, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False]]),),
        ((4.5, 3.5), np.array([
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False,  True,  True, False, False],
            [False, False, False,  True,  True,  True,  True, False],
            [False, False, False,  True,  True,  True,  True, False],
            [False, False, False, False,  True,  True, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False]]),)
    ])
    def test_get_direct_beam_mask(self, diffraction_pattern, center, mask_expected):
        mask_calculated = diffraction_pattern.get_direct_beam_mask(2, center=center)
        assert isinstance(mask_calculated, Signal2D)
        assert np.equal(mask_calculated, mask_expected)

    @pytest.mark.parametrize('closing, opening, mask_expected', [
        (False, False, np.array([True, True, False, True])),
        (True, False, np.array([True, True, True, True])),
        (False, True, np.array([True, True, False, False])),
    ])
    def test_get_vacuum_mask(self, diffraction_pattern, closing, opening, mask_expected):
        mask_calculated = diffraction_pattern.get_vacuum_mask(
            radius=3, threshold=1, closing=closing, opening=opening)
        assert np.allclose(mask_calculated, mask_expected)


class Test_radial_profile:

    def setUp(self):
        dp = ElectronDiffraction(np.zeros((2, 8, 8)))
        dp.data[0]= np.array([[0., 0., 1., 2., 2., 1., 0., 0.],
                              [0., 1., 2., 3., 3., 2., 1., 0.],
                              [1., 2., 3., 4., 4., 3., 2., 1.],
                              [2., 3., 4., 5., 5., 4., 3., 2.],
                              [2., 3., 4., 5., 5., 4., 3., 2.],
                              [1., 2., 3., 4., 4., 3., 2., 1.],
                              [0., 1., 2., 3., 3., 2., 1., 0.],
                              [0., 0., 1., 2., 2., 1., 0., 0.]])

        dp.data[1]= np.array([[0., 1., 2., 3., 3., 3., 2., 1.],
                              [1., 2., 3., 4., 4., 4., 3., 2.],
                              [2., 3., 4., 5., 5., 5., 4., 3.],
                              [2., 3., 4., 5., 6., 5., 4., 3.],
                              [2., 3., 4., 5., 5., 5., 4., 3.],
                              [1., 2., 3., 4., 4., 4., 3., 2.],
                              [0., 1., 2., 3., 3., 3., 2., 1.],
                              [0., 0., 1., 2., 2., 2., 1., 0.]])
        self.signal = dp

    def test_radial_profile_signal_type(self):
        dp=self.signal
        rp = dp.get_radial_profile()
        nt.assert_true(isinstance(rp, Signal1D))

    def test_radial_profile_no_centers(self):
        dp = self.signal
        rp = dp.get_radial_profile()
        np.testing.assert_allclose(rp.data, np.array([[5., 4.25, 2.875,
                                                       1.7, 0.92857143, 0.],
                                                      [5., 4.75, 3.625,
                                                       2.5, 1.71428571,0.6]]),
                                                      atol=1e-3)

    def test_radial_profile_with_centers(self):
        dp = self.signal
        rp = dp.get_radial_profile(centers=np.array([[4, 3], [4, 3]]))
        np.testing.assert_allclose(rp.data, np.array([[5., 4.25, 2.875,
                                                       1.7, 0.92857143, 0.],
                                                      [5., 4.375, 3.5,
                                                       2.4, 2.07142857, 1.]]),
                                                      atol=1e-3)


class Test_correct_geometric_distortion:

    def setUp(self):
        dp = ElectronDiffraction(np.zeros((2, 8, 8)))
        dp.data[0]= np.array([[0., 0., 1., 2., 2., 1., 0., 0.],
                              [0., 1., 2., 3., 3., 2., 1., 0.],
                              [1., 2., 3., 4., 4., 3., 2., 1.],
                              [2., 3., 4., 5., 5., 4., 3., 2.],
                              [2., 3., 4., 5., 5., 4., 3., 2.],
                              [1., 2., 3., 4., 4., 3., 2., 1.],
                              [0., 1., 2., 3., 3., 2., 1., 0.],
                              [0., 0., 1., 2., 2., 1., 0., 0.]])

        dp.data[1]= np.array([[0., 1., 2., 3., 3., 3., 2., 1.],
                              [1., 2., 3., 4., 4., 4., 3., 2.],
                              [2., 3., 4., 5., 5., 5., 4., 3.],
                              [2., 3., 4., 5., 6., 5., 4., 3.],
                              [2., 3., 4., 5., 5., 5., 4., 3.],
                              [1., 2., 3., 4., 4., 4., 3., 2.],
                              [0., 1., 2., 3., 3., 3., 2., 1.],
                              [0., 0., 1., 2., 2., 2., 1., 0.]])
        self.signal = dp

    def test_correct_geometric_distortion_signal_type(self):
        dp=self.signal
        dp.correct_geometric_distortion(D=np.array([[1., 0., 0.],
                                                    [0., 1., 0.],
                                                    [0., 0., 1.]]))
        nt.assert_true(isinstance(dp, ElectronDiffraction))

#    def test_geometric_distortion_rotation_origin(self):
#        dp = self.signal
#        dp.correct_geometric_distortion()
#        np.testing.assert_allclose(rp.data, np.array([[5., 4.25, 2.875,
#                                                       1.7, 0.92857143, 0.],
#                                                      [5., 4.75, 3.625,
#                                                       2.5, 1.71428571,0.6]]),
#                                                      atol=1e-3)

#    def test_geometric_distortion(self):
#        dp = self.signal
#        rp = dp.get_radial_profile(centers=np.array([[4, 3], [4, 3]]))
#        np.testing.assert_allclose(rp.data, np.array([[5., 4.25, 2.875,
#                                                       1.7, 0.92857143, 0.],
#                                                      [5., 4.375, 3.5,
#                                                       2.4, 2.07142857, 1.]]),
#                                                      atol=1e-3)
