# -*- coding: utf-8 -*-
# Copyright 2017-2019 The pyXem developers
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
import numpy as np

from pyxem.generators.variance_generator import VarianceGenerator
from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D

from pyxem.signals.diffraction_variance2d import DiffractionVariance2D
from pyxem.signals.diffraction_variance2d import ImageVariance


@pytest.fixture
def variance_generator(diffraction_pattern):
    return VarianceGenerator(diffraction_pattern)


class TestVarianceGenerator:

    @pytest.mark.parametrize('dqe', [
        0.5,
        0.6
    ])
    def test_get_diffraction_variance(
            self,
            variance_generator: VarianceGenerator,
            dqe
    ):

        vardps = variance_generator.get_diffraction_variance(dqe)
        assert isinstance(vardps, DiffractionVariance2D)

        mean_dp = np.array(
            [[0., 0., 0., 0., 0., 0., 0., 0.5],
             [0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0.5, 0., 0., 0., 0.],
                [0., 0., 0.5, 1., 1.25, 0., 0., 0.],
                [0., 0., 0., 1.25, 1., 0.75, 0., 0.],
                [0., 0., 0., 0., 0.75, 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.]]).reshape(8, 8)
        meansq_dp = np.array(
            [[0., 0., 0., 0., 0., 0., 0., 1.],
             [0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0.5, 0., 0., 0., 0.],
                [0., 0., 0.5, 2., 1.75, 0., 0., 0.],
                [0., 0., 0., 1.75, 2., 1.25, 0., 0.],
                [0., 0., 0., 0., 1.25, 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.]]).reshape(8, 8)
        var_dp = np.array(
            [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3.],
             [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, 1., np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, 1., 1., 0.12, np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan, 0.12, 1., 0.6875 / 0.75 / 0.75, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, 0.6875 / 0.75 / 0.75, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]).reshape(8, 8)
        corr_var_dp = var_dp - np.divide(dqe, mean_dp)
        corr_var_dp[np.isinf(corr_var_dp)] = 0
        corr_var_dp[np.isnan(corr_var_dp)] = 0

        assert np.array_equal(vardps.data[0, 0], mean_dp)
        assert np.array_equal(vardps.data[0, 1], meansq_dp)
        assert np.allclose(vardps.data[1, 0], var_dp, atol=1e-14, equal_nan=True)
        assert np.allclose(vardps.data[1, 1], corr_var_dp, atol=1e-14, equal_nan=True)

    def test_set_data_type(
            self):
        # set 8 bit data type which will result in an incorrect mean square
        # pattern if not changed.

        dp_array = np.array([[
            [[10, 10], [50, 30]],
            [[8, 20], [50, 30]]],
            [[[12, 30], [0, 30]],
             [[10, 10], [50, 30]]
             ]]).astype(np.uint8)

        variance_test_diffraction_pattern = ElectronDiffraction2D(dp_array)
        vargen = VarianceGenerator(variance_test_diffraction_pattern)

        vardps_8 = vargen.get_diffraction_variance(dqe=1)
        assert isinstance(vardps_8, DiffractionVariance2D)
        corr_var_dp_8 = np.array([[-0.08, -0.66857143],
                                  [-0.92213333, -0.88666667]])
        assert np.allclose(vardps_8.data[1, 1], corr_var_dp_8, atol=1e-6, equal_nan=True)

        vardps_16 = vargen.get_diffraction_variance(dqe=1, set_data_type=np.uint16)
        assert isinstance(vardps_16, DiffractionVariance2D)
        corr_var_dp_16 = np.array([[-0.08, 0.16734694],
                                   [0.30666667, -0.0333333]])

        assert np.allclose(vardps_16.data[1, 1], corr_var_dp_16, atol=1e-6, equal_nan=True)

    @pytest.mark.parametrize('dqe', [
        0.5,
        0.6
    ])
    def test_get_image_variance(
            self,
            variance_generator: VarianceGenerator,
            dqe):

        varims = variance_generator.get_image_variance(dqe)
        assert isinstance(varims, ImageVariance)

        mean_im = np.array([[6 / 64, 6 / 64], [8 / 64, 10 / 64]]).reshape(2, 2)
        meansq_im = np.array([[8 / 64, 8 / 64], [12 / 64, 20 / 64]]).reshape(2, 2)
        var_im = np.array([[13.222222222222221, 13.222222222222221],
                           [11.0, 11.8]]).reshape(2, 2)
        corr_var_im = var_im - np.divide(dqe, mean_im)

        assert np.array_equal(varims.data[0, 0], mean_im)
        assert np.array_equal(varims.data[0, 1], meansq_im)
        assert np.allclose(varims.data[1, 0], var_im, atol=1e-14)
        assert np.allclose(varims.data[1, 1], corr_var_im, atol=1e-14)
