# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
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

from pyxem.generators import VarianceGenerator
from pyxem.signals import ElectronDiffraction2D, DiffractionVariance2D, ImageVariance


@pytest.fixture
def variance_generator():
    diffraction_pattern = ElectronDiffraction2D(
        np.array(
            [
                [
                    [
                        [0.66, 0.5, 0.01, 0.54, 0.83, 0.23, 0.92, 0.14],
                        [0.61, 0.45, 0.01, 0.42, 0.33, 0.48, 0.35, 0.19],
                        [0.87, 0.98, 0.58, 0.51, 0.75, 0.95, 0.22, 0.52],
                        [0.29, 0.13, 0.22, 0.98, 0.04, 0.12, 0.26, 0.67],
                        [0.59, 0.55, 0.63, 0.57, 0.89, 0.39, 0.81, 0.43],
                        [0.9, 0.54, 0.24, 0.92, 0.77, 0.33, 0.22, 0.7],
                        [0.7, 0.3, 0.76, 0.27, 0.93, 0.28, 0.73, 0.22],
                        [0.33, 0.7, 0.32, 0.54, 0.27, 0.87, 0.97, 0.7],
                    ],
                    [
                        [0.68, 0.33, 0.11, 0.26, 0.88, 0.02, 0.71, 0.98],
                        [0.62, 0.55, 0.22, 0.43, 0.17, 0.77, 0.42, 0.52],
                        [0.45, 0.2, 0.21, 0.53, 0.38, 0.64, 0.32, 0.38],
                        [0.5, 0.23, 0.25, 0.91, 0.64, 1.0, 0.69, 0.16],
                        [0.11, 0.82, 0.23, 0.91, 0.46, 0.08, 0.55, 0.86],
                        [0.95, 0.68, 0.74, 0.44, 0.65, 0.92, 0.04, 0.62],
                        [0.59, 0.08, 0.2, 0.33, 0.94, 0.67, 0.31, 0.62],
                        [0.77, 0.01, 0.46, 0.81, 0.92, 0.72, 0.38, 0.25],
                    ],
                ],
                [
                    [
                        [0.07, 0.63, 0.89, 0.71, 0.65, 0.1, 0.58, 0.78],
                        [0.05, 0.35, 0.02, 0.87, 0.06, 0.06, 0.23, 0.52],
                        [0.44, 0.77, 0.93, 0.94, 0.22, 0.29, 0.22, 0.03],
                        [0.9, 0.79, 0.82, 0.61, 0.6, 0.96, 0.13, 0.63],
                        [0.57, 0.63, 0.53, 0.81, 0.38, 0.92, 0.92, 0.07],
                        [0.01, 0.33, 0.69, 0.36, 0.91, 0.24, 0.05, 0.85],
                        [0.81, 0.38, 0.74, 0.12, 0.33, 0.29, 0.25, 0.06],
                        [0.28, 0.61, 0.37, 0.17, 0.38, 0.52, 0.36, 0.69],
                    ],
                    [
                        [0.77, 0.68, 0.76, 0.62, 0.52, 0.69, 0.63, 0.11],
                        [0.6, 0.96, 0.17, 0.58, 0.12, 0.23, 0.1, 0.2],
                        [0.4, 0.24, 0.25, 0.61, 0.27, 0.02, 0.03, 0.12],
                        [0.83, 0.66, 0.15, 0.74, 0.91, 0.19, 0.97, 0.58],
                        [0.5, 0.27, 0.93, 0.1, 0.78, 0.73, 0.56, 0.82],
                        [0.49, 0.35, 0.9, 0.48, 0.04, 0.46, 0.16, 0.1],
                        [1.0, 0.13, 0.56, 0.12, 0.16, 0.49, 0.63, 0.5],
                        [0.97, 0.71, 0.78, 0.38, 0.08, 0.79, 0.41, 0.07],
                    ],
                ],
            ]
        )
    )
    return VarianceGenerator(diffraction_pattern)


class TestVarianceGenerator:
    @pytest.mark.parametrize("dqe", [0.5, 0.6])
    def test_get_diffraction_variance(self, variance_generator: VarianceGenerator, dqe):

        vardps = variance_generator.get_diffraction_variance(dqe)
        assert isinstance(vardps, DiffractionVariance2D)

        mean_dp = np.array(
            [
                [0.545, 0.535, 0.4425, 0.5325, 0.72, 0.26, 0.71, 0.5025],
                [0.47, 0.5775, 0.105, 0.575, 0.17, 0.385, 0.275, 0.3575],
                [0.54, 0.5475, 0.4925, 0.6475, 0.405, 0.475, 0.1975, 0.2625],
                [0.63, 0.4525, 0.36, 0.81, 0.5475, 0.5675, 0.5125, 0.51],
                [0.4425, 0.5675, 0.58, 0.5975, 0.6275, 0.53, 0.71, 0.545],
                [0.5875, 0.475, 0.6425, 0.55, 0.5925, 0.4875, 0.1175, 0.5675],
                [0.775, 0.2225, 0.565, 0.21, 0.59, 0.4325, 0.48, 0.35],
                [0.5875, 0.5075, 0.4825, 0.475, 0.4125, 0.725, 0.53, 0.4275],
            ]
        ).reshape(8, 8)
        meansq_dp = np.array(
            [
                [
                    0.37395,
                    0.30455,
                    0.345475,
                    0.311925,
                    0.53905,
                    0.13485,
                    0.52095,
                    0.400125,
                ],
                [
                    0.27975,
                    0.387275,
                    0.01945,
                    0.36365,
                    0.03895,
                    0.21995,
                    0.09045,
                    0.154225,
                ],
                [
                    0.32825,
                    0.412725,
                    0.326975,
                    0.449175,
                    0.20705,
                    0.34915,
                    0.050025,
                    0.107525,
                ],
                [
                    0.45825,
                    0.282375,
                    0.20145,
                    0.67705,
                    0.399825,
                    0.493025,
                    0.375375,
                    0.30195,
                ],
                [
                    0.233775,
                    0.361175,
                    0.3989,
                    0.454775,
                    0.439125,
                    0.38445,
                    0.52965,
                    0.40045,
                ],
                [
                    0.488175,
                    0.24635,
                    0.472825,
                    0.35,
                    0.461275,
                    0.306125,
                    0.019525,
                    0.401725,
                ],
                [0.62355, 0.064425, 0.3697, 0.05265, 0.47075, 0.212875, 0.2721, 0.1716],
                [
                    0.430275,
                    0.341575,
                    0.264825,
                    0.28025,
                    0.267525,
                    0.54245,
                    0.34575,
                    0.258375,
                ],
            ]
        ).reshape(8, 8)
        var_dp = np.array(
            [
                [
                    0.25898493,
                    0.06402306,
                    0.76437167,
                    0.10004629,
                    0.0398341,
                    0.99482249,
                    0.03342591,
                    0.58461424,
                ],
                [
                    0.26641014,
                    0.16122262,
                    0.76417234,
                    0.09988658,
                    0.34775087,
                    0.48389273,
                    0.19603306,
                    0.20670937,
                ],
                [
                    0.12568587,
                    0.37686871,
                    0.34803783,
                    0.07136149,
                    0.26230758,
                    0.54747922,
                    0.28248678,
                    0.56045351,
                ],
                [
                    0.15457294,
                    0.37907878,
                    0.55439815,
                    0.03193111,
                    0.33383374,
                    0.53086611,
                    0.42914932,
                    0.16089965,
                ],
                [
                    0.1939098,
                    0.12146558,
                    0.18579073,
                    0.27385725,
                    0.11522039,
                    0.36863653,
                    0.05068439,
                    0.34820301,
                ],
                [
                    0.41435944,
                    0.09185596,
                    0.14539206,
                    0.15702479,
                    0.31396322,
                    0.28809993,
                    0.41421458,
                    0.24737526,
                ],
                [
                    0.03816857,
                    0.30135084,
                    0.15811732,
                    0.19387755,
                    0.35234128,
                    0.13802666,
                    0.18098958,
                    0.40081633,
                ],
                [
                    0.24660933,
                    0.32621515,
                    0.13753389,
                    0.24210526,
                    0.5722314,
                    0.03200951,
                    0.23086508,
                    0.41376834,
                ],
            ]
        ).reshape(8, 8)
        corr_var_dp = var_dp - np.divide(dqe, mean_dp)
        corr_var_dp[np.isinf(corr_var_dp)] = 0
        corr_var_dp[np.isnan(corr_var_dp)] = 0

        assert np.allclose(vardps.data[0, 0], mean_dp, atol=1e-6)
        assert np.allclose(vardps.data[0, 1], meansq_dp, atol=1e-6)
        assert np.allclose(vardps.data[1, 0], var_dp, atol=1e-14, equal_nan=True)
        assert np.allclose(vardps.data[1, 1], corr_var_dp, atol=1e-14, equal_nan=True)

    def test_set_data_type(self):
        # set 8 bit data type which will result in an incorrect mean square
        # pattern if not changed.

        dp_array = np.array(
            [
                [[[10, 10], [50, 30]], [[8, 20], [50, 30]]],
                [[[12, 30], [0, 30]], [[10, 10], [50, 30]]],
            ]
        ).astype(np.uint8)

        variance_test_diffraction_pattern = ElectronDiffraction2D(dp_array)
        vargen = VarianceGenerator(variance_test_diffraction_pattern)

        vardps_8 = vargen.get_diffraction_variance(dqe=1)
        assert isinstance(vardps_8, DiffractionVariance2D)
        corr_var_dp_8 = np.array([[-0.08, -0.66857143], [-0.92213333, -0.88666667]])
        assert np.allclose(
            vardps_8.data[1, 1], corr_var_dp_8, atol=1e-6, equal_nan=True
        )

        vardps_16 = vargen.get_diffraction_variance(dqe=1, set_data_type=np.uint16)
        assert isinstance(vardps_16, DiffractionVariance2D)
        corr_var_dp_16 = np.array([[-0.08, 0.16734694], [0.30666667, -0.0333333]])

        assert np.allclose(
            vardps_16.data[1, 1], corr_var_dp_16, atol=1e-6, equal_nan=True
        )

    @pytest.mark.parametrize("dqe", [0.5, 0.6])
    def test_get_image_variance(self, variance_generator: VarianceGenerator, dqe):

        varims = variance_generator.get_image_variance(dqe)
        assert isinstance(varims, ImageVariance)

        mean_im = np.array([[0.51765625, 0.504375], [0.47625, 0.47125]]).reshape(2, 2)
        meansq_im = np.array(
            [[0.34353281, 0.33261875], [0.31661875, 0.3085875]]
        ).reshape(2, 2)
        var_im = np.array([[0.28199196, 0.30749375], [0.39593968, 0.38955456]]).reshape(
            2, 2
        )
        corr_var_im = var_im - np.divide(dqe, mean_im)

        assert np.allclose(varims.data[0, 0], mean_im, atol=1e-6)
        assert np.allclose(varims.data[0, 1], meansq_im, atol=1e-6)
        assert np.allclose(varims.data[1, 0], var_im, atol=1e-14)
        assert np.allclose(varims.data[1, 1], corr_var_im, atol=1e-14)
