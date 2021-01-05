# -*- coding: utf-8 -*-
# Copyright 2016-2020 The pyXem developers
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
