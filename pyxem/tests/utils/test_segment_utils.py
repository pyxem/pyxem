# -*- coding: utf-8 -*-
# Copyright 2019 The pyXem developers
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

from pyxem.utils.segment_utils import (
    norm_cross_corr,
    separate_watershed,
    get_gaussian2d,
)


@pytest.mark.parametrize(
    "img, template, corr_expt",
    [
        (
            np.array([[0.1, 50.0, 0.0], [0.0, 0.0, 44], [0.3, 37, 11.1]]),
            np.array([[0.2, 0.0, 1.4], [10.9, 22.0, 15.0], [0.28, 36, 0.36]]),
            0.28918685246563913,
        ),
        (
            np.array([[1, 9, 5], [42, 42, 44], [6, 37, 11]]),
            np.array([[0, 1, 0], [1, 0, 0], [40, 90, 35]]),
            0.075079495399324,
        ),
        (
            np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0]]),
            np.array([[0, 1, 0], [1, 0, 0], [0, 1, 1]]),
            -1.0,
        ),
        (
            np.array([[1, 9, 5], [42, 42, 44], [6, 37, 11]]),
            np.array([[1, 9, 5], [42, 42, 44], [6, 37, 11]]),
            1.0,
        ),
    ],
)
def test_norm_cross_corr(img, template, corr_expt):
    c = norm_cross_corr(img, template)
    np.testing.assert_allclose(c, corr_expt)


@pytest.fixture
def vdf_image():
    stest = np.zeros((7, 6))
    stest[1:3, :2] = 1
    stest[2:5, 1:3] = 2
    stest[1:3, 4:] = 3
    stest[4, 4:] = 4
    stest[6:, :] = 5
    return stest


@pytest.mark.parametrize(
    "min_distance, min_size, max_size, "
    "max_number_of_grains, exclude_border, threshold, "
    "sep_shape_expt",
    [
        (1, 1, np.inf, np.inf, 0, True, (4, 6, 7)),
        (1, 1, np.inf, np.inf, 1, False, (1, 6, 7)),
        (1, 5, np.inf, np.inf, 0, True, (2, 6, 7)),
        (1, 1, np.inf, 4, 0, True, (3, 6, 7)),
        (1, 1, 3, np.inf, 0, True, (1, 6, 7)),
        (3, 1, np.inf, np.inf, 0, True, (1, 6, 7)),
    ],
)
def test_separate_watershed(
    vdf_image,
    min_distance,
    min_size,
    max_size,
    max_number_of_grains,
    exclude_border,
    threshold,
    sep_shape_expt,
):
    sep = separate_watershed(
        vdf_image,
        min_distance=min_distance,
        min_size=min_size,
        max_size=max_size,
        max_number_of_grains=max_number_of_grains,
        exclude_border=exclude_border,
        threshold=threshold,
    )
    # we don't care how many clusters we generate
    assert sep.shape[1:] == sep_shape_expt[1:]


@pytest.mark.parametrize(
    "a, xo, yo, x, y, sigma, gauss_shape_expt",
    [
        (10, 5, 5, np.arange(10), np.arange(10), 0.2, (10,)),
        (
            10,
            5,
            5,
            np.tile(np.arange(10), (5, 1)),
            np.tile(np.arange(10), (5, 1)),
            0.2,
            (5, 10),
        ),
    ],
)
def test_get_gaussian2d(a, xo, yo, x, y, sigma, gauss_shape_expt):
    gauss = get_gaussian2d(a, xo, yo, x, y, sigma)
    assert isinstance(gauss, np.ndarray)
    assert gauss.dtype == float
    np.testing.assert_equal(gauss.shape, gauss_shape_expt)
