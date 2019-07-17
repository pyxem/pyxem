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

from pyxem.utils.subpixel_refinements_utils import _conventional_xc
from pyxem.utils.subpixel_refinements_utils import get_experimental_square
from pyxem.utils.subpixel_refinements_utils import get_simulated_disc

from skimage.transform import rescale
from skimage import draw


@pytest.fixture()
def exp_disc():
    ss, disc_radius, upsample_factor = int(60), 6, 10

    arr = np.zeros((ss, ss))
    rr, cc = draw.circle(int(ss / 2) + 20, int(ss / 2) - 10, radius=disc_radius, shape=arr.shape)
    arr[rr, cc] = 1
    return arr


@pytest.fixture()
def sim_disc():
    return get_simulated_disc(60, 5)


@pytest.fixture()
def upsample_factor():
    return int(10)


@pytest.mark.filterwarnings('ignore::UserWarning')  # various skimage warnings
def test___conventional_xc(exp_disc, sim_disc, upsample_factor):
    # this work (and measures) on the upsampled versions of the images
    s = _conventional_xc(exp_disc, sim_disc, upsample_factor)
    error = np.subtract(s, np.asarray([20, -10]))
    rms = np.sqrt(error[0]**2 + error[1]**2)
    assert rms < 1  # which corresponds to a 10th of a pixel


@pytest.mark.filterwarnings('ignore::UserWarning')  # various skimage warnings
def test_get_experimental_square(exp_disc):
    square = get_experimental_square(exp_disc, [17, 19], 6)
    assert square.shape[0] == int(6)
    assert square.shape[1] == int(6)


@pytest.mark.xfail(strict=True)
def test_non_even_errors_get_simulated_disc():
    disc = get_simulated_disc(61, 5)


@pytest.mark.xfail(strict=True)
def test_non_even_errors_get_experimental_errors(exp_disc):
    square = get_experimental_square(exp_disc, [17, 19], 7)
