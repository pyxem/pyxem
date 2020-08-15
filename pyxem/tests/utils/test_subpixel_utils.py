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

from pyxem.utils.subpixel_refinements_utils import get_experimental_square
from pyxem.utils.subpixel_refinements_utils import get_simulated_disc

from skimage import draw


@pytest.fixture()
def exp_disc():
    ss, disc_radius, upsample_factor = int(60), 6, 10

    arr = np.zeros((ss, ss))
    rr, cc = draw.circle(
        int(ss / 2) + 20, int(ss / 2) - 10, radius=disc_radius, shape=arr.shape
    )
    arr[rr, cc] = 1
    return arr


@pytest.mark.filterwarnings("ignore::UserWarning")  # various skimage warnings
def test_experimental_square_size(exp_disc):
    square = get_experimental_square(exp_disc, [17, 19], 6)
    assert square.shape[0] == int(6)
    assert square.shape[1] == int(6)


def test_failure_for_non_even_entry_to_get_simulated_disc():
    with pytest.raises(ValueError, match="'square_size' must be an even number"):
        disc = get_simulated_disc(61, 5)


def test_failure_for_non_even_errors_get_experimental_square(exp_disc):
    with pytest.raises(ValueError, match="'square_size' must be an even number"):
        square = get_experimental_square(exp_disc, [17, 19], 7)
