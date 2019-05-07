# Copyright 2017-2019 The pyXem developers
# -*- coding: utf-8 -*-
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

"""
utils to support SubpixelrefinementGenerator
"""

import numpy as np
from skimage.feature import register_translation
from skimage import draw
from skimage.transform import rescale


def get_experimental_square(z, vector, square_size):
    """Defines a square region around a given diffraction vector and returns an
    upsampled copy.

    Parameters
    ----------
    z : np.array()
        Single diffraction pattern
    vector : np.array()
        Single vector in pixels (int) [x,y] with top left as [0,0]
    square_size : int
        The length of one side of the bounding square (must be even)

    Returns
    -------
    square : np.array()
        Of size (L,L) where L = square_size

    """
    if square_size % 2 != 0:
        raise ValueError("'square_size' must be an even number")

    cx, cy, half_ss = vector[0], vector[1], int(square_size / 2)
    # select square with correct x,y see PR for details
    _z = z[cy - half_ss:cy + half_ss, cx - half_ss:cx + half_ss]
    return _z


def get_simulated_disc(square_size, disc_radius):
    """Create a uniform disc for correlating with the experimental square.

    Parameters
    ----------
    square size : int
        (even) - size of the bounding box
    disc_radius : int
        radius of the disc

    Returns
    -------
    arr: np.array()
        Upsampled copy of the simulated disc as a numpy array

    """

    if square_size % 2 != 0:
        raise ValueError("'square_size' must be an even number")

    ss = int(square_size)
    arr = np.zeros((ss, ss))
    rr, cc = draw.circle(int(ss / 2), int(ss / 2), radius=disc_radius,
                         shape=arr.shape)  # is the thin disc a good idea
    arr[rr, cc] = 1
    return arr


def _conventional_xc(exp_disc, sim_disc, upsample_factor):
    """Takes two images of disc and finds the shift between them using
    conventional (phase) cross correlation.

    Parameters
    ----------
    exp_disc : np.array()
        A numpy array of the "experimental" disc
    sim_disc : np.array()
        A numpy array of the disc used as a template
    upsample_factor: int (must be even)
        Factor to upsample by, reciprocal of the subpixel resolution
        (eg 10 ==> 1/10th of a pixel)

    Returns
    -------
    shifts
        Pixel shifts required to register the two images

    """

    shifts, error, _ = register_translation(exp_disc, sim_disc, upsample_factor)
    return shifts
