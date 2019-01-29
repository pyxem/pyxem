# -*- coding: utf-8 -*-
# Copyright 2017-2018 The pyXem developers
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


def get_experimental_square(z, vector, half_square_size):
    """
    'Cuts out' a region around a given diffraction vector and returns it.

    Parameters
    ----------
    z : np.array()
        Single diffraction pattern
    vector : np.array()
        Single vector to be cut out, in pixels (int) [x,y] with top left as [0,0]
    half_square_size : int
        The length of half of one side of the bounding square. eg 1 will return 3x3

    Returns
    -------
    square : np.array()
        Of size (2L+1,2L+1) where L = square_size

    """

    cx, cy, r= vector[0], vector[1], int(half_square_size)
    # select square with correct x,y see PR for details
    _z = z[cy - r :cy + r + 1, cx - r:cx + r + 1]
    return _z


def get_simulated_disc(half_square_size, disc_radius):
    """
    Create a uniform disc for correlating with the experimental square

    Parameters
    ----------
    half_square_size : int
        The length of half of one side of the bounding square. eg 1 will return 3x3
    disc_radius : int
        radius of the disc

    Returns
    -------

    arr: np.array()
        Upsampled copy of the simulated disc as a numpy array

    """

    ss = int(2*half_square_size+1)
    arr = np.zeros((ss, ss))
    rr, cc = draw.circle(int(ss / 2), int(ss / 2), radius=disc_radius,
                         shape=arr.shape)  # is the thin disc a good idea
    arr[rr, cc] = 1
    return arr


def _conventional_xc(exp_disc, sim_disc, upsample_factor):
    """
    Takes two images of disc and finds the shift between them using conventional (phase) cross correlation

    Parameters
    ----------
    exp_disc : np.array()
        A numpy array of the "experimental" disc
    sim_disc : np.array()
        A numpy array of the disc used as a template
    upsample_factor: int (must be even)
        Factor to upsample by, reciprocal of the subpixel resolution (eg 10 ==> 1/10th of a pixel)

    Returns
    -------
    shifts
        Pixel shifts required to register the two images
    """

    shifts, error, _ = register_translation(exp_disc, sim_disc, upsample_factor)
    return shifts
