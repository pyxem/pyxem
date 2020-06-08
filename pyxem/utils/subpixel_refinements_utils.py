# Copyright 2016-2020 The pyXem developers
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
from skimage import draw
from skimage.transform import rescale
from pyxem.signals.diffraction_vectors import DiffractionVectors


def get_experimental_square(z, vector, square_size):
    """Defines a square region around a given diffraction vector and returns.

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
    _z = z[cy - half_ss : cy + half_ss, cx - half_ss : cx + half_ss]
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
    rr, cc = draw.circle(
        int(ss / 2), int(ss / 2), radius=disc_radius, shape=arr.shape
    )  # is the thin disc a good idea
    arr[rr, cc] = 1
    return arr


def _get_pixel_vectors(dp, vectors, calibration, center):
    """Get the pixel coordinates for the given diffraction
    pattern and vectors.

    Parameters
    ----------
    dp: :obj:`pyxem.signals.ElectronDiffraction2D`
        Instance of ElectronDiffraction2D
    vectors : :obj:`pyxem.signals.diffraction_vectors.DiffractionVectors`
        List of diffraction vectors
    calibration : [float, float]
        Calibration values
    center : float, float
        Image origin in pixel coordinates

    Returns
    -------
    vector_pixels : :obj:`pyxem.signals.diffraction_vectors.DiffractionVectors`
        Pixel coordinates for given diffraction pattern and vectors.
    """

    def _floor(vectors, calibration, center):
        if vectors.shape == (1,) and vectors.dtype == np.object:
            vectors = vectors[0]
        return np.floor((vectors.astype(np.float64) / calibration) + center).astype(
            np.int
        )

    if isinstance(vectors, DiffractionVectors):
        if vectors.axes_manager.navigation_shape != dp.axes_manager.navigation_shape:
            raise ValueError(
                "Vectors with shape {} must have the same navigation shape "
                "as the diffraction patterns which has shape {}.".format(
                    vectors.axes_manager.navigation_shape,
                    dp.axes_manager.navigation_shape,
                )
            )
        vector_pixels = vectors.map(
            _floor, calibration=calibration, center=center, inplace=False
        )
    else:
        vector_pixels = _floor(vectors, calibration, center)

    if isinstance(vector_pixels, DiffractionVectors):
        if np.any(vector_pixels.data > (np.max(dp.data.shape) - 1)) or (
            np.any(vector_pixels.data < 0)
        ):
            raise ValueError(
                "Some of your vectors do not lie within your diffraction pattern, check your calibration"
            )
    elif isinstance(vector_pixels, np.ndarray):
        if np.any((vector_pixels > np.max(dp.data.shape) - 1)) or (
            np.any(vector_pixels < 0)
        ):
            raise ValueError(
                "Some of your vectors do not lie within your diffraction pattern, check your calibration"
            )

    return vector_pixels
