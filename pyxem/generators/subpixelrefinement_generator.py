# -*- coding: utf-8 -*-
# Copyright 2016-2024 The pyXem developers
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

"""Generating subpixel resolution on diffraction vectors."""

import numpy as np
from skimage.registration import phase_cross_correlation
from skimage import draw

from pyxem.signals import DiffractionVectors


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
    rr, cc = draw.disk(
        (int(ss / 2), int(ss / 2)), radius=disc_radius, shape=arr.shape
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
        if vectors.shape == (1,) and vectors.dtype == object:
            vectors = vectors[0]
        return np.floor((vectors.astype(np.float64) / calibration) + center).astype(int)

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

    dp_max_size = np.max(dp.data.shape) - 1
    if isinstance(vector_pixels, DiffractionVectors):
        min_value, max_value = 100, -100
        for index in np.ndindex(dp.axes_manager.navigation_shape):
            islice = np.s_[index]
            vec = vector_pixels.inav[islice].data[0]
            if len(vec) == 0:
                temp_min = 0
                temp_max = 0
            else:
                temp_min = vec.flatten().min()
                temp_max = vec.flatten().max()
            if min_value > temp_min:
                min_value = temp_min
            if max_value < temp_max:
                max_value = temp_max
        if max_value > dp_max_size or (min_value < 0):
            raise ValueError(
                "Some of your vectors do not lie within your "
                "diffraction pattern, check your calibration"
            )
    elif isinstance(vector_pixels, np.ndarray):
        if np.any(vector_pixels > dp_max_size) or np.any(vector_pixels < 0):
            raise ValueError(
                "Some of your vectors do not lie within your "
                "diffraction pattern, check your calibration"
            )
    else:
        raise ValueError(
            "The vectors are not recognized as either DiffractionVectors "
            "or a NumPy array"
        )
    return vector_pixels


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

    shifts, error, _ = phase_cross_correlation(
        exp_disc, sim_disc, upsample_factor=upsample_factor
    )
    shifts = np.flip(shifts)  # to comply with hyperspy conventions - see issue#490
    return shifts


def _center_of_mass_hs(z):
    """Return the center of mass of an array with coordinates in the
    hyperspy convention

    Parameters
    ----------
    z : np.array

    Returns
    -------
    (x,y) : tuple of floats
        The x and y locations of the center of mass of the parsed square
    """

    s = np.sum(z)
    if s != 0:
        z *= 1 / s
    dx = np.sum(z, axis=0)
    dy = np.sum(z, axis=1)
    h, w = z.shape
    cx = np.sum(dx * np.arange(w))
    cy = np.sum(dy * np.arange(h))
    return cx, cy


def _com_experimental_square(z, vector, square_size):
    """Wrapper for get_experimental_square that makes the non-zero
    elements symmetrical around the 'unsubpixeled' peak by zeroing a
    'spare' row and column (top and left).

    Parameters
    ----------
    z : np.array

    vector : np.array([x,y])

    square_size : int (even)

    Returns
    -------
    z_adpt : np.array
        z, but with row and column zero set to 0
    """
    # Copy to make sure we don't change the dp
    z_adpt = np.copy(get_experimental_square(z, vector=vector, square_size=square_size))
    z_adpt[:, 0] = 0
    z_adpt[0, :] = 0
    return z_adpt


def _center_of_mass_map(dp, vectors, square_size, center, calibration, columns=None):
    if vectors.shape == (2,):
        vectors = np.array(
            [
                vectors,
            ]
        )
    shifts = np.zeros_like(vectors, dtype=np.float64)
    if columns is not None:
        num_cols = vectors.shape[1]
        other_indexes = np.arange(num_cols)[
            np.logical_not(np.isin(np.arange(num_cols), columns))
        ]
        vectors = vectors[:, columns]
        extra_columns = vectors[:, other_indexes]

    for i, vector in enumerate(vectors):
        expt_disc = _com_experimental_square(dp, vector, square_size)
        shifts[i] = [a - square_size / 2 for a in _center_of_mass_hs(expt_disc)]
    new_vectors = ((vectors + shifts) - center) * calibration
    if columns is not None:
        new_vectors = np.hstack(new_vectors, extra_columns)
    return new_vectors
