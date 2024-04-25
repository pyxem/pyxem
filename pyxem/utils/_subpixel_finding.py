# -*- coding: utf-8 -*-
# Copyright 2016-2023 The pyXem developers
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

"""Utils for subpixel vectors refinement."""

import numpy as np
from skimage.transform import rescale
from skimage import draw

from pyxem.utils.diffraction import normalize_template_match


def _get_experimental_square(z, vector, square_size):
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
    # reverse the order of the vector
    cx, cy, half_ss = int(vector[1]), int(vector[0]), int(square_size / 2)
    _z = z[cy - half_ss : cy + half_ss + 1, cx - half_ss : cx + half_ss + 1]
    return _z


def _get_simulated_disc(square_size, disc_radius):
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
    return cy, cx


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
    z_adpt = np.copy(
        _get_experimental_square(z, vector=vector, square_size=square_size)
    )
    z_adpt[:, 0] = 0
    z_adpt[0, :] = 0
    return z_adpt


def _conventional_xc(slic, kernel, upsample_factor):
    """Takes two images of disc and finds the shift between them using
    conventional cross correlation.
    """
    half_ss = int((slic.shape[0]) / 2)
    slic = rescale(slic, upsample_factor, order=1, mode="reflect")
    kernel = rescale(kernel, upsample_factor, order=1, mode="reflect")

    temp = normalize_template_match(slic, kernel)
    max = np.array(np.unravel_index(np.argmax(temp), temp.shape))
    shifts = np.array(max) / upsample_factor - half_ss
    return shifts


#####################################################
# Methods for subpixel refinement on a set of vectors
#####################################################


def _center_of_mass_map(dp, vectors, square_size, offsets, scales):
    shifts = np.zeros_like(vectors, dtype=np.float64)
    for i, vector in enumerate(vectors):
        square = _get_experimental_square(dp, vector, square_size)
        shifts[i] = [a - square_size / 2 for a in _center_of_mass_hs(square)]

    new_vectors = (vectors + shifts) * scales + offsets
    return new_vectors


def _conventional_xc_map(
    dp,
    vectors,
    kernel,
    square_size,
    upsample_factor,
    offsets,
    scales,
):
    vectors = np.array(vectors).astype(int)
    shifts = np.zeros_like(vectors, dtype=np.float64)
    for i, vector in enumerate(vectors):
        expt_disc = _get_experimental_square(dp, vector, square_size)
        shifts[i] = _conventional_xc(expt_disc, kernel, upsample_factor)

    return (vectors + shifts) * scales + offsets


def _wrap_columns(dp, vectors, f, columns=None, **kwargs):
    """
    Take some function, f, and apply it to the 2d array X returning a list
    of peak positions.  The intensity at each peak position is then appended
    to the returned list of peaks

    Parameters
    ----------
    X: 2-D array-like
        The input image used to find peaks
    f: func
        The passed function to find peaks
    kwargs:
        Any additional keyword arguments passed to f

    Returns
    -------
        peaks:array-like
            A 2d array with columns [x, y, intensity]

    """
    if vectors.shape == (2,):
        vectors = np.array(
            [
                vectors,
            ]
        )
    if columns is not None:
        num_cols = vectors.shape[1]
        other_indexes = np.arange(num_cols)[
            np.logical_not(np.isin(np.arange(num_cols), columns))
        ]
        extra_columns = vectors[:, other_indexes]
        vectors = vectors[:, columns]
    new_vectors = f(dp, vectors[:, columns], **kwargs)
    if columns is not None:
        new_vectors = np.hstack((new_vectors, extra_columns))
    return new_vectors
