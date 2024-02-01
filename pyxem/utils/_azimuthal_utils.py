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


import numpy as np

from shapely import Polygon, box
import shapely
from numba import cuda

try:
    import cupy

    CUPY_INSTALLED = True
except ImportError:
    CUPY_INSTALLED = False


@numba.njit
def _slice_radial_integrate(
    img,
    factors,
    factors_slice,
    slices,
    npt_rad,
    npt_azim,
    mask=None,
):  # pragma: no cover
    """Slice the image into small chunks and multiply by the factors.

    Parameters
    ----------
    img: np.array
        The image to be sliced
    factors:
        The factors to multiply the slices by
    slices:
        The slices to slice the image by
    npt_rad:
        The number of radial points
    npt_azim:
        The number of azimuthal points

    Note
    ----
    This function is much faster with numba than without. There is probably a factor
    of 2-10 speedup that could be achieved  by using cython or c++ instead of python

    """
    if mask is not None:
        img = img * np.logical_not(mask)
    val = np.empty(slices.shape[0])
    for i, (s, f) in enumerate(zip(slices, factors_slice)):
        val[i] = np.sum(
            img[s[0] : s[2], s[1] : s[3]]
            * factors[f[0] : f[1]].reshape((s[2] - s[0], s[3] - s[1]))
        )
    return val.reshape((npt_azim, npt_rad)).T


@cuda.jit
def _slice_radial_integrate_cupy(
    img, factors, factors_slice, slices, npt_rad, npt_azim, val
):
    """Slice the image into small chunks and multiply by the factors.
    Parameters
    ----------
    img: np.array
        The image to be sliced
    factors:
        The factors to multiply the slices by
    slices:
        The slices to slice the image by
    npt_rad:
        The number of radial points
    npt_azim:
        The number of azimuthal points
    Note
    ----
    This function is run by every single thread once!
    """
    tx = cuda.threadIdx.x  # current thread
    bx = cuda.blockIdx.x  # Current block
    bw = cuda.blockDim.x  # Should be equal to blocks!
    x = tx + bx * bw
    if x < val.shape[0]:  # account for slices out of range!
        factors_ind = factors_slice[x]
        current_slice = slices[x]
        sum = 0
        ind = 0
        for i in range(current_slice[0], current_slice[2]):
            for j in range(current_slice[1], current_slice[3]):
                sum += factors[ind + factors_ind[0]] * img[i, j]
                ind += 1
        val[x] = sum
    return


def slice_radial_integrate_cupy(image, factors, factor_slices, slices):
    blocks = 141
    threads = 256
    val = cupy.empty(slices.shape[0])
    _slice_radial_integrate_cupy[blocks, threads](
        image, factors, factor_slices, slices, 100, 360, val
    )
    val_reshaped = val.reshape(360, 100).T
    del val
    return val_reshaped


@numba.njit
def _slice_radial_integrate1d(
    img, indexes, factors, factor_slices, mask=None, mean=False
):  # pragma: no cover
    """Slice the image into small chunks and multiply by the factors.

    Parameters
    ----------
    img: np.array
        The image to be sliced
    indexes:
        The indexes of the pixels to multiply by the `factors`
    factors:
        The percentage of the pixel for each radial bin associated with some index
    factor_slices:
        The slices to slice the factors and the indexes by
    mask:
        The mask to apply to the image
    mean:
        If True, return the mean of the pixels in the slice rather than the sum

    Note
    ----
    This function is much faster with numba than without. There is probably a factor
    of 2-10 speedup that could be achieved  by using cython or c++ instead of python

    """
    if mask is not None:
        img = img * np.logical_not(mask)
    ans = np.empty(len(factor_slices) - 1)
    for i in range(len(factor_slices) - 1):
        ind = indexes[factor_slices[i] : factor_slices[i + 1]]
        f = factors[factor_slices[i] : factor_slices[i + 1]]
        total = 0.0
        for index, fa in zip(ind, f):
            total = total + img[index[0], index[1]] * fa
        if mean:
            total_f = 0.0
            if mask is not None:
                total_f = total_f + mask[ind[0], ind[1]] * fa
            else:
                for index, fa in zip(ind, f):
                    total_f = total_f + fa
            total = total / total_f
        ans[i] = total
    return ans


def _get_factors(control_points, slices, axes):
    """This function takes a set of control points (vertices of bounding polygons) and
    slices (min and max indices for each control point) and returns the factors for
    each slice. The factors are the area of the intersection of the polygon and the
    sliced pixels.
    """
    factors = []
    factors_slice = []
    start = 0
    for cp, sl in zip(control_points, slices):
        p = Polygon(cp)
        x_edges = list(range(sl[0], sl[2]))
        y_edges = list(range(sl[1], sl[3]))
        boxes = []
        for i, x in enumerate(x_edges):
            for j, y in enumerate(y_edges):
                b = box(axes[0][x], axes[1][y], axes[0][x + 1], axes[1][y + 1])
                boxes.append(b)
        factors += list(
            shapely.area(shapely.intersection(boxes, p)) / shapely.area(boxes)
        )
        factors_slice.append([start, start + len(boxes)])
        start += len(boxes)
    return np.array(factors), np.array(factors_slice)


def _get_control_points(npt, npt_azim, radial_range, affine):
    """Get the control points in the form of an array (npt_azim*npt, 4, 2) representing
    the cartesian coordinates of the control points for each azimuthal pixel.

    Parameters
    ----------
    npt: int
        The number of radial points
    npt_azim:
        The number of azimuthal points
    affine: (3x3)
        The affine transformation to apply to the data
    center: (float, float)
        The center of the diffraction pattern
    radial_range: (float, float)
        The radial range of the data

    Returns
    -------
    control_points: (npt_azim*npt, 4, 2)
        The cartesian coordinates of the control points of the polygon for each azimuthal pixel.

    """
    r = np.linspace(radial_range[0], radial_range[1], npt + 1)
    phi = np.linspace(0, 2 * np.pi, npt_azim + 1)
    control_points = np.empty(((len(r) - 1) * (len(phi) - 1), 4, 2))
    # lower left
    control_points[:, 0, 0] = (r[:-1] * np.cos(phi[:-1])[:, np.newaxis]).ravel()
    control_points[:, 0, 1] = (r[:-1] * np.sin(phi[:-1])[:, np.newaxis]).ravel()
    # lower right
    control_points[:, 1, 0] = (r[:-1] * np.cos(phi[1:])[:, np.newaxis]).ravel()
    control_points[:, 1, 1] = (r[:-1] * np.sin(phi[1:])[:, np.newaxis]).ravel()
    # upper left
    control_points[:, 2, 0] = (r[1:] * np.cos(phi[1:])[:, np.newaxis]).ravel()
    control_points[:, 2, 1] = (r[1:] * np.sin(phi[1:])[:, np.newaxis]).ravel()
    # upper right
    control_points[:, 3, 0] = (r[1:] * np.cos(phi[:-1])[:, np.newaxis]).ravel()
    control_points[:, 3, 1] = (r[1:] * np.sin(phi[:-1])[:, np.newaxis]).ravel()

    # apply the affine transformation to the control points
    if affine is not None:
        affine[0, 1] = -affine[0, 1]  # changing the rotation direction
        affine[1, 0] = -affine[1, 0]
        control_points = np.dot(control_points, affine[:2, :2])
    return control_points
