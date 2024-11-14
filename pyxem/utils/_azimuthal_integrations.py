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

"""Utils for azimuthal integration."""

import numpy as np

from shapely import Polygon, box
import shapely
from numba import cuda, prange
import numba

from pyxem import CUPY_INSTALLED

if CUPY_INSTALLED:
    import cupy as cp


@numba.njit(parallel=True, nogil=True)
def _slice_radial_integrate(
    img,
    factors,
    factors_slice,
    slices,
    npt_rad,
    npt_azim,
    mask=None,
    mean=False,
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
    val = np.empty((npt_rad, npt_azim))
    for i in prange(len(factors_slice)):
        ii, jj = i // npt_azim, i % npt_azim
        if mean:  # divide by the total number of pixels
            val[ii, jj] = np.sum(
                img[slices[i][0] : slices[i][2], slices[i][1] : slices[i][3]]
                * factors[factors_slice[i][0] : factors_slice[i][1]].reshape(
                    (slices[i][2] - slices[i][0], slices[i][3] - slices[i][1])
                )
            ) / np.sum(
                factors[factors_slice[i][0] : factors_slice[i][1]].reshape(
                    (slices[i][2] - slices[i][0], slices[i][3] - slices[i][1])
                )
            )
        else:
            val[ii, jj] = np.sum(
                img[slices[i][0] : slices[i][2], slices[i][1] : slices[i][3]]
                * factors[factors_slice[i][0] : factors_slice[i][1]].reshape(
                    (slices[i][2] - slices[i][0], slices[i][3] - slices[i][1])
                )
            )
    return val


def _slice_radial_integrate_cupy(
    img, factors, factors_slice, slices, mask, npt, npt_azim
):
    original_nav = img.shape[:-2]
    img = img.reshape((-1,) + img.shape[-2:])
    val = cp.empty((img.shape[0], npt, npt_azim))
    if mask is None:
        mask = cp.zeros((img.shape[-2:]))
    __slice_radial_integrate_cupy[(img.shape[0], npt), (npt_azim)](
        img, factors, factors_slice, slices, npt_azim, mask, val
    )
    val = val.reshape(original_nav + (npt, npt_azim))
    return val


@cuda.jit
def __slice_radial_integrate_cupy(
    img, factors, factors_slice, slices, npt_azim, mask, val
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
    val:
        The array to store the result in
    Note
    ----
    This function is run by every single thread once!
    """
    tx = cuda.threadIdx.x  # current thread (azimuthal)
    bx = cuda.blockIdx.x  # Current block (navigation flattened)
    by = cuda.blockIdx.y  # Current block (radial)
    pos = cuda.grid(1)  # current thread
    index = tx + npt_azim * by
    if pos < val.size:  # account for slices out of range!
        factors_ind = factors_slice[index]
        current_slice = slices[index]
        sum = 0
        ind = 0
        for i in range(current_slice[0], current_slice[2]):
            for j in range(current_slice[1], current_slice[3]):
                is_mask = not mask[i, j]
                sum += factors[ind + factors_ind[0]] * img[bx, i, j] * is_mask
                ind += 1
        val[bx, by, tx] = sum
    return


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
    This function is much faster with numba than without. Additionally,  a GPU version of
    this function is not implemented because it is a bit more complicated than the 2D
    version and doesn't perform well using the `map` function.
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
            total_f = np.finfo(np.float32).eps
            if mask is not None:
                for index, fa in zip(ind, f):
                    if not mask[index[0], index[1]]:
                        total_f = total_f + fa
            else:
                for index, fa in zip(ind, f):
                    total_f = total_f + fa
            total = total / total_f
        ans[i] = total
    return ans


def _get_factors(control_points, slices, pixel_extents):
    """This function takes a set of control points (vertices of bounding polygons) and
    slices (min and max indices for each control point) and returns the factors for
    each slice. The factors are the area of the intersection of the polygon and the
    sliced pixels.
    """
    all_boxes = get_boxes(slices, pixel_extent=pixel_extents)
    max_num = np.max([len(x) for x in all_boxes])
    num_box = len(all_boxes)
    boxes = shapely.empty((num_box, max_num))

    p = shapely.polygons(control_points)
    for i, bx in enumerate(all_boxes):
        try:
            b = shapely.box(bx[:, 0], bx[:, 1], bx[:, 2], bx[:, 3])
            boxes[i, : len(b)] = b
        except IndexError:  # the box is empty.
            pass

    factors = shapely.area(
        shapely.intersection(boxes, p[:, np.newaxis])
    ) / shapely.area(boxes)
    not_nan = np.logical_not(np.isnan(factors))

    factors = factors.flatten()
    factors = factors[not_nan.flatten()]

    num = np.sum(not_nan, axis=1)
    factors_slice = np.cumsum(num)
    factors_slice = np.hstack(([0], factors_slice))
    factors_slice = np.stack((factors_slice[:-1], factors_slice[1:])).T
    return factors, factors_slice


def get_boxes(slices, pixel_extent):
    all_boxes = []
    x_extent, y_extent = pixel_extent
    x_ext_left, x_ext_right = x_extent
    y_ext_left, y_ext_right = y_extent
    for sl in slices:
        x_edges = list(range(sl[0], sl[2]))
        y_edges = list(range(sl[1], sl[3]))
        boxes = []
        for i, x in enumerate(x_edges):
            for j, y in enumerate(y_edges):
                b = [
                    x_ext_left[x],
                    y_ext_left[y],
                    x_ext_right[x],
                    y_ext_right[y],
                ]
                boxes.append(b)
        all_boxes.append(np.array(boxes))
    return all_boxes


def _get_control_points(npt, npt_azim, radial_range, azimuthal_range, affine):
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
    azimuthal_range: (float, float)
        The azumuthal range of the data, in radians

    Returns
    -------
    control_points: (npt_azim*npt, 4, 2)
        The cartesian coordinates of the control points of the polygon for each azimuthal pixel.

    """
    r = np.linspace(radial_range[0], radial_range[1], npt + 1)
    phi = np.linspace(azimuthal_range[0], azimuthal_range[1], npt_azim + 1)
    control_points = np.empty(((len(r) - 1) * (len(phi) - 1), 4, 2))
    # lower left
    control_points[:, 0, 0] = (np.cos(phi[:-1]) * r[:-1][:, np.newaxis]).ravel()
    control_points[:, 0, 1] = (np.sin(phi[:-1]) * r[:-1][:, np.newaxis]).ravel()
    # lower right
    control_points[:, 1, 0] = (np.cos(phi[1:]) * r[:-1][:, np.newaxis]).ravel()
    control_points[:, 1, 1] = (np.sin(phi[1:]) * r[:-1][:, np.newaxis]).ravel()
    # upper left
    control_points[:, 2, 0] = (np.cos(phi[1:]) * r[1:][:, np.newaxis]).ravel()
    control_points[:, 2, 1] = (np.sin(phi[1:]) * r[1:][:, np.newaxis]).ravel()
    # upper right
    control_points[:, 3, 0] = (np.cos(phi[:-1]) * r[1:][:, np.newaxis]).ravel()
    control_points[:, 3, 1] = (np.sin(phi[:-1]) * r[1:][:, np.newaxis]).ravel()

    # apply the affine transformation to the control points
    if affine is not None:
        affine[0, 1] = -affine[0, 1]  # changing the rotation direction
        affine[1, 0] = -affine[1, 0]
        control_points = np.dot(control_points, affine[:2, :2])
    return control_points
