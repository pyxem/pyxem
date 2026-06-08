# -*- coding: utf-8 -*-
# Copyright 2016-2025 The pyXem developers
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
    img: ~numpy.ndarray
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
            total = np.sum(
                factors[factors_slice[i][0] : factors_slice[i][1]].reshape(
                    (slices[i][2] - slices[i][0], slices[i][3] - slices[i][1])
                )
            )
            if total == 0:
                val[ii, jj] = 0
            else:
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
    img: ~numpy.ndarray
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
    img: ~numpy.ndarray
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


@numba.njit(cache=True)
def _clip_poly_by_box(
    px, py, n_in, xmin, ymin, xmax, ymax, out_x, out_y
):  # pragma: no cover
    """Clip a polygon against an axis-aligned box using Sutherland-Hodgman.

    Parameters
    ----------
    px, py : float array
        Input polygon vertex coordinates (length >= n_in).
    n_in : int
        Number of input polygon vertices.
    xmin, ymin, xmax, ymax : float
        Axis-aligned clipping rectangle.
    out_x, out_y : float array
        Pre-allocated output vertex buffers (length >= 32).

    Returns
    -------
    int
        Number of vertices in the clipped polygon stored in out_x/out_y.
    """
    _MAX_V = 32
    tmp_x = np.empty(_MAX_V)
    tmp_y = np.empty(_MAX_V)

    # --- clip by x >= xmin ---
    m = 0
    for i in range(n_in):
        j = (i + 1) % n_in
        ci = px[i] >= xmin
        nj = px[j] >= xmin
        if ci:
            tmp_x[m] = px[i]
            tmp_y[m] = py[i]
            m += 1
        if ci != nj:
            dx = px[j] - px[i]
            if dx != 0.0:
                t = (xmin - px[i]) / dx
                tmp_x[m] = xmin
                tmp_y[m] = py[i] + t * (py[j] - py[i])
                m += 1
    if m == 0:
        return 0

    # --- clip by x <= xmax ---
    n2 = 0
    for i in range(m):
        j = (i + 1) % m
        ci = tmp_x[i] <= xmax
        nj = tmp_x[j] <= xmax
        if ci:
            out_x[n2] = tmp_x[i]
            out_y[n2] = tmp_y[i]
            n2 += 1
        if ci != nj:
            dx = tmp_x[j] - tmp_x[i]
            if dx != 0.0:
                t = (xmax - tmp_x[i]) / dx
                out_x[n2] = xmax
                out_y[n2] = tmp_y[i] + t * (tmp_y[j] - tmp_y[i])
                n2 += 1
    if n2 == 0:
        return 0

    # --- clip by y >= ymin ---
    m = 0
    for i in range(n2):
        j = (i + 1) % n2
        ci = out_y[i] >= ymin
        nj = out_y[j] >= ymin
        if ci:
            tmp_x[m] = out_x[i]
            tmp_y[m] = out_y[i]
            m += 1
        if ci != nj:
            dy = out_y[j] - out_y[i]
            if dy != 0.0:
                t = (ymin - out_y[i]) / dy
                tmp_x[m] = out_x[i] + t * (out_x[j] - out_x[i])
                tmp_y[m] = ymin
                m += 1
    if m == 0:
        return 0

    # --- clip by y <= ymax ---
    n3 = 0
    for i in range(m):
        j = (i + 1) % m
        ci = tmp_y[i] <= ymax
        nj = tmp_y[j] <= ymax
        if ci:
            out_x[n3] = tmp_x[i]
            out_y[n3] = tmp_y[i]
            n3 += 1
        if ci != nj:
            dy = tmp_y[j] - tmp_y[i]
            if dy != 0.0:
                t = (ymax - tmp_y[i]) / dy
                out_x[n3] = tmp_x[i] + t * (tmp_x[j] - tmp_x[i])
                out_y[n3] = ymax
                n3 += 1
    return n3


@numba.njit(cache=True)
def _poly_area_2d(vx, vy, n):  # pragma: no cover
    """Shoelace formula for the signed area of an n-vertex polygon."""
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vx[i] * vy[j] - vx[j] * vy[i]
    return abs(area) * 0.5


@numba.njit(cache=True, parallel=True)
def _compute_factors_numba(  # pragma: no cover
    control_points, slices, x_ext_left, x_ext_right, y_ext_left, y_ext_right
):
    """Parallel numba kernel that replaces the shapely intersection loop.

    Parameters
    ----------
    control_points : float64 array (N, 4, 2)
    slices : int64 array (N, 4)  –  [row_min, col_min, row_max, col_max]
    x_ext_left, x_ext_right : float64 1-D arrays  (pixel edges along the x axis)
    y_ext_left, y_ext_right : float64 1-D arrays  (pixel edges along the y axis)

    Returns
    -------
    factors : float64 1-D array
    factors_slice : int64 array (N, 2)
    """
    N = len(control_points)

    # Build prefix-sum of pixel counts so each polygon knows its output slice.
    offsets = np.empty(N + 1, dtype=np.int64)
    offsets[0] = 0
    for i in range(N):
        nr = slices[i, 2] - slices[i, 0]
        nc = slices[i, 3] - slices[i, 1]
        offsets[i + 1] = offsets[i] + nr * nc

    total = offsets[N]
    factors = np.zeros(total, dtype=np.float64)
    factors_slice = np.empty((N, 2), dtype=np.int64)
    for i in range(N):
        factors_slice[i, 0] = offsets[i]
        factors_slice[i, 1] = offsets[i + 1]

    # Parallel loop: each polygon is independent.
    for i in prange(N):
        poly_x = control_points[i, :, 0]
        poly_y = control_points[i, :, 1]
        row_min = slices[i, 0]
        col_min = slices[i, 1]
        row_max = slices[i, 2]
        col_max = slices[i, 3]

        out_x = np.empty(32)
        out_y = np.empty(32)
        idx = offsets[i]
        for r in range(row_min, row_max):
            for c in range(col_min, col_max):
                xmin_b = x_ext_left[r]
                xmax_b = x_ext_right[r]
                ymin_b = y_ext_left[c]
                ymax_b = y_ext_right[c]
                box_area = (xmax_b - xmin_b) * (ymax_b - ymin_b)
                nv = _clip_poly_by_box(
                    poly_x, poly_y, 4, xmin_b, ymin_b, xmax_b, ymax_b, out_x, out_y
                )
                if box_area > 0.0 and nv >= 3:
                    factors[idx] = _poly_area_2d(out_x, out_y, nv) / box_area
                idx += 1

    return factors, factors_slice


def _get_factors(control_points, slices, pixel_extents):
    """Compute per-pixel overlap factors for each azimuthal/radial bin polygon.

    Takes a set of control points (vertices of bounding polygons) and slices
    (min/max pixel indices for each polygon) and returns the fractional pixel
    overlap factors.

    The implementation uses a numba-parallelised Sutherland-Hodgman polygon
    clipper, avoiding the shapely dependency and giving a large speedup.
    """
    x_extent, y_extent = pixel_extents
    x_ext_left = np.asarray(x_extent[0], dtype=np.float64)
    x_ext_right = np.asarray(x_extent[1], dtype=np.float64)
    y_ext_left = np.asarray(y_extent[0], dtype=np.float64)
    y_ext_right = np.asarray(y_extent[1], dtype=np.float64)
    slices_arr = np.asarray(slices, dtype=np.int64)
    cp_arr = np.asarray(control_points, dtype=np.float64)

    return _compute_factors_numba(
        cp_arr, slices_arr, x_ext_left, x_ext_right, y_ext_left, y_ext_right
    )


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
