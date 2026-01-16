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

"""Utilities for indexing electron diffraction spot patterns."""

import numpy as np

from pyxem.common import CUPY_INSTALLED
from pyxem.utils.cuda import (
    _correlate_polar_image_to_library_gpu,
    TPB,
)
from pyxem.utils._indexation import _index_chunk

if CUPY_INSTALLED:
    import cupy as cp


def _match_polar_to_polar_library_gpu(
    polar_image,
    r_templates,
    theta_templates,
    intensities_templates,
):
    """
    Correlates a polar pattern to all polar templates on GPU

    Parameters
    ----------
    polar_image : 2D cupy.ndarray
        The image converted to polar coordinates
    r_templates : 2D cupy.ndarray
        r-coordinates of diffraction spots in templates.
    theta_templates : 2D cupy ndarray
        theta-coordinates of diffraction spots in templates.
    intensities_templates : 2D cupy.ndarray
        intensities of the spots in each template

    Returns
    -------
    best_in_plane_shift : (N) 1D cupy.ndarray
        Shift for all templates that yields best correlation
    best_in_plane_corr : (N) 1D cupy.ndarray
        Correlation at best match for each template
    best_in_plane_shift_m : (N) 1D cupy.ndarray
        Shift for all mirrored templates that yields best correlation
    best_in_plane_corr_m : (N) 1D cupy.ndarray
        Correlation at best match for each mirrored template

    Notes
    -----
    The dimensions of r_templates and theta_templates should be (N, R) where
    N is the number of templates and R the number of spots in the template
    with the maximum number of spots
    """
    correlation = cp.empty(
        (r_templates.shape[0], polar_image.shape[0]), dtype=cp.float32
    )
    correlation_m = cp.empty(
        (r_templates.shape[0], polar_image.shape[0]), dtype=cp.float32
    )
    threadsperblock = (1, TPB)
    blockspergrid = (r_templates.shape[0], int(np.ceil(polar_image.shape[0] / TPB)))
    _correlate_polar_image_to_library_gpu[blockspergrid, threadsperblock](
        polar_image,
        r_templates,
        theta_templates,
        intensities_templates,
        correlation,
        correlation_m,
    )
    best_in_plane_shift = cp.argmax(correlation, axis=1).astype(np.int32)
    best_in_plane_shift_m = cp.argmax(correlation_m, axis=1).astype(np.int32)
    rows = cp.arange(correlation.shape[0], dtype=np.int32)
    best_in_plane_corr = correlation[rows, best_in_plane_shift]
    best_in_plane_corr_m = correlation_m[rows, best_in_plane_shift_m]
    return (
        best_in_plane_shift,
        best_in_plane_corr,
        best_in_plane_shift_m,
        best_in_plane_corr_m,
    )


def _index_chunk_gpu(images, *args, **kwargs):
    import cupy as cp

    gpu_im = cp.asarray(images)
    indexed_chunk = _index_chunk(gpu_im, *args, **kwargs)
    return cp.asnumpy(indexed_chunk)
