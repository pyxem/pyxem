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

from numba import njit, prange
import numpy as np


@njit(parallel=True, nogil=True)
def _match_polar_to_polar_library_cpu(
    polar_image,
    r_templates,
    theta_templates,
    intensities_templates,
):
    """
    Correlates a polar pattern to all polar templates on CPU

    Parameters
    ----------
    polar_image : 2D numpy.ndarray
        The image converted to polar coordinates
    r_templates : 2D numpy.ndarray
        r-coordinates of diffraction spots in templates.
    theta_templates : 2D numpy ndarray
        theta-coordinates of diffraction spots in templates.
    intensities_templates : 2D numpy.ndarray
        intensities of the spots in each template

    Returns
    -------
    best_in_plane_shift : (N) 1D numpy.ndarray
        Shift for all templates that yields best correlation
    best_in_plane_corr : (N) 1D numpy.ndarray
        Correlation at best match for each template
    best_in_plane_shift_m : (N) 1D numpy.ndarray
        Shift for all mirrored templates that yields best correlation
    best_in_plane_corr_m : (N) 1D numpy.ndarray
        Correlation at best match for each mirrored template

    Notes
    -----
    The dimensions of r_templates and theta_templates should be (N, R) where
    N is the number of templates and R the number of spots in the template
    with the maximum number of spots
    """
    N = r_templates.shape[0]
    R = r_templates.shape[1]
    n_shifts = polar_image.shape[0]
    best_in_plane_shift = np.empty(N, dtype=np.int32)
    best_in_plane_shift_m = np.empty(N, dtype=np.int32)
    best_in_plane_corr = np.empty(N, dtype=polar_image.dtype)
    best_in_plane_corr_m = np.empty(N, dtype=polar_image.dtype)

    for template in prange(N):
        inplane_cor = np.zeros(n_shifts)
        inplane_cor_m = np.zeros(n_shifts)
        for spot in range(R):
            rsp = r_templates[template, spot]
            if rsp == 0:
                break
            tsp = theta_templates[template, spot]
            isp = intensities_templates[template, spot]
            split = n_shifts - tsp
            column = polar_image[:, rsp] * isp
            inplane_cor[:split] += column[tsp:]
            inplane_cor[split:] += column[:tsp]
            inplane_cor_m[:tsp] += column[split:]
            inplane_cor_m[tsp:] += column[:split]

        best_shift = np.argmax(inplane_cor)
        best_shift_m = np.argmax(inplane_cor_m)
        best_in_plane_shift[template] = best_shift
        best_in_plane_shift_m[template] = best_shift_m
        best_in_plane_corr[template] = inplane_cor[best_shift]
        best_in_plane_corr_m[template] = inplane_cor_m[best_shift_m]

    return (
        best_in_plane_shift,
        best_in_plane_corr,
        best_in_plane_shift_m,
        best_in_plane_corr_m,
    )
