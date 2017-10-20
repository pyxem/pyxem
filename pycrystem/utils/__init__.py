# -*- coding: utf-8 -*-
# Copyright 2017 The PyCrystEM developers
#
# This file is part of PyCrystEM.
#
# PyCrystEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyCrystEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyCrystEM.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from scipy.interpolate import RectBivariateSpline


def correlate(image, pattern,
              include_direct_beam=False,
              sim_threshold=1e-5,
              interpolate=False,
              **kwargs):
    """The correlation between a diffraction pattern and a simulation.
    Calculated using
        .. math::
            \frac{\sum_{j=1}^m P(x_j, y_j) T(x_j, y_j)}{\sqrt{\sum_{j=1}^m P^2(x_j, y_j)} \sqrt{\sum_{j=1}^m T^2(x_j, y_j)}}
    Parameters
    ----------
    image : :class:`np.ndarray`
        A single electron diffraction signal. Should be appropriately scaled
        and centered.
    pattern : :class:`DiffractionSimulation`
        The pattern to compare to.
    sim_threshold : float
        The threshold simulation intensity to consider for correlation
    interpolate : bool
        If True, perform sub-pixel interpolation of the image.
    **kwargs
        Arguments to pass to scipy.interpolate.RectBivariateSpline
    Returns
    -------
    float
        The correlation coefficient.
    References
    ----------
    E. F. Rauch and L. Dupuy, “Rapid Diffraction Patterns identification through
        template matching,” vol. 50, no. 1, pp. 87–99, 2005.
    """
    shape = image.shape
    half_shape = tuple(i // 2 for i in shape)

    pixel_coordinates = pattern.calibrated_coordinates.astype(int)[
        :, :2] + half_shape
    in_bounds = np.product((pixel_coordinates > 0) *
                           (pixel_coordinates < shape[0]), axis=1).astype(bool)
    pattern_intensities = pattern.intensities
    large_intensities = pattern_intensities > sim_threshold
    mask = np.logical_and(in_bounds, large_intensities)

    if interpolate:
        x = np.arange(shape[0], dtype='float') - half_shape[0]
        y = np.arange(shape[1], dtype='float') - half_shape[1]
        for ar, i in zip([x, y], shape):
            if not i % 2:
                ar += 0.5
        x = x * pattern.calibration[0]
        y = y * pattern.calibration[1]
        ip = RectBivariateSpline(x, y, image.T, **kwargs)
        image_intensities = ip.ev(pattern.coordinates[:, 0][mask],
                                  pattern.coordinates[:, 1][mask])
    else:
        image_intensities = image.T[pixel_coordinates[:, 0][in_bounds], pixel_coordinates[:, 1][in_bounds]]
    pattern_intensities = pattern_intensities[mask]
    return np.nan_to_num(_correlate(image_intensities, pattern_intensities))


def correlate_component(image, pattern):
    """The correlation between a diffraction pattern and a simulation.

    Calculated using
        .. math::
            \frac{\sum_{j=1}^m P(x_j, y_j) T(x_j, y_j)}{\sqrt{\sum_{j=1}^m P^2(x_j, y_j)} \sqrt{\sum_{j=1}^m T^2(x_j, y_j)}}

    Parameters
    ----------
    image : :class:`ElectronDiffraction`
        A single electron diffraction signal. Should be appropriately scaled
        and centered.
    pattern : :class:`DiffractionSimulation`
        The pattern to compare to.

    Returns
    -------
    float
        The correlation coefficient.

    References
    ----------
    E. F. Rauch and L. Dupuy, “Rapid Diffraction Patterns identification through
        template matching,” vol. 50, no. 1, pp. 87–99, 2005.

    """
    image_intensities = np.array(
        [image.isig[c[0], c[1]].data for c in pattern.coordinates]
    ).flatten()
    pattern_intensities = pattern.intensities
    return _correlate(image_intensities, pattern_intensities)


def _correlate(intensities_1, intensities_2):
    return np.dot(intensities_1, intensities_2) / (
        np.sqrt(np.dot(intensities_1, intensities_1)) *
        np.sqrt(np.dot(intensities_2, intensities_2))
    )

