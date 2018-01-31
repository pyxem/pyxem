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

        \\frac{\sum_{j=1}^m P(x_j, y_j) T(x_j, y_j)}{\sqrt{\sum_{j=1}^m P^2(x_j, y_j)} \sqrt{\sum_{j=1}^m T^2(x_j, y_j)}}

    Parameters
    ----------
    image : :class:`numpy.ndarray`
        A single electron diffraction signal. Should be appropriately scaled
        and centered.
    pattern : :class:`pyxem.DiffractionSimulation`
        The pattern to compare to.
    sim_threshold : float
        The threshold simulation intensity to consider for correlation
    interpolate : bool
        If True, perform sub-pixel interpolation of the image.
    **kwargs
        Arguments to pass to :class:`scipy.interpolate.RectBivariateSpline`

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
    
    #the hard code on this number should be considered    
    in_bounds_z = np.abs(pattern.coordinates[:,2]) < 1e-2
    
    pixel_coordinates = np.rint(pattern.calibrated_coordinates[:,:2]+half_shape).astype(int)
    in_bounds = np.product((pixel_coordinates > 0)*(pixel_coordinates < shape[0]), axis=1)

    pattern_intensities = pattern.intensities
    large_intensities = pattern_intensities > sim_threshold
    mask = np.logical_and(in_bounds_z,np.logical_and(in_bounds, large_intensities))

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
        image_intensities = image[pixel_coordinates[:, 0][mask], pixel_coordinates[:, 1][mask]]
    
    pattern_intensities = pattern_intensities[mask]
    # Take care here to use ALL image pixels in the imageimage product.
    numerator = np.dot(image_intensities,pattern_intensities)
    denominator = np.sqrt(np.dot(pattern_intensities,pattern_intensities)*np.sum(np.dot(image,image)))
    
    return np.nan_to_num(numerator/denominator)


def _correlate(intensities_1, intensities_2):
    return np.dot(intensities_1, intensities_2) / (np.sqrt(np.dot(intensities_1, intensities_1) * np.dot(intensities_2, intensities_2)))
