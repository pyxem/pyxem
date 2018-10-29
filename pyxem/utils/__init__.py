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


def correlate(image, pattern_dictionary):
    """The correlation between a diffraction pattern and a simulation.

    Calculated using

    .. math::

        \\frac{\sum_{j=1}^m P(x_j, y_j) T(x_j, y_j)}{\sqrt{\sum_{j=1}^m T^2(x_j, y_j)}}

    for T a template and P an experimental pattern

    Parameters
    ----------
    image : :class:`numpy.ndarray`
        A single electron diffraction signal. Should be appropriately scaled
        and centered.
    pattern_dictrionary : dict
        Containing:
            :class:`pyxem.DiffractionSimulation`
            The calibrated coords as 'pixel_coords',masked correctly
            The intensities as 'intensities', masked correctly
    Returns
    -------
    float
        A coefficient of correlation, unnormalised. This is in contrast to the
        reference work.

    References
    ----------
    E. F. Rauch and L. Dupuy, “Rapid Diffraction Patterns identification through
       template matching,” vol. 50, no. 1, pp. 87–99, 2005.
    """

    pattern_intensities = pattern_dictionary['intensities']
    pixel_coordinates = pattern_dictionary['pixel_coords']
    pattern_normalization = pattern_dictionary['pattern_norm']

    # The x,y choice here is correct. Basically the numpy/hyperspy conversion is a danger
    image_intensities = image[pixel_coordinates[:, 1], pixel_coordinates[:, 0]]

    return np.dot(image_intensities, pattern_intensities) / pattern_normalization
