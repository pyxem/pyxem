# -*- coding: utf-8 -*-
# Copyright 2017-2019 The pyXem developers
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

from pyxem.signals.diffraction_vectors import DiffractionVectors

def _get_pixel_vectors(dp, vectors, calibration, center):
    """Helper function to get the pixel coordinates for the given diffraction
    pattern and vectors

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

    """

    def _floor(vectors, calibration, center):
        if vectors.shape == (1,) and vectors.dtype == np.object:
            vectors = vectors[0]
        return np.floor((vectors.astype(np.float64) / calibration) + center).astype(np.int)

    if isinstance(vectors, DiffractionVectors):
        if vectors.axes_manager.navigation_shape != dp.axes_manager.navigation_shape:
            raise ValueError('Vectors with shape {} must have the same navigation shape '
                             'as the diffraction patterns which has shape {}.'.format(
                                 vectors.axes_manager.navigation_shape, dp.axes_manager.navigation_shape))
        vector_pixels = vectors.map(_floor,
                                         calibration=calibration,
                                         center=center,
                                         inplace=False)
    else:
        vector_pixels = _floor(vectors, calibration, center)

    if isinstance(vector_pixels, DiffractionVectors):
        if np.any(vector_pixels.data > (np.max(dp.data.shape) - 1)) or (np.any(vector_pixels.data < 0)):
            raise ValueError('Some of your vectors do not lie within your diffraction pattern, check your calibration')
    elif isinstance(vector_pixels, np.ndarray):
        if np.any((vector_pixels > np.max(dp.data.shape) - 1)) or (np.any(vector_pixels < 0)):
            raise ValueError('Some of your vectors do not lie within your diffraction pattern, check your calibration')

    return vector_pixels
