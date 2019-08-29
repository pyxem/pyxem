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

"""
Generating subpixel resolution on diffraction vectors.
"""

import numpy as np
from hyperspy.signals import BaseSignal
from skimage import morphology

from pyxem.signals.diffraction_vectors import DiffractionVectors

import warnings


def _get_intensities(z, vectors, radius=1):
    """
    Basic intensity integration routine, takes the maximum value at the
    given vector positions with the number of pixels given by `radius`.

    Parameters
    ----------
    vectors : DiffractionVectors
        Vectors to the locations of the spots to be
        integrated.
    radius: int,
        Number of pixels within which to find the largest maximum

    Returns
    -------
    intensities : np.array
        List of extracted intensities
    """
    i, j = np.array(vectors.data).astype(int).T
    
    if radius > 1:
        footprint = morphology.disk(radius)
        filtered = ndimage.maximum_filter(z, footprint=footprint)
        intensities = filtered[j, i].reshape(-1,1)  # note that the indices are flipped
    else:
        intensities = z[j, i].reshape(-1,1)  # note that the indices are flipped
    
    return np.array(intensities)


class IntegrationGenerator():
    """
    Integrates reflections at the given vector positions.

    Parameters
    ----------
    dp : ElectronDiffraction2D
        The electron diffraction patterns to be refined
    vectors : DiffractionVectors | ndarray
        Vectors (in calibrated units) to the locations of the spots to be
        integrated. If given as DiffractionVectors, it must have the same
        navigation shape as the electron diffraction patterns. If an ndarray,
        the same set of vectors is mapped over all electron diffraction
        patterns.

    TODO: extend with more sophisticated intensity extraction methods
    """

    def __init__(self, dp, vectors):
        self.dp = dp
        self.vectors_init = vectors
        self.last_method = None
        sig_ax = dp.axes_manager.signal_axes
        self.calibration = [sig_ax[0].scale, sig_ax[1].scale]
        self.center = [sig_ax[0].size / 2, sig_ax[1].size / 2]

        def _floor(vectors, calibration, center):
            if vectors.shape == (1,) and vectors.dtype == np.object:
                vectors = vectors[0]
            return np.floor((vectors.astype(np.float64) / calibration) + center).astype(np.int)

        if isinstance(vectors, DiffractionVectors):
            if vectors.axes_manager.navigation_shape != dp.axes_manager.navigation_shape:
                raise ValueError('Vectors with shape {} must have the same navigation shape '
                                 'as the diffraction patterns which has shape {}.'.format(
                                     vectors.axes_manager.navigation_shape, dp.axes_manager.navigation_shape))
            self.vector_pixels = vectors.map(_floor,
                                             calibration=self.calibration,
                                             center=self.center,
                                             inplace=False)
        else:
            self.vector_pixels = _floor(vectors, self.calibration, self.center)

        if isinstance(self.vector_pixels, DiffractionVectors):
            if np.any(self.vector_pixels.data > (np.max(dp.data.shape) - 1)) or (np.any(self.vector_pixels.data < 0)):
                raise ValueError('Some of your vectors do not lie within your diffraction pattern, check your calibration')
        elif isinstance(self.vector_pixels, np.ndarray):
            if np.any((self.vector_pixels > np.max(dp.data.shape) - 1)) or (np.any(self.vector_pixels < 0)):
                raise ValueError('Some of your vectors do not lie within your diffraction pattern, check your calibration')

    def extract_intensities(self, radius: int=2):
        """
        Basic intensity integration routine, takes the maximum value at the
        given vector positions with the number of pixels given by `radius`.
    
        Parameters
        ----------
        radius: int,
            Number of pixels within which to find the largest maximum
    
        Returns
        -------
        intensities : :obj:`hyperspy.signals.BaseSignal`
            List of extracted intensities
        """
        intensities = self.dp.map(_get_intensities, 
                                  vectors=self.vector_pixels, 
                                  radius=radius, 
                                  inplace=False)

        intensities = BaseSignal(intensities)
        intensities.axes_manager.set_signal_dimension(0)

        return intensities
