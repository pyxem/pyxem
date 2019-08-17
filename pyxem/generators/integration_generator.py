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

from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.utils.expt_utils import peaks_as_gvectors
from pyxem.utils.subpixel_refinements_utils import _conventional_xc
from pyxem.utils.subpixel_refinements_utils import get_experimental_square
from pyxem.utils.subpixel_refinements_utils import get_simulated_disc

import warnings


class OrientationRefinementGenerator():
    """Generates subpixel refinement of DiffractionVectors.

    Parameters
    ----------
    dp : ElectronDiffraction2D
        The electron diffraction patterns to be refined
    vectors : DiffractionVectors | ndarray
        Vectors (in calibrated units) to the locations of the spots to be
        refined. If given as DiffractionVectors, it must have the same
        navigation shape as the electron diffraction patterns. If an ndarray,
        the same set of vectors is mapped over all electron diffraction
        patterns.

    References
    ----------
    [1] Pekin et al. Ultramicroscopy 176 (2017) 170-176

    """

    def __init__(self, dp, vectors):
        self.dp = dp
        self.vectors_init = vectors
        self.last_method = None
        sig_ax = dp.axes_manager.signal_axes
        self.calibration = [sig_ax[0].scale, sig_ax[1].scale]
        self.center = [sig_ax[0].size / 2, sig_ax[1].size / 2]

    def get_intensities(img, result, projector, radius=1):
        """
        Grab reflection intensities at given projection
    
        radius: int, optional
            Search for largest point in defined radius around projected peak positions
        """
        proj = projector.get_projection(result.alpha, result.beta, result.gamma)
        i, j, hkl = get_indices(proj[:,3:5], result.scale, (result.center_x, result.center_y), img.shape, hkl=proj[:,0:3])
        if radius > 1:
            footprint = morphology.disk(radius)
            img = ndimage.maximum_filter(img, footprint=footprint)
        inty = img[i, j].reshape(-1,1)
        return np.hstack((hkl, inty, np.ones_like(inty))).astype(int)

    def get_indices(pks, scale, center, shape, hkl=None):
        """Get the pixel indices for an image"""
        shapex, shapey = shape
        i, j = (pks * scale + center).astype(int).T
        sel = (0 < j) & (j < shapey) & (0 < i) & (i < shapex)
        if hkl is None:
            return i[sel], j[sel]
        else:
            return i[sel], j[sel], hkl[sel] 
