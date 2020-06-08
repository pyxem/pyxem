# -*- coding: utf-8 -*-
# Copyright 2016-2020 The pyXem developers
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

from skimage.filters import gaussian as point_spread

from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D


def sim_as_signal(diffsim, size, sigma, max_r):
    """Returns the diffraction data as an ElectronDiffraction signal with
    two-dimensional Gaussians representing each diffracted peak. Should only
    be used for qualitative work.
    Parameters
    ----------
    diffsim : diffsims.DiffractionSimulation
        A DiffractionSimulation object
    size : int
        Side length (in pixels) for the signal to be simulated.
    sigma : float
        Standard deviation of the Gaussian function to be plotted.
    max_r : float
        Half the side length in reciprocal Angstroms. Defines the signal's
        calibration
    Returns
    -------
    dp : ElectronDiffraction
        Simulated electron diffraction pattern.
    """
    l, delta_l = np.linspace(-max_r, max_r, size, retstep=True)

    mask_for_max_r = np.logical_and(
        np.abs(diffsim.coordinates[:, 0]) < max_r,
        np.abs(diffsim.coordinates[:, 1]) < max_r,
    )

    coords = diffsim.coordinates[mask_for_max_r]
    inten = diffsim.intensities[mask_for_max_r]

    dp_dat = np.zeros([size, size])
    x, y = (coords)[:, 0], (coords)[:, 1]
    if len(x) > 0:  # avoiding problems in the peakless case
        num = np.digitize(x, l, right=True), np.digitize(y, l, right=True)
        dp_dat[num] = inten
        # sigma in terms of pixels. transpose for Hyperspy
        dp_dat = point_spread(dp_dat, sigma=sigma / delta_l).T
        dp_dat = dp_dat / np.max(dp_dat)

    dp = ElectronDiffraction2D(dp_dat)
    dp.set_diffraction_calibration(2 * max_r / size)

    return dp
