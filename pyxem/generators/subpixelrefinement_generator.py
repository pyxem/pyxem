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

"""
Generating subpixel resolution on diffraction vectors
"""

import numpy as np
from pyxem.utils.subpixel_refinements_utils import *
from pyxem.utils.subpixel_refinements_utils import _conventional_xc
from pyxem.utils.expt_utils import peaks_as_gvectors
from scipy.ndimage.measurements import center_of_mass

class SubpixelrefinementGenerator():
    """
    Generates subpixel refinement of DiffractionVectors through a range of methods.

    Parameters
    ----------
    dp : ElectronDiffraction
        The electron diffraction patterns to be refined
    vectors : numpy.array()
        An array containing the vectors (in calibrated units) to the locations of the spots to be refined

    References
    ----------
    [1] Pekin et al. Ultramicroscopy 176 (2017) 170-176

    """

    def __init__(self, dp, vectors):
        self.dp = dp
        self.vectors_init = vectors
        self.last_method = None
        # this hard codes the squareness of patterns
        self.calibration = dp.axes_manager.signal_axes[0].scale
        self.center      = (dp.axes_manager.signal_axes[0].size)/2
        self.vectors_pixels = ((vectors/self.calibration)+self.center).astype(int)

    def conventional_xc(self,square_size,disc_radius,upsample_factor):
        """
        Refines the peaks using a conventional form of (phase) cross correlation

        Parameters
        ----------
        square_size : int
            Length (in pixels) of one side of a square the contains the peak to be refined
        disc_radius:  int
            Radius (in pixels) of the discs that you seek to refine
        upsample_factor: int
            Factor by which to upsample the patterns

        Returns
        -------
        vector_out: np.array()
            array containing the refined vectors in calibrated units
        """
        self.vectors_out = np.zeros((self.dp.data.shape[0],self.dp.data.shape[1],self.vectors_init.shape[0],self.vectors_init.shape[1]))
        sim_disc = get_simulated_disc(square_size,disc_radius)
        for i in np.arange(0,len(self.vectors_init)):
            vect = self.vectors_pixels[i]
            expt_disc = self.dp.map(get_experimental_square,vector=vect,square_size=square_size,inplace=False)
            shifts = expt_disc.map(_conventional_xc,sim_disc=sim_disc,upsample_factor=upsample_factor,inplace=False)
            self.vectors_out[:,:,i,:] = (((vect + shifts.data) - self.center)*self.calibration)

        self.last_method = "conventional_xc"
        return self.vectors_out

    def center_of_mass_method(self,square_size):
        """
        Find the subpixel refinement of a peak by assuming it lies at the center of intensity

        Parameters
        ----------
        square_size : int
            Length (in pixels) of one side of a square the contains the peak to be refined

        Returns
        -------
        vector_out: np.array()
            array containing the refined vectors in calibrated units

        Notes
        -----
        This will work poorly on disks with strong dynamic contrast
        """

        self.vectors_out = np.zeros((self.dp.data.shape[0],self.dp.data.shape[1],self.vectors_init.shape[0],self.vectors_init.shape[1]))
        for i in np.arange(0,len(self.vectors_init)):
            vect = self.vectors_pixels[i]
            expt_disc = self.dp.map(get_experimental_square,vector=vect,square_size=square_size,inplace=False)
            shifts = expt_disc.map(center_of_mass,inplace=False) - (square_size/2)
            self.vectors_out[:,:,i,:] = (((vect + shifts.data) - self.center)*self.calibration)

        self.last_method = "center_of_mass_method"
        return self.vectors_out
