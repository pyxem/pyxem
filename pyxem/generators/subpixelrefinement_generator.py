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
from pyxem.utils.subpixel_refinements_utils import _sobel_filtered_xc, _conventional_xc

class SubpixelrefinementGenerator():
    """
    Generates subpixel refinement of DiffractionVectors through a range of methods.

    Parameters
    ----------
    dp : ElectronDiffraction
        The electron diffraction patterns to be refined
    vectors : DiffractionLibrary
        The library of simulated diffraction patterns for indexation

    Notes
    -----

    "xc" stands for cross correlation.
    Readers are refered to Pekin et al. Ultramicroscopy 176 (2017) 170-176 for a detailed comparision
    of phase, cross and hybrid correlation schemes, as well as the use of a sobel filter.

    """

    def __init__(self, dp, vectors):
        self.dp = dp
        self.vectors_init = vectors
        ### for keeping track of work
        self.last_method = None


    def sobel_filtered_xc(self,square_size,disc_radius,upsample_factor):
        self.vectors_out = np.zeros((self.dp.data.shape[0],self.dp.data.shape[1],self.vectors_init.shape[0],self.vectors_init.shape[1]))
        sim_disc = get_simulated_disc(square_size,disc_radius,upsample_factor)
        for i in np.arange(0,len(self.vectors_init)):
            vect = self.vectors_init[i]
            expt_disc = self.dp.map(get_experimental_square,vector=vect,square_size=square_size,upsample_factor=upsample_factor,inplace=False)
            shifts = expt_disc.map(_sobel_filtered_xc,sim_disc=sim_disc,inplace=False)
            self.vectors_out[:,:,i,:] = vect + shifts.data / upsample_factor

        self.last_method = "sobel_filtered"
        return "Solution stored in self.vectors_out"

    def conventional_xc(self,square_size,disc_radius,upsample_factor):
        self.vectors_out = np.zeros((self.dp.data.shape[0],self.dp.data.shape[1],self.vectors_init.shape[0],self.vectors_init.shape[1]))
        sim_disc = get_simulated_disc(square_size,disc_radius,upsample_factor)
        for i in np.arange(0,len(self.vectors_init)):
            vect = self.vectors_init[i]
            expt_disc = self.dp.map(get_experimental_square,vector=vect,square_size=square_size,upsample_factor=upsample_factor,inplace=False)
            shifts = expt_disc.map(_conventional_xc,sim_disc=sim_disc,inplace=False)
            self.vectors_out[:,:,i,:] = vect + shifts.data / upsample_factor

        self.last_method = "conventional_xc"
        return "Solution stored in self.vectors_out"
