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

""" Generating subpixel resolution on diffraction vectors

"""

import numpy as np

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

    """

    def __init__(self, dp, vectors):
        self.dp = dp
        self.vectors_init = vectors

    def conventional_xc(self):
        pass

    def sobel_filtered_xc(self):
        pass

    """
    #Potential addition methods
    def hybrid_xc(self):
        pass

    def radial_gradient(self):
        pass
    """
