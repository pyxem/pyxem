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
utils to support SubpixelrefinementGenerator
"""

import numpy as np
from skimage.filters import sobel
from skimage.feature import register_translation
from skimage import draw
from skimage.transform import rescale

def get_experimental_square(z,vector,square_size,upsample_factor):
    """
    'Cuts out' a a region around a given diffraction vector and returns and upsample_factor
    copy.

    Args
    ----

    z: np.array() - Single diffraction pattern
    vector: np.array() - Single vector to be cut out
    square_size: The length of one side of the bounding square_size (must be even)
    upsample_factor: The factor by which to up-sample (must be even)

    """
    cx,cy,half_ss = vector[0], vector[1], int(square_size/2)
    _z = z[cx-half_ss:cx+half_ss,cy-half_ss:cy+half_ss]
    _z = rescale(_z,upsample_factor)
    return _z

def get_simulated_disc(square_size,disc_radius,upsample_factor):
    upsss = int(square_size)#*upsample_factor) #upsample square size
    arr = np.zeros((upsss,upsss))
    rr, cc = draw.circle(int(upsss/2), int(upsss/2), radius=disc_radius*1, shape=arr.shape) #is the thin disc a good idea
    arr[rr, cc] = 1
    arr = rescale(arr,upsample_factor)
    return arr

def _sobel_filtered_xc(exp_disc,sim_disc):
    sobel_exp_disc = sobel(exp_disc)
    h0,h1= np.hanning(np.size(sim_disc,0)),np.hanning(np.size(sim_disc,1))
    hann2d = np.sqrt(np.outer(h0,h1))
    shifts,error,_ = register_translation(sobel_exp_disc*hann2d,sim_disc*hann2d)
    return shifts

def _conventional_xc(exp_disc,sim_disc):
    h0,h1= np.hanning(np.size(sim_disc,0)),np.hanning(np.size(sim_disc,1))
    hann2d = np.sqrt(np.outer(h0,h1))
    shifts,error,_ = register_translation(exp_disc,sim_disc)
    return shifts
