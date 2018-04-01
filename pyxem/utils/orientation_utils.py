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
import pickle

def build_linear_grid_in_euler(alpha,beta,gamma,width,resolution):
    """ 
    Returns tuples between (alpha,beta,gamma) and
    (alpha,beta,gamma) -+ width at steps of resolution. Each
    angle increased incrementaly. Depending on width and resolution
    the tuple (alpha,beta,gamma) may not be included
    
    Parameters:
    ----------
    
    alpha,beta,gamma: Lower angles (0-90,0-180,0-90)
    width           : gives the distance to the min/max. ie) from center-width to center+width
    resolution      : gives the size of the steps
    
    Returns:
    --------
    
    list: rotation list
    """
    a = np.arange(alpha-width,alpha+width,step=resolution)
    b = np.arange(beta-width,beta+width,step=resolution)
    c = np.arange(gamma-width,gamma+width,step=resolution)
    from itertools import product
    return list(product(a,b,c))

def save_DiffractionLibrary(library,filename):
    """
    pickles (saves) the DiffractionLibrary
    """
    
def load_DiffractionLibrary(filename):
    """
    unpickles the DiffractionLibrary. Note risk.
    """
    
_sting_ray = np.array([[0.99,0,0],
                       [0,0.69,0],
                       [0,0,1]])
