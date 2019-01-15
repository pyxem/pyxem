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
Low level interface for generating DisplacementGradientMaps from diffraction vectors
"""

import numpy as np

def get_DisplacementGradientMap(strained_vectors, Vu):
    """
    Vectors Strained : Signal2D
    Vector Unstrained : np.array(2x2)
    kwargs: To be passed to the hyperspy map function

    returns DisplacementGradientMap
    """

    if 'inplace' in kwargs:
        raise ValueError

    D = strained_vectors.map(_get_single_DisplacementGradientTensor,Vu=Vu,kwargs)
    #return DisplacementGradientMap(D)
    pass

def _get_single_DisplacementGradientTensor(Vs,Vu=None):
    """
    Vector Strained:   Vs : (2x2) np.array [vax,vbx] [vay,vby]
    Vector Unstrained: Vu :(2x2) np.array

    X = d11*x + d12*y
    Y = d21*x + d22*y

    where X and Y are the strained answers. 4 equation 4 unknowns.
    """
    if Vu is None:
        raise ValueError
    #check that none-of the vectors move too much
    L = np.matmult(Vs,np.linalg.inv(Vu))
    return L
