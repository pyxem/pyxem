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
Generating DisplacementGradientMaps from diffraction vectors
"""

import numpy as np

def get_DisplacementGradientMap(strained_vectors, unstrained_vectors):
    """
    Calculates the displacement gradient by comparing vectors with linear algebra

    Parameters
    ----------
    strained_vectors : Signal2D

    unstrained_vectors : numpy.array with shape (2,2)
        For two vectors: V and U measured in x and y the components should fill the array as
        >>> array([[Vx, Vy],
                   [Ux, Uy]])
    Returns
    -------
    D : DisplacementGradientMap
        The 3x3 displacement gradient tensor (measured in reciprocal space)
        at every navigation position

    See Also
    --------
    get_single_DisplacementGradientTensor()

    Notes
    -----
    This function does not currently support keyword arguments to the underlying map function.
    """

    D = strained_vectors.map(get_single_DisplacementGradientTensor,Vu=unstrained_vectors,inplace=False)
    return DisplacementGradientMap(D)

def get_single_DisplacementGradientTensor(Vs,Vu=None):
    """
    Calculates the displacement gradient tensor by relating two pairs of vectors

    Parameters
    ----------
    Vs : numpy.array with shape (2,2)
        For two vectors V and U measured in x and y the components should fill the array as
        >>> array([[Vx, Vy],
                   [Ux, Uy]])
    Vu : numpy.array with shape (2,2)
        For two vectors V and U measured in x and y the components should fill the array as
        >>> array([[Vx, Vy],
                   [Ux, Uy]])

    Returns
    -------
    D : numpy.array of shape (3x3)
        Components are [[,,0],[,,0],[0,0,1]]

    Notes
    -----
    This routine is based on the equation

    Vs = L Vu

    Where L is a (2x2) transform matrix that takes Vu (unstrained) onto Vs (strained).
    We find L by using the np.linalg.inv() function
    """

    L = np.matmul(Vs,np.linalg.inv(Vu))
    D = np.eye(3)
    D[0:2,0:2] = L
    return D
