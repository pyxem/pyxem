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
Generating DisplacementGradientMaps from diffraction vectors
"""

import numpy as np
from pyxem.signals.tensor_field import DisplacementGradientMap


def get_DisplacementGradientMap(strained_vectors, unstrained_vectors):
    """Calculates the displacement gradient tensor at each navigation position
    in a map by comparing vectors to determine the 2 x 2 matrix,
    :math:`\\mathbf(L)`, that maps unstrained vectors, Vu, to strained vectors,
    Vs, using the np.lingalg.inv() function to find L that satisfies
    :math:`Vs = \\mathbf(L) Vu`.

    The transformation is returned as a 3 x 3 displacement gradient tensor.

    Parameters
    ----------
    strained_vectors : hyperspy.Signal2D
        Signal2D with a 2 x 2 array at each navigation position containing the
        Cartesian components of two strained basis vectors, V and U, defined as
        row vectors.
    unstrained_vectors : numpy.array
        A 2 x 2 array containing the Cartesian components of two unstrained
        basis vectors, V and U, defined as row vectors.

    Returns
    -------
    D : DisplacementGradientMap
        The 3 x 3 displacement gradient tensor (measured in reciprocal space) at
        every navigation position.

    See Also
    --------
    get_single_DisplacementGradientTensor()

    """
    # Calculate displacement gradient tensor across map.
    D = strained_vectors.map(get_single_DisplacementGradientTensor,
                             Vu=unstrained_vectors, inplace=False)

    return DisplacementGradientMap(D)


def get_single_DisplacementGradientTensor(Vs, Vu=None):
    """Calculates the displacement gradient tensor from a pairs of vectors by
    determining the 2 x 2 matrix, :math:`\\mathbf(L)`, that maps unstrained
    vectors, Vu, onto strained vectors, Vs, using the np.lingalg.inv() function
    to find :math:`\\mathbf(L)` that satisfies :math:`Vs = \\mathbf(L) Vu`.

    The transformation is returned as a 3 x 3 displacement gradient tensor.

    Parameters
    ----------
    Vs : numpy.array
        A 2 x 2 array containing the Cartesian components of two strained basis
        vectors, V and U, defined as row vectors.
    Vu : numpy.array
        A 2 x 2 array containing the Cartesian components of two unstrained
        basis vectors, V and U, defined as row vectors.

    Returns
    -------
    D : numpy.array
        A 3 x 3 displacement gradient tensor (measured in reciprocal space).

    See Also
    --------
    get_DisplacementGradientMap()

    """
    # Take transpose to ensure conventions obeyed.
    Vs, Vu = Vs.T, Vu.T
    # Perform matrix multiplication to calculate 2 x 2 L-matrix.
    L = np.matmul(Vs, np.linalg.inv(Vu))
    # Put cacluated matrix values into 3 x 3 matrix to be returned.
    D = np.eye(3)
    D[0:2, 0:2] = L

    return D
