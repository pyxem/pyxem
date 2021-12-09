# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
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

"""Generating DisplacementGradientMaps from diffraction vectors."""

import numpy as np
from pyxem.signals.tensor_field import DisplacementGradientMap


def get_DisplacementGradientMap(strained_vectors, unstrained_vectors, weights=None):
    r"""Calculates the displacement gradient tensor at each navigation position in a map.

    Compares vectors to determine the 2 x 2 matrix,
    :math:`\\mathbf(L)`, that maps unstrained vectors, Vu, to strained vectors,
    Vs, using the np.lingalg.inv() function to find L that satisfies
    :math:`Vs = \\mathbf(L) Vu`.

    The transformation is returned as a 3 x 3 displacement gradient tensor.

    Parameters
    ----------
    strained_vectors : hyperspy.Signal2D
        Signal2D with a 2 x n array at each navigation position containing the
        Cartesian components of two strained basis vectors, V and U, defined as
        row vectors.
    unstrained_vectors : numpy.array
        A 2 x n array containing the Cartesian components of two unstrained
        basis vectors, V and U, defined as row vectors.
    weights : list
        of weights to be passed to the least squares optimiser, not used for n=2

    Returns
    -------
    D : DisplacementGradientMap
        The 3 x 3 displacement gradient tensor (measured in reciprocal space) at
        every navigation position.

    Notes
    -----
    n=2 now behaves the same as the n>2 case; see Release Notes for 0.10.0 for details.

    See Also
    --------
    get_single_DisplacementGradientTensor()

    """
    # Calculate displacement gradient tensor across map.
    D = strained_vectors.map(
        get_single_DisplacementGradientTensor,
        Vu=unstrained_vectors,
        weights=weights,
        inplace=False,
    )

    return DisplacementGradientMap(D)


def get_single_DisplacementGradientTensor(Vs, Vu=None, weights=None):
    r"""Calculates the displacement gradient tensor from a pairs of vectors.

    Determines the 2 x 2 matrix, :math:`\\mathbf(L)`, that maps unstrained
    vectors, Vu, onto strained vectors, Vs

    The transformation is returned as a 3 x 3 displacement gradient tensor.

    Parameters
    ----------
    Vs : numpy.array
        A 2 x n array containing the Cartesian components of two strained basis
        vectors, V and U, defined as row vectors.
    Vu : numpy.array
        A 2 x n array containing the Cartesian components of two unstrained
        basis vectors, V and U, defined as row vectors.
    weights : list
        of weights to be passed to the least squares optimiser
    Returns
    -------
    D : numpy.array
        A 3 x 3 displacement gradient tensor (measured in reciprocal space).

    Notes
    -----
    n=2 now behaves the same as the n>2 case; see Release Notes for 0.10.0 for details.

    See Also
    --------
    get_DisplacementGradientMap()

    """
    if weights is not None:
        # see https://stackoverflow.com/questions/27128688
        weights = np.asarray(weights)
        # Need vectors normalized to the unstrained region otherwise the weighting breaks down
        Vs = (
            np.divide(Vs, np.linalg.norm(Vu, axis=0)) * np.sqrt(weights)
        ).T  # transpose for conventions
        Vu = (np.divide(Vu, np.linalg.norm(Vu, axis=0)) * np.sqrt(weights)).T
    else:
        Vs, Vu = Vs.T, Vu.T

    L = np.linalg.lstsq(Vu, Vs, rcond=-1)[
        0
    ]  # only need the return array, see np,linalg.lstsq docs
    # Put caculated matrix values into 3 x 3 matrix to be returned.
    D = np.eye(3)
    D[0:2, 0:2] = L

    return D
