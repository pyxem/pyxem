# -*- coding: utf-8 -*-
# Copyright 2016-2024 The pyXem developers
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


def get_DisplacementGradientMap(
    strained_vectors, unstrained_vectors, weights=None, return_residuals=False, **kwargs
):
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
    return_residuals: Bool
        If the residuals for the least squares optimiser should be returned.
    kwargs: dict
        Any additional keyword arguments passed to the `hyperspy.signals.BaseSignal.map`
        function.

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
        output_signal_size=(3, 3),
        output_dtype=np.float64,
        **kwargs
    )

    if return_residuals:
        R = strained_vectors.map(
            get_single_DisplacementGradientTensor,
            Vu=unstrained_vectors,
            weights=weights,
            inplace=False,
            output_dtype=np.float64,
            return_residuals=True,
            **kwargs
        )
        return DisplacementGradientMap(D), R
    else:
        return DisplacementGradientMap(D)


def get_single_DisplacementGradientTensor(
    Vs, Vu=None, weights=None, return_residuals=False
):
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
    return_residuals: Bool
        If the residuals for the least squares optimiser should be returned.
    Returns
    -------
    D : numpy.array
        A 3 x 3 displacement gradient tensor (measured in reciprocal space).
    residuals : numpy.array
        The residuals for the least squares fitting.

    Notes
    -----
    n=2 now behaves the same as the n>2 case; see Release Notes for 0.10.0 for details.

    See Also
    --------
    get_DisplacementGradientMap()

    """
    is_row_nan = np.logical_not(np.any(np.isnan(Vs), axis=1))
    Vs = Vs[is_row_nan]
    Vu = Vu[is_row_nan]

    if Vu is not None:
        if Vu.dtype == object:
            Vu = Vu[()]
    if weights is not None:
        # see https://stackoverflow.com/questions/27128688
        weights = np.asarray(weights)
        # Need vectors normalized to the unstrained region otherwise the weighting breaks down
        Vs = (np.divide(Vs.T, np.linalg.norm(Vu, axis=1)) * np.sqrt(weights)).T
        Vu = (np.divide(Vu.T, np.linalg.norm(Vu, axis=1)) * np.sqrt(weights)).T
    else:
        Vs, Vu = Vs, Vu

    L, residuals, rank, s = np.linalg.lstsq(Vu, Vs, rcond=-1)
    # only need the return array, see np,linalg.lstsq doc
    # Put caculated matrix values into 3 x 3 matrix to be returned.
    D = np.eye(3)
    D[0:2, 0:2] = L

    if return_residuals:
        return residuals
    else:
        return D
