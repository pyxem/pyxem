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

"""Utils for vectors."""

# TODO: Delete this entire module

import numpy as np
from pyxem.utils._deprecated import deprecated


@deprecated(since="0.18.0", removal="1.0.0")
def normalize_virtual_images(im):
    """Normalizes image intensity by dividing by maximum value.

    Parameters
    ----------
    im : np.array()
        Array of image intensities

    Returns
    -------
    imn : np.array()
        Array of normalized image intensities

    """
    imn = im / im.max()
    return imn


@deprecated(
    since="0.18.0", removal="1.0.0", alternative="utils.vectors.get_vector_mesh"
)
def get_vectors_mesh(g1_norm, g2_norm, g_norm_max, angle=0.0, shear=0.0):
    """
    Calculate vectors coordinates of a mesh defined by a norm, a rotation and
    a shear component.

    Parameters
    ----------
    g1_norm, g2_norm : float
        The norm of the two vectors of the mesh.
    g_norm_max : float
        The maximum value for the norm of each vector.
    angle : float, optional
        The rotation of the mesh in degree.
    shear : float, optional
        The shear of the mesh. It must be in the interval [0, 1].
        The default is 0.0.

    Returns
    -------
    np.ndarray
        x and y coordinates of the vectors of the mesh

    """
    from pyxem.utils.vectors import get_vectors_mesh

    return get_vectors_mesh(g1_norm, g2_norm, g_norm_max, angle, shear)
