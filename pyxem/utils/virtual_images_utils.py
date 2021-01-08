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

import numpy as np


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

    def rotation_matrix(angle):
        return np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )

    def shear_matrix(shear):
        return np.array([[1.0, shear], [0.0, 1.0]])

    if shear < 0 or shear > 1:
        raise ValueError("The `shear` value must be in the interval [0, 1].")

    order1 = int(np.ceil(g_norm_max / g1_norm))
    order2 = int(np.ceil(g_norm_max / g2_norm))
    order = max(order1, order2)

    x = np.arange(-g1_norm * order, g1_norm * (order + 1), g1_norm)
    y = np.arange(-g2_norm * order, g2_norm * (order + 1), g2_norm)

    xx, yy = np.meshgrid(x, y)
    vectors = np.stack(np.meshgrid(x, y)).reshape((2, (2 * order + 1) ** 2))

    transformation = rotation_matrix(np.radians(angle)) @ shear_matrix(shear)

    vectors = transformation @ vectors
    norm = np.linalg.norm(vectors, axis=0)

    return vectors[:, norm <= g_norm_max].T
