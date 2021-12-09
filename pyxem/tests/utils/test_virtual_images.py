# -*- coding: utf-8 -*-
# Copyright 2021 The pyXem developers
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
import pytest

from pyxem.utils.virtual_images_utils import get_vectors_mesh


def test_get_vectors_mesh():
    mesh1 = np.array(
        [
            [-1.0, -1.0],
            [0.0, -1.0],
            [1.0, -1.0],
            [-1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    mesh = get_vectors_mesh(1.0, 1.0, g_norm_max=1.5, angle=0.0, shear=0.0)
    np.testing.assert_allclose(mesh, mesh1)

    mesh = get_vectors_mesh(2.0, 2.0, g_norm_max=3.0, angle=0.0, shear=0.0)
    np.testing.assert_allclose(mesh, mesh1 * 2)

    mesh = get_vectors_mesh(1.5, 1.0, g_norm_max=1.9, angle=0.0, shear=0.0)
    np.testing.assert_allclose(mesh.T[0], mesh1.T[0] * 1.5)
    np.testing.assert_allclose(mesh.T[1], mesh1.T[1])

    mesh2 = np.array(
        [
            [0.5, -0.8660254],
            [-0.8660254, -0.5],
            [0.0, 0.0],
            [0.8660254, 0.5],
            [-0.5, 0.8660254],
        ]
    )

    mesh = get_vectors_mesh(1.0, 1.0, g_norm_max=1.0, angle=30, shear=0.0)
    np.testing.assert_allclose(mesh, mesh2)

    mesh3 = np.array([[-0.1, -1.0], [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.1, 1.0]])

    mesh = get_vectors_mesh(1.0, 1.0, g_norm_max=1.2, angle=0.0, shear=0.1)
    np.testing.assert_allclose(mesh, mesh3)

    mesh4 = np.array(
        [
            [0.49607695, -1.09923048],
            [1.36210236, -0.59923048],
            [-0.8660254, -0.5],
            [0.0, 0.0],
            [0.8660254, 0.5],
            [-1.36210236, 0.59923048],
            [-0.49607695, 1.09923048],
        ]
    )

    mesh = get_vectors_mesh(1.0, 1.2, g_norm_max=1.5, angle=30.0, shear=0.1)
    np.testing.assert_allclose(mesh, mesh4)

    with pytest.raises(ValueError):
        get_vectors_mesh(1.0, 1.0, g_norm_max=1.5, angle=0.0, shear=2.0)
