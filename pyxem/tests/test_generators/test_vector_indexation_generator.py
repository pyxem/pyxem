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

import pytest
import numpy as np

from pyxem.generators.indexation_generator3d import IndexationGenerator3D

from diffsims.libraries.vector_library import DiffractionVectorLibrary
from pyxem.signals.diffraction_vectors2d import DiffractionVectors2D

from pyxem.utils.indexation_utils import OrientationResult


def test_vector_indexation_generator_init():
    vectors = DiffractionVectors2D([[1], [2], [3]])
    vector_library = DiffractionVectorLibrary()
    indexation_generator3d = IndexationGenerator3D(vectors, vector_library)
    assert isinstance(indexation_generator3d, IndexationGenerator3D)
    assert indexation_generator3d.vectors == vectors
    assert indexation_generator3d.library == vector_library


@pytest.mark.xfail(raises=ValueError)
def test_vector_indexation_generator_cartesian_check():
    vectors = DiffractionVectors2D([[1], [2], [3]])
    vector_library = DiffractionVectorLibrary()
    vector_indexation_generator = IndexationGenerator3D(vectors, vector_library)


#def test_vector_indexation_generator_index_vectors(vector_match_peaks,
#                                                   vector_library):
#    # vectors not used directly
#    vectors = DiffractionVectors2D(np.array(vector_match_peaks[:, :2]))
#    vectors.cartesian = DiffractionVectors2D(np.array(vector_match_peaks))
#    gen = VectorIndexationGenerator3D(vectors, vector_library)
#    indexation = gen.index_vectors(
#        mag_tol=0.1,
#        angle_tol=6,
#        index_error_tol=0.3,
#        n_peaks_to_index=2,
#        n_best=5)
#
#    # Values are tested directly on the match_vector in the util tests
#    assert isinstance(indexation.vectors, DiffractionVectors2D)
#
#    # (n_best=1, 5 result values from each)
#    np.testing.assert_equal(indexation.data.shape, (5,))
#
#    # n_best=1, 3 peaks with hkl)
#    np.testing.assert_equal(indexation.hkls.shape, (1, 3, 3))
#
#    refined1 = gen.refine_n_best_orientations(indexation, 1.0, 1.0, n_best=0)
#
#    assert isinstance(refined1.vectors, DiffractionVectors2D)
#    np.testing.assert_equal(refined1.data.shape, (5,))
#
#    refined2 = gen.refine_best_orientation(indexation, 1.0, 1.0)
#
#    assert isinstance(refined2.vectors, DiffractionVectors2D)
#    np.testing.assert_equal(refined2.data.shape, (1,))
#    assert isinstance(refined2.data[0], OrientationResult)
#
#    assert refined2.data[0].phase_index == indexation.data[0].phase_index
#    assert refined2.data[0].match_rate == indexation.data[0].match_rate
#
#    # Must use a large tolerance here, because there are only 3 vectors
#    np.testing.assert_almost_equal(np.diag(refined1.data[0].rotation_matrix),
#                                   np.diag(indexation.data[0].rotation_matrix), 1)
#    np.testing.assert_almost_equal(np.diag(refined2.data[0].rotation_matrix),
#                                   np.diag(indexation.data[0].rotation_matrix), 1)
