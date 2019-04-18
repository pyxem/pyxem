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

import numpy as np
import pytest

from pyxem.signals.indexation_results import TemplateMatchingResults
from pyxem.signals.indexation_results import VectorMatchingResults
from pyxem.signals.diffraction_vectors import DiffractionVectors


def test_template_get_crystallographic_map(dp_template_match_result,
                                           sp_template_match_result):
    # Assertion free test, as the tests in test_indexation_utils do the heavy
    # lifting
    match_results = np.array(np.vstack((dp_template_match_result[0], sp_template_match_result[0])))
    match_results = TemplateMatchingResults(match_results)
    cryst_map = match_results.get_crystallographic_map()
    assert cryst_map.method == 'template_matching'


def test_vector_get_crystallographic_map(dp_vector_match_result,
                                         sp_vector_match_result):
    # Assertion free test, as the tests in test_indexation_utils do the heavy
    # lifting
    match_results = np.array([np.vstack((dp_vector_match_result, sp_vector_match_result))])
    match_results = VectorMatchingResults(match_results)
    match_results.axes_manager.set_signal_dimension(2)
    cryst_map = match_results.get_crystallographic_map()
    assert cryst_map.method == 'vector_matching'


@pytest.mark.parametrize('overwrite, result_hkl, current_hkl, expected_hkl', [
    (True, [0, 0, 1], None, [0, 0, 1]),
    (False, [0, 0, 1], None, [0, 0, 1]),
])
def test_vector_get_indexed_diffraction_vectors(overwrite,
                                                result_hkl,
                                                current_hkl,
                                                expected_hkl):
    match_results = VectorMatchingResults(np.array([[1], [2]]))
    match_results.hkls = result_hkl
    vectors = DiffractionVectors(np.array([[1], [2]]))
    vectors.hkls = current_hkl
    match_results.get_indexed_diffraction_vectors(vectors, overwrite)
    np.testing.assert_allclose(vectors.hkls, expected_hkl)


def test_vector_get_indexed_diffraction_vectors_warn():
    match_results = VectorMatchingResults(np.array([[1], [2]]))
    match_results.hkls = [0, 0, 1]
    vectors = DiffractionVectors(np.array([[1], [2]]))
    vectors.hkls = [0, 0, 0]
    with pytest.warns(Warning):
        match_results.get_indexed_diffraction_vectors(vectors)
    np.testing.assert_allclose(vectors.hkls, [0, 0, 0])
