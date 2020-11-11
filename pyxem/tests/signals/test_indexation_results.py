# -*- coding: utf-8 -*-
# Copyright 2016-2020 The pyXem developers
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

from pyxem.signals.indexation_results import GenericMatchingResults,TemplateMatchingResults
from pyxem.signals.diffraction_vectors import DiffractionVectors


def test_init_GenericMatchingResults():
    _ = GenericMatchingResults(np.empty((10,10,10,5)))

@pytest.mark.xfail(strict=True)
def test_init_GenericMatchingResults_bad():
    # final dimension must be 5
    _ = GenericMatchingResults(np.empty((10,10,10,6)))

def test_TemplateMatchingResults_to_crystal_map():
    pass

def test_VectorMatchingResults_to_crystal_map():
    pass


@pytest.mark.parametrize(
    "overwrite, result_hkl, current_hkl, expected_hkl",
    [(True, [0, 0, 1], None, [0, 0, 1]), (False, [0, 0, 1], None, [0, 0, 1]),],
)
def test_vector_get_indexed_diffraction_vectors(
    overwrite, result_hkl, current_hkl, expected_hkl
):
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
