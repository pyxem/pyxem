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
import pytest
from matplotlib import pyplot as plt
from transforms3d.euler import euler2mat

from diffsims.libraries.structure_library import StructureLibrary
from diffsims.generators.diffraction_generator import DiffractionGenerator
from diffsims.generators.library_generator import DiffractionLibraryGenerator

from pyxem.generators import TemplateIndexationGenerator
from pyxem.signals import (
    TemplateMatchingResults,
    VectorMatchingResults,
    DiffractionVectors,
)
from pyxem.utils.indexation_utils import OrientationResult


def test_TemplateMatchingResults_to_crystal_map():
    # multiphase
    t = TemplateMatchingResults(np.empty((10, 10, 10, 5)))
    t.to_crystal_map()
    # singlephase
    t = TemplateMatchingResults(np.ones((10, 10, 10, 5)))
    t.to_crystal_map()



def test_TemplateMatchingResults_plot_best_results_on_signal(
    diffraction_pattern, default_structure
):
    """ Coverage testing """
    edc = DiffractionGenerator(300)
    half_side_length = 4
    rot_list = [[0, 1, 0], [1, 0, 0]]

    diff_gen = DiffractionLibraryGenerator(edc)
    struc_lib = StructureLibrary(["A"], [default_structure], [rot_list])
    library = diff_gen.get_diffraction_library(
        struc_lib,
        calibration=1 / half_side_length,
        reciprocal_radius=0.8,
        half_shape=(half_side_length, half_side_length),
        with_direct_beam=True,
    )
    indexer = TemplateIndexationGenerator(diffraction_pattern, library)
    match_results = indexer.correlate()
    match_results.plot_best_matching_results_on_signal(
        diffraction_pattern, library=library
    )
    plt.close("all")


@pytest.fixture
def sp_vector_match_result():
    # We require (total_error of row_1 > correlation row_2)
    res = np.empty(2, dtype="object")
    res[0] = OrientationResult(
        0,
        euler2mat(*np.deg2rad([0, 0, 90]), "rzxz"),
        0.5,
        np.array([0.1, 0.05, 0.2]),
        0.1,
        1.0,
        0,
        0,
    )
    res[1] = OrientationResult(
        0,
        euler2mat(*np.deg2rad([0, 0, 90]), "rzxz"),
        0.6,
        np.array([0.1, 0.10, 0.2]),
        0.2,
        1.0,
        0,
        0,
    )
    return VectorMatchingResults(res)


@pytest.fixture
def dp_vector_match_result():
    res = np.empty(4, dtype="object")
    res = res.reshape(2, 2)
    res[0, 0] = OrientationResult(
        0,
        euler2mat(*np.deg2rad([90, 0, 0]), "rzxz"),
        0.6,
        np.array([0.1, 0.10, 0.2]),
        0.3,
        1.0,
        0,
        0,
    )
    res[0, 1] = OrientationResult(
        0,
        euler2mat(*np.deg2rad([0, 10, 20]), "rzxz"),
        0.5,
        np.array([0.1, 0.05, 0.2]),
        0.4,
        1.0,
        0,
        0,
    )
    res[1, 0] = OrientationResult(
        1,
        euler2mat(*np.deg2rad([0, 45, 45]), "rzxz"),
        0.8,
        np.array([0.1, 0.30, 0.2]),
        0.1,
        1.0,
        0,
        0,
    )
    res[1, 1] = OrientationResult(
        1,
        euler2mat(*np.deg2rad([0, 0, 90]), "rzxz"),
        0.7,
        np.array([0.1, 0.05, 0.1]),
        0.2,
        1.0,
        0,
        0,
    )
    return VectorMatchingResults(res)


def test_single_vector_get_crystallographic_map(sp_vector_match_result):
    _ = sp_vector_match_result.get_crystallographic_map()


def test_double_vector_get_crystallographic_map(dp_vector_match_result):
    _ = dp_vector_match_result.get_crystallographic_map()


@pytest.mark.parametrize(
    "overwrite, result_hkl, current_hkl, expected_hkl",
    [
        (True, [0, 0, 1], None, [0, 0, 1]),
        (False, [0, 0, 1], None, [0, 0, 1]),
    ],
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
