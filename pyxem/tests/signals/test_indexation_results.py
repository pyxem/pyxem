# -*- coding: utf-8 -*-
# Copyright 2016-2022 The pyXem developers
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

from hyperspy.signals import Signal2D
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
from pyxem.utils.indexation_utils import index_dataset_with_template_rotation
from pyxem.signals.indexation_results import results_dict_to_crystal_map


def test_TemplateMatchingResults_to_crystal_map():
    t = TemplateMatchingResults(np.empty((10, 10, 10, 5)))
    return t.to_crystal_map()


def test_TemplateMatchingResults_plot_best_results_on_signal(
    diffraction_pattern, default_structure
):
    """Coverage testing"""
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


def test_results_dict_to_crystal_map(test_library_phases, test_lib_gen):
    library_phases_Ni = test_library_phases
    lib_gen = test_lib_gen
    diff_lib_Ni = lib_gen.get_diffraction_library(
        library_phases_Ni,
        calibration=0.015,
        reciprocal_radius=1.182243629714282,
        half_shape=64,
        with_direct_beam=False,
        max_excitation_error=0.07,
    )
    test_set = Signal2D(np.zeros((3, 128, 128)))
    for i in range(3):
        test_pattern = diff_lib_Ni["Test"]["simulations"][i].get_diffraction_pattern(
            size=128, sigma=4
        )
        test_set.inav[i] = test_pattern

    result, phasedict = index_dataset_with_template_rotation(
        test_set,
        diff_lib_Ni,
        phases=["Test"],
        n_best=3,
    )
    return results_dict_to_crystal_map(
        result, phasedict, diffraction_library=diff_lib_Ni
    )


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
