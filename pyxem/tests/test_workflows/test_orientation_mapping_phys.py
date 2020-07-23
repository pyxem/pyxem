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
import pyxem as pxm
import hyperspy.api as hs
import diffpy.structure

from transforms3d.euler import euler2mat

from diffsims.generators.library_generator import VectorLibraryGenerator
from diffsims.libraries.structure_library import StructureLibrary
from diffsims.utils.sim_utils import get_kinematical_intensities

from pyxem.generators.indexation_generator import IndexationGenerator
from pyxem.generators.indexation_generator import VectorIndexationGenerator
from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D
from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.utils.indexation_utils import peaks_from_best_template
from pyxem.utils.indexation_utils import peaks_from_best_vector_match
from pyxem.utils.indexation_utils import OrientationResult
from pyxem.utils.sim_utils import sim_as_signal

half_side_length = 72

""" Template Matching set up """


def create_Ortho():
    latt = diffpy.structure.lattice.Lattice(3, 4, 5, 90, 90, 90)
    atom = diffpy.structure.atom.Atom(atype="Zn", xyz=[0, 0, 0], lattice=latt)
    return diffpy.structure.Structure(atoms=[atom], lattice=latt)


def create_Hex():
    latt = diffpy.structure.lattice.Lattice(3, 3, 5, 90, 90, 120)
    atom = diffpy.structure.atom.Atom(atype="Ni", xyz=[0, 0, 0], lattice=latt)
    return diffpy.structure.Structure(atoms=[atom], lattice=latt)


@pytest.fixture
def edc():
    return pxm.DiffractionGenerator(300, 5e-2)


@pytest.fixture
def rot_list():
    from itertools import product

    a, b, c = np.arange(0, 5, step=1), [0], [0]
    rot_list_temp = list(product(a, b, c))  # rotations around A
    a, b, c = [0], [90], np.arange(0, 5, step=1)
    rot_list_temp += list(product(a, b, c))  # rotations around B
    a, b, c = [90], [90], np.arange(0, 5, step=1)
    rot_list_temp += list(product(a, b, c))  # rotations around C
    return rot_list_temp


def get_template_library(structure, rot_list, edc):
    diff_gen = pxm.DiffractionLibraryGenerator(edc)
    struc_lib = StructureLibrary(["A"], [structure], [rot_list])
    library = diff_gen.get_diffraction_library(
        struc_lib,
        calibration=1 / half_side_length,
        reciprocal_radius=0.8,
        half_shape=(half_side_length, half_side_length),
        with_direct_beam=False,
    )
    library.diffraction_generator = edc
    library.reciprocal_radius = 0.8
    library.with_direct_beam = False
    return library


def generate_diffraction_patterns(structure, edc):
    # generates 4, dumb patterns in a 2x2 array, rotated around c axis
    dp_library = get_template_library(structure, [(90, 90, 4)], edc)
    for sim in dp_library["A"]["simulations"]:
        pattern = sim_as_signal(sim, 2 * half_side_length, 0.025, 1).data
    dp = pxm.ElectronDiffraction2D([[pattern, pattern], [pattern, pattern]])
    return dp


def generate_difficult_diffraction_patterns(structure, edc):
    # generates 4, dumb patterns in a 2x2 array, rotated around c axis
    difficult_diffraction_half_side_length = 73
    dp_library = get_template_library(structure, [(90, 90, 4)], edc)
    for sim in dp_library["A"]["simulations"]:
        pattern = sim_as_signal(
            sim, 2 * difficult_diffraction_half_side_length, 0.025, 1
        ).data
    dp = pxm.ElectronDiffraction2D([[pattern, pattern], [pattern, pattern]])
    return dp


def get_template_match_results(structure, edc, rot_list, mask=None):
    dp = generate_diffraction_patterns(structure, edc)
    library = get_template_library(structure, rot_list, edc)
    indexer = IndexationGenerator(dp, library)
    return indexer.correlate(mask=mask, method="zero_mean_normalized_correlation")


def get_template_match_results_fullframe(structure, edc, rot_list, mask=None):
    dp = generate_diffraction_patterns(structure, edc)
    library = get_template_library(structure, rot_list, edc)
    indexer = IndexationGenerator(dp, library)
    return indexer.correlate(mask=mask, method="full_frame_correlation")


def get_template_match_results_fullframe_bad_size(structure, edc, rot_list, mask=None):
    dp = generate_difficult_diffraction_patterns(structure, edc)
    library = get_template_library(structure, rot_list, edc)
    indexer = IndexationGenerator(dp, library)
    return indexer.correlate(mask=mask, method="full_frame_correlation")


""" Tests for template matching """


@pytest.mark.parametrize("structure", [create_Ortho(), create_Hex()])
def test_orientation_mapping_physical(structure, rot_list, edc):
    M = get_template_match_results(structure, edc, rot_list)
    assert np.all(M.inav[0, 0] == M.inav[1, 0])
    match_data = M.inav[0, 0].isig[1].data
    for result_number in [0, 1, 2]:
        np.testing.assert_allclose(
            match_data[result_number][:2], [90, 90]
        )  # always looking down c


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize("structure", [create_Ortho(), create_Hex()])
def test_fullframe_bad_size(structure, rot_list, edc):
    high_scores = get_template_match_results_fullframe_bad_size(
        structure, edc, rot_list
    )


@pytest.mark.parametrize("structure", [create_Ortho(), create_Hex()])
def test_fullframe_orientation_mapping_physical(structure, rot_list, edc):
    M = get_template_match_results_fullframe(structure, edc, rot_list)
    assert np.all(M.inav[0, 0] == M.inav[1, 0])
    match_data = M.inav[0, 0].isig[1].data
    for result_number in [0, 1, 2]:
        np.testing.assert_allclose(
            match_data[result_number][:2], [90, 90]
        )  # always looking down c


def test_masked_template_matching(default_structure, rot_list, edc):
    mask = hs.signals.Signal1D(([[[1], [1]], [[0], [1]]]))
    M = get_template_match_results(default_structure, edc, rot_list, mask)
    assert np.all(np.equal(M.inav[0, 1].data, None))


def test_masked_fullframe_template_matching(default_structure, rot_list, edc):
    mask = hs.signals.Signal1D(([[[1], [1]], [[0], [1]]]))
    M = get_template_match_results_fullframe(default_structure, edc, rot_list, mask)
    assert np.all(np.equal(M.inav[0, 1].data, None))


""" Testing Vector Matching Results """


def get_vector_match_results(structure, rot_list, edc):
    diffraction_library = get_template_library(structure, rot_list, edc)
    peak_lists = []
    for pixel_coords in diffraction_library["A"]["pixel_coords"]:
        peak_lists.append(pixel_coords)
    peaks = DiffractionVectors(
        (np.array([peak_lists, peak_lists]) - half_side_length) / half_side_length
    )
    peaks.axes_manager.set_signal_dimension(2)
    peaks.calculate_cartesian_coordinates(200, 0.2)
    peaks.cartesian.axes_manager.set_signal_dimension(2)
    structure_library = StructureLibrary(["A"], [structure], [[]])
    library_generator = VectorLibraryGenerator(structure_library)
    vector_library = library_generator.get_vector_library(1)
    indexation_generator = VectorIndexationGenerator(peaks, vector_library)
    indexation = indexation_generator.index_vectors(
        mag_tol=1.5 / half_side_length,
        angle_tol=1,
        index_error_tol=0.2,
        n_peaks_to_index=5,
        n_best=2,
    )
    return diffraction_library, indexation


@pytest.mark.parametrize(
    "structure, rot_list", [(create_Hex(), [(0, 0, 10), (0, 0, 0)])]
)
def test_vector_matching_physical(structure, rot_list, edc):
    _, match_results = get_vector_match_results(structure, rot_list, edc)
    assert match_results.data.shape == (2, 2)  # 2x2 rotations, 2 best peaks, 5 values
    isinstance(match_results.data[0, 0][0], OrientationResult)
    np.testing.assert_allclose(
        match_results.data[0, 0][0].match_rate, 1.0
    )  # match rate for best orientation
    np.testing.assert_allclose(
        match_results.data[0, 1][0].match_rate, 1.0
    )  # match rate for best orientation


@pytest.mark.parametrize(
    "structure, rot_list", [(create_Hex(), [(0, 0, 10), (0, 0, 0)])]
)
def test_peaks_from_best_vector_match(structure, rot_list, edc):
    library, match_results = get_vector_match_results(structure, rot_list, edc)
    peaks = match_results.map(
        peaks_from_best_vector_match, library=library, inplace=False
    )
    # Unordered compare within absolute tolerance
    for i in range(2):
        lib = library["A"]["simulations"][i].coordinates[:, :2]
        for p in peaks.data[0, i]:
            assert (
                np.isclose(p[0], lib[:, 0], atol=0.1).any()
                and np.isclose(p[1], lib[:, 1], atol=0.1).any()
            )


@pytest.mark.parametrize(
    "structure, rot_list", [(create_Hex(), [(0, 0, 10), (0, 0, 0)])]
)
def test_plot_best_vector_matching_results_on_signal(structure, rot_list, edc):
    library, match_results = get_vector_match_results(structure, rot_list, edc)
    match_results.data = np.vstack(
        (match_results.data, match_results.data)
    )  # Hyperspy can only add markers to square signals
    dp = ElectronDiffraction2D(2 * [2 * [np.zeros((144, 144))]])
    match_results.plot_best_matching_results_on_signal(dp)
