# -*- coding: utf-8 -*-
# Copyright 2018 The pyXem developers
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

from pyxem.generators.indexation_generator import IndexationGenerator, VectorIndexationGenerator
from pyxem.generators.library_generator import VectorLibraryGenerator
from pyxem.libraries.structure_library import StructureLibrary
from pyxem.signals.electron_diffraction import ElectronDiffraction
from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.utils.sim_utils import (peaks_from_best_template,
                                   peaks_from_best_vector_match,
                                   get_kinematical_intensities)

"""
The test are designed to make sure orientation mapping works when actual
rotation are considered.

Specifically we test (for both an orthorhombic and hexagonal samples) that:

- The algorithm can tell the difference between down a, down b and down c axes
"""


half_side_length = 72


def create_Ortho():
    latt = diffpy.structure.lattice.Lattice(3, 4, 5, 90, 90, 90)
    atom = diffpy.structure.atom.Atom(atype='Zn', xyz=[0, 0, 0], lattice=latt)
    return diffpy.structure.Structure(atoms=[atom], lattice=latt)


def create_Hex():
    latt = diffpy.structure.lattice.Lattice(3, 3, 5, 90, 90, 120)
    atom = diffpy.structure.atom.Atom(atype='Ni', xyz=[0, 0, 0], lattice=latt)
    return diffpy.structure.Structure(atoms=[atom], lattice=latt)


def create_wurtzite(rotation=None):
    """Construct a hexagonal P63mc GaAs Wurtzite structure. """
    if rotation is None:
        rotation = np.eye(3)
    a = 4.053
    c = 6.680
    lattice = diffpy.structure.lattice.Lattice(a, a, c, 90, 90, 120)
    atom_list = []
    for x, y, z in [(1 / 3, 2 / 3, 0), (2 / 3, 1 / 3, 1 / 2)]:
        atom_list.append(
            diffpy.structure.atom.Atom(
                atype='Ga',
                xyz=[x, y, z],
                lattice=lattice))
        atom_list.append(
            diffpy.structure.atom.Atom(
                atype='As',
                xyz=[x + 3 / 8, y + 3 / 8, z + 3 / 8],
                lattice=lattice))
    return diffpy.structure.Structure(atoms=atom_list, lattice=lattice)


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


@pytest.fixture
def pattern_list():
    return [(0, 0, 2)]


def get_template_library(structure, rot_list, edc):
    diff_gen = pxm.DiffractionLibraryGenerator(edc)
    struc_lib = StructureLibrary(['A'], [structure], [rot_list])
    library = diff_gen.get_diffraction_library(struc_lib,
                                               calibration=1 / half_side_length,
                                               reciprocal_radius=0.8,
                                               half_shape=(half_side_length,
                                                           half_side_length),
                                               with_direct_beam=False)
    library.diffraction_generator = edc
    library.reciprocal_radius = 0.8
    library.with_direct_beam = False
    return library


def get_template_match_results(structure, pattern_list, edc, rot_list, mask=None, inplane_rotations=[0]):
    dp_library = get_template_library(structure, pattern_list, edc)
    for sim in dp_library['A']['simulations']:
        pattern = (sim.as_signal(2 * half_side_length, 0.025, 1).data)
    dp = pxm.ElectronDiffraction([[pattern, pattern], [pattern, pattern]])
    library = get_template_library(structure, rot_list, edc)
    indexer = IndexationGenerator(dp, library)
    return indexer.correlate(mask=mask, inplane_rotations=inplane_rotations)


def get_vector_match_results(structure, rot_list, edc):
    diffraction_library = get_template_library(structure, rot_list, edc)
    peak_lists = []
    for pixel_coords in diffraction_library['A']['pixel_coords']:
        peak_lists.append(pixel_coords)
    peaks = DiffractionVectors((np.array([peak_lists, peak_lists]) - half_side_length) / half_side_length)
    peaks.axes_manager.set_signal_dimension(2)
    peaks.calculate_cartesian_coordinates(200, 0.2)
    peaks.cartesian.axes_manager.set_signal_dimension(2)
    structure_library = StructureLibrary(['A'], [structure], [[]])
    library_generator = VectorLibraryGenerator(structure_library)
    vector_library = library_generator.get_vector_library(1)
    indexation_generator = VectorIndexationGenerator(peaks, vector_library)
    indexation = indexation_generator.index_vectors(
        mag_tol=1.5 / half_side_length,
        angle_tol=1,
        index_error_tol=0.2,
        n_peaks_to_index=5,
        n_best=2,
        keys=['A'])
    return diffraction_library, indexation


@pytest.mark.parametrize("structure", [create_Ortho(), create_Hex()])
def test_orientation_mapping_physical(structure, rot_list, pattern_list, edc):
    M = get_template_match_results(structure, pattern_list, edc, rot_list)
    assert np.all(M.inav[0, 0] == M.inav[1, 0])
    match_data = M.inav[0, 0].isig[:4, 0].data
    assert match_data[0] == 0
    np.testing.assert_allclose(match_data[1], [2, 0, 0])


def test_masked_OM(default_structure, rot_list, pattern_list, edc):
    mask = hs.signals.Signal1D(([[[1], [1]], [[0], [1]]]))
    M = get_template_match_results(default_structure, pattern_list, edc, rot_list, mask)
    assert np.all(np.equal(M.inav[0, 1].data, None))


def expected_best_peaks_pattern_list(library, _):
    return library.get_library_entry("A", (0, 0, 0))['Sim'].coordinates[:, :2]


def expected_best_peaks_rotated(library, rotation_euler):
    angle = np.deg2rad(rotation_euler[0][2])
    rotation = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]])
    coords = library.get_library_entry("A", (0, 0, 0))['Sim'].coordinates[:, :2]
    return (rotation @ coords.T).T


@pytest.mark.parametrize('pattern_list, inplane_rotations, get_expected_peaks', [
    ([(0, 0, 2)], [0], expected_best_peaks_pattern_list),
    ([(0, 0, 45)], np.arange(0, 360, 1), expected_best_peaks_rotated),
])
def test_generate_peaks_from_best_template(default_structure, rot_list, pattern_list, edc, get_expected_peaks, inplane_rotations):
    library = get_template_library(default_structure, rot_list, edc)
    M = get_template_match_results(default_structure, pattern_list, edc, rot_list, inplane_rotations=inplane_rotations)
    peaks = M.map(peaks_from_best_template,
                  library=library,
                  inplace=False)
    expected_peaks = get_expected_peaks(library, pattern_list)
    for expected_peak in expected_peaks:
        assert np.any(np.isclose(np.linalg.norm(peaks.inav[0, 0] - expected_peak, axis=1), 0, atol=0.03))


@pytest.mark.parametrize('structure, rot_list', [(create_Hex(), [(0, 0, 10), (0, 0, 0)])])
def test_vector_matching_physical(structure, rot_list, edc):
    _, match_results = get_vector_match_results(structure, rot_list, edc)
    assert match_results.data.shape == (2, 2, 2, 5)  # 2x2 rotations, 2 best peaks, 5 values
    np.testing.assert_allclose(match_results.data[0, 0, 0, 2], 1.0)  # match rate for best orientation
    np.testing.assert_allclose(match_results.data[0, 1, 0, 2], 1.0)  # match rate for best orientation


@pytest.mark.parametrize('structure, rot_list', [(create_Hex(), [(0, 0, 10), (0, 0, 0)])])
def test_peaks_from_best_vector_match(structure, rot_list, edc):
    library, match_results = get_vector_match_results(structure, rot_list, edc)
    peaks = match_results.map(peaks_from_best_vector_match,
                              library=library,
                              inplace=False)
    # Unordered compare within absolute tolerance
    for i in range(2):
        lib = library['A']['simulations'][i].coordinates[:, :2]
        for p in peaks.data[0, i]:
            assert np.isclose(p[0], lib[:, 0], atol=0.1).any() and np.isclose(p[1], lib[:, 1], atol=0.1).any()


@pytest.mark.parametrize('structure, rot_list', [(create_Hex(), [(0, 0, 10), (0, 0, 0)])])
def test_plot_best_matching_results_on_signal_vector(structure, rot_list, edc):
    # Just test that the code runs
    library, match_results = get_vector_match_results(structure, rot_list, edc)
    # Hyperspy can only add markers to square signals
    match_results.data = np.vstack((match_results.data, match_results.data))
    dp = ElectronDiffraction(2 * [2 * [np.zeros((144, 144))]])
    match_results.plot_best_matching_results_on_signal(dp,
                                                       library=library)


@pytest.mark.parametrize("structure, rotation", [(create_wurtzite(), euler2mat(0, np.pi / 2, 0, 'rzxz'))])
def test_kinematic_intensities_rotation(structure, rotation):
    """Test that kinematically forbidden diffraction spots gets zero intensity also after rotation."""
    rotated_lattice = diffpy.structure.lattice.Lattice(structure.lattice)
    rotated_lattice.setLatPar(baserot=rotation)
    structure.placeInLattice(rotated_lattice)
    reciprocal_lattice = structure.lattice.reciprocal()
    g_indices = [(0, 0, 1)]
    g_hkls = reciprocal_lattice.dist(g_indices, [0, 0, 0])

    scattering_params_list = ['lobato', 'xtables']
    for scattering_params in scattering_params_list:
        intensities = get_kinematical_intensities(
            structure,
            g_indices,
            g_hkls,
            excitation_error=0,
            maximum_excitation_error=1,
            debye_waller_factors={},
            scattering_params=scattering_params)

        np.testing.assert_almost_equal(intensities, [0])


@pytest.mark.parametrize("structure, rotation", [(create_wurtzite(), euler2mat(0, np.pi / 2, 0, 'rzxz'))])
@pytest.mark.xfail(raises=NotImplementedError)
def test_kinematic_intensities_error_raise(structure, rotation):
    """Test that kinematically forbidden diffraction spots gets zero intensity also after rotation."""
    rotated_lattice = diffpy.structure.lattice.Lattice(structure.lattice)
    rotated_lattice.setLatPar(baserot=rotation)
    structure.placeInLattice(rotated_lattice)
    reciprocal_lattice = structure.lattice.reciprocal()
    g_indices = [(0, 0, 1)]
    g_hkls = reciprocal_lattice.dist(g_indices, [0, 0, 0])

    intensities = get_kinematical_intensities(
        structure,
        g_indices,
        g_hkls,
        excitation_error=0,
        maximum_excitation_error=1,
        debye_waller_factors={},
        scattering_params='_empty')
