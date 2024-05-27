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

import numpy as np
import pytest
from matplotlib import pyplot as plt
from transforms3d.euler import euler2mat

from diffsims.libraries.structure_library import StructureLibrary
from diffsims.generators.diffraction_generator import DiffractionGenerator
from diffsims.generators.library_generator import DiffractionLibraryGenerator

from diffsims.generators.simulation_generator import SimulationGenerator
from orix.sampling import get_sample_reduced_fundamental
from orix.quaternion import Rotation, Orientation
from orix.crystal_map import CrystalMap

from pyxem.generators import TemplateIndexationGenerator
from pyxem.signals import VectorMatchingResults, DiffractionVectors, OrientationMap
from pyxem.utils.indexation_utils import OrientationResult
from pyxem.data import (
    si_grains,
    si_phase,
    si_tilt,
    si_grains_simple,
    fe_multi_phase_grains,
    fe_bcc_phase,
    fe_fcc_phase,
)
import hyperspy.api as hs


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


@pytest.mark.skip(reason="This functionality is under limited support as of 0.14.0")
def test_single_vector_get_crystallographic_map(sp_vector_match_result):
    _ = sp_vector_match_result.get_crystallographic_map()


@pytest.mark.skip(reason="This functionality is under limited support as of 0.14.0")
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


class TestOrientationResult:
    """Testing the OrientationMap class for valid outputs. These tests are based on the
    examples provided in the documentation.
    """

    @pytest.fixture
    def single_rot_orientation_result(self):
        s = si_tilt()
        s.calibration.center = None
        polar_si_tilt = s.get_azimuthal_integral2d(
            npt=100, npt_azim=360, inplace=False, mean=True
        )
        phase = si_phase()
        generator = SimulationGenerator(200)
        sim = generator.calculate_diffraction2d(
            phase,
            rotation=Rotation.from_euler(
                [0, 0, 0],
                degrees=True,
            ),
            max_excitation_error=0.1,
            reciprocal_radius=1.5,
        )
        orientation_map = polar_si_tilt.get_orientation(sim)
        return orientation_map

    @pytest.fixture
    def multi_rot_orientation_result(self):
        s, r = si_grains(return_rotations=True)
        s.calibration.center = None
        polar = s.get_azimuthal_integral2d(
            npt=100, npt_azim=180, inplace=False, mean=True
        )
        phase = si_phase()
        generator = SimulationGenerator(200, minimum_intensity=0.05)
        rotations = get_sample_reduced_fundamental(
            resolution=1, point_group=phase.point_group
        )
        sims = generator.calculate_diffraction2d(
            phase, rotation=rotations, max_excitation_error=0.1, reciprocal_radius=2
        )
        orientations = polar.get_orientation(sims)
        return orientations, r

    @pytest.fixture
    def simple_multi_rot_orientation_result(self):
        s, r = si_grains_simple(return_rotations=True)
        s.calibration.center = None
        polar = s.get_azimuthal_integral2d(
            npt=100, npt_azim=180, inplace=False, mean=True
        )
        phase = si_phase()
        generator = SimulationGenerator(200, minimum_intensity=0.05)
        rotations = get_sample_reduced_fundamental(
            resolution=1, point_group=phase.point_group
        )
        sims = generator.calculate_diffraction2d(
            phase, rotation=rotations, max_excitation_error=0.1, reciprocal_radius=2
        )
        orientations = polar.get_orientation(sims)
        return orientations, r

    @pytest.fixture
    def multi_phase_orientation_result(self):
        s = fe_multi_phase_grains()
        s.calibration.center = None
        polar = s.get_azimuthal_integral2d(
            npt=100, npt_azim=180, inplace=False, mean=True
        )
        phase = fe_fcc_phase()
        phase2 = fe_bcc_phase()

        generator = SimulationGenerator(200, minimum_intensity=0.05)
        rotations = get_sample_reduced_fundamental(
            resolution=1, point_group=phase.point_group
        )
        rotations2 = get_sample_reduced_fundamental(
            resolution=1, point_group=phase2.point_group
        )
        sims = generator.calculate_diffraction2d(
            [phase, phase2],
            rotation=[rotations, rotations2],
            max_excitation_error=0.1,
            reciprocal_radius=2,
        )
        orientations = polar.get_orientation(sims)
        return orientations

    def test_tilt_orientation_result(self, single_rot_orientation_result):
        assert isinstance(single_rot_orientation_result, OrientationMap)
        orients = single_rot_orientation_result.to_single_phase_orientations()
        # Check that the orientations are within 1 degree of the expected value
        degrees_between = orients.angle_with(
            Orientation.from_euler([0, 0, 0]), degrees=True
        )
        assert np.all(
            degrees_between[:, :5] <= 1
        )  # off by 1 degree (due to pixelation?)
        degrees_between = orients.angle_with(
            Orientation.from_euler([10, 0, 0], degrees=True), degrees=True
        )
        assert np.all(degrees_between[:, 5:] <= 1)

    def test_grain_orientation_result(self, simple_multi_rot_orientation_result):
        orientations, rotations = simple_multi_rot_orientation_result
        assert isinstance(rotations, Orientation)
        assert isinstance(orientations, OrientationMap)
        orients = orientations.to_single_phase_orientations()

        # Check that the orientations are within 2 degrees of the expected value.
        # Use 2 degrees since that is the angular resolution of the polar dataset
        degrees_between = orients.angle_with(rotations, degrees=True)
        assert np.all(np.min(degrees_between, axis=2) <= 2)

    def test_to_crystal_map(self, simple_multi_rot_orientation_result):
        orientations, rotations = simple_multi_rot_orientation_result
        crystal_map = orientations.to_crystal_map()
        assert isinstance(crystal_map, CrystalMap)
        assert np.all(crystal_map.phase_id == 0)

    def test_to_crystal_map_multi_phase(self, multi_phase_orientation_result):
        crystal_map = multi_phase_orientation_result.to_crystal_map()
        assert isinstance(crystal_map, CrystalMap)
        assert np.all(crystal_map.phase_id < 2)
