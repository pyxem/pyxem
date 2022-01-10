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

import pytest

import diffpy
import hyperspy.api as hs
import numpy as np

from diffsims.generators.diffraction_generator import DiffractionGenerator
from diffsims.generators.library_generator import DiffractionLibraryGenerator
from diffsims.libraries.structure_library import StructureLibrary
from pyxem.utils import calibration_utils


class TestCalibrations:
    @pytest.fixture
    def test_patterns(self):
        patterns = hs.signals.Signal2D(np.zeros((3, 1, 128, 128)))
        _001_indexs = (
            np.array([31, 31, 31, 64, 64, 97, 97, 97]),
            np.array([31, 64, 97, 31, 97, 31, 64, 97]),
        )
        _101_indexs = (
            np.array([17, 17, 17, 64, 64, 111, 111, 111]),
            np.array([31, 64, 97, 31, 97, 31, 64, 97]),
        )
        _111_indexs = (
            np.array([23, 23, 64, 64, 105, 105]),
            np.array([40, 88, 17, 111, 40, 88]),
        )
        for i in range(8):
            patterns.inav[0, 0].isig[_001_indexs[1][i], _001_indexs[0][i]] = 24.0
            patterns.inav[0, 1].isig[_101_indexs[1][i], _101_indexs[0][i]] = 24.0
        for i in range(6):
            patterns.inav[0, 2].isig[_111_indexs[1][i], _111_indexs[0][i]] = 24.0
        return patterns

    @pytest.fixture
    def test_lib_gen(self):
        diff_gen = DiffractionGenerator(
            accelerating_voltage=200,
            precession_angle=1,
            scattering_params=None,
            shape_factor_model="linear",
            minimum_intensity=0.1,
        )

        lib_gen = DiffractionLibraryGenerator(diff_gen)
        return lib_gen

    @pytest.fixture
    def test_library_phases(self):

        latt = diffpy.structure.lattice.Lattice(3, 3, 3, 90, 90, 90)
        atom = diffpy.structure.atom.Atom(atype="Ni", xyz=[0, 0, 0], lattice=latt)
        structure = diffpy.structure.Structure(atoms=[atom], lattice=latt)

        library_phases_test = StructureLibrary(
            ["Test"],
            [structure],
            [np.array([(0, 0, 90), (0, 44, 90), (0, 54.735, 45)])],
        )
        return library_phases_test

    def test_find_diffraction_calibration(
        self, test_patterns, test_lib_gen, test_library_phases
    ):
        cal, corrlines, cals = calibration_utils.find_diffraction_calibration(
            test_patterns,
            0.0097,
            test_library_phases,
            test_lib_gen,
            10,
            max_excitation_error=0.08,
        )
        np.testing.assert_array_equal(cals, np.array([0.009991, 0.010088, 0.009991]))

    def test_calibration_iteration(
        self, test_patterns, test_lib_gen, test_library_phases
    ):
        test_corrlines = calibration_utils._calibration_iteration(
            test_patterns,
            0.0097,
            test_library_phases,
            test_lib_gen,
            0.0097 * 0.01,
            2,
            3,
            max_excitation_error=0.08,
        )
        np.testing.assert_array_equal(
            test_corrlines,
            np.array(
                [
                    [
                        [0.0097, 0.0097, 0.0097],
                        [0.04868166319084026, 0.030438239690292822, 0.0],
                    ],
                    [
                        [0.0097, 0.0097, 0.0097],
                        [0.048681663190840262, 0.030438239690292822, 0.0],
                    ],
                ]
            ),
        )

    def test_create_check_diflib(
        self, test_patterns, test_lib_gen, test_library_phases
    ):
        test_corrs = calibration_utils._create_check_diflib(
            test_patterns,
            0.0097,
            test_library_phases,
            test_lib_gen,
            max_excitation_error=0.08,
        )
        np.testing.assert_array_equal(
            test_corrs, np.array([0.04868166319084026, 0.030438239690292822, 0.0])
        )
