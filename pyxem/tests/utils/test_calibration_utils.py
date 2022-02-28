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
import numpy as np

from pyxem.utils import calibration_utils


@pytest.mark.skip(reason="This functionality already smells, skipping while new things are built")
class TestCalibrations:
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
        np.testing.assert_allclose(
            cals, np.array([0.009991, 0.010088, 0.009991]), atol=1e-6
        )

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
        true_corrlines = np.array(
            [
                [
                    [0.0097, 0.0097, 0.0097],
                    [0.085312, 0.056166, 0.0],
                ],
                [
                    [0.0097, 0.0097, 0.0097],
                    [0.085312, 0.056166, 0.0],
                ],
            ]
        )
        np.testing.assert_allclose(
            test_corrlines,
            true_corrlines,
            atol=1e-6,
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
        np.testing.assert_allclose(
            test_corrs,
            np.array([0.085313, 0.056166, 0.0]),
            atol=1e-6,
        )
