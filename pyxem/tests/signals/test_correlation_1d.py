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

from pyxem.signals import Correlation1D, PolarDiffraction2D


class TestCorrelation1D:
    @pytest.fixture
    def flat_pattern(self):
        pd = Correlation1D(data=np.ones(shape=(2, 2, 20)))
        pd.axes_manager.signal_axes[0].scale = 0.5
        pd.axes_manager.signal_axes[0].name = "theta"
        return pd

    @pytest.mark.parametrize("method", ["max", "first"])
    def test_symmetry_stem_max_first(self, flat_pattern, method):
        sym_coeff = flat_pattern.get_symmetry_coefficient(method=method)
        np.testing.assert_array_almost_equal(sym_coeff.data, sym_coeff.data[0, 0, 0])

    def test_symmetry_stem_max(self, flat_pattern):
        sym_coeff = flat_pattern.get_symmetry_coefficient(method="max")
        np.testing.assert_array_almost_equal(sym_coeff.data, sym_coeff.data[0, 0, 0])

    def test_symmetry_stem_average(self, flat_pattern):
        sym_coeff = flat_pattern.get_symmetry_coefficient(
            method="average", angular_range=0.01, normalize=True
        )
        np.testing.assert_array_almost_equal(sym_coeff.data, sym_coeff.data[0, 0, 0])

    def test_method_not_supported(self, flat_pattern):
        with pytest.raises(ValueError):
            flat_pattern.get_symmetry_coefficient(
                method="avg", angular_range=0.01, normalize=True
            )

    def test_include_duplicates(self, flat_pattern):
        sym_coeff = flat_pattern.get_symmetry_coefficient(
            method="average",
            angular_range=0.02,
            include_duplicates=True,
            normalize=True,
            symmetries=[3,4,5,6,7,8],
        )
        np.testing.assert_array_almost_equal(sym_coeff.data, sym_coeff.data[0, 0, 0])

    def test_corr(self):
        rand_number = np.random.random(10)
        tiled_random = np.tile(rand_number, (2, 3, 2))
        p = PolarDiffraction2D(tiled_random)
        cor = p.get_pearson_correlation()
        sym_coeff = cor.get_symmetry_coefficient(
            method="average",
            angular_range=0.0,
            include_duplicates=True,
            normalize=False,
        )
        np.testing.assert_almost_equal(sym_coeff.data[0, 0], 1)

    def test_2_fold_corr(self):
        rand_number = np.random.random(10)
        tiled_random = np.tile(rand_number, (2, 3, 2))
        p = PolarDiffraction2D(tiled_random)
        cor = p.get_pearson_correlation()
        sym_coeff = cor.get_symmetry_coefficient(
            method="average",
            angular_range=0.0,
            include_duplicates=True,
            normalize=False,
        )
        np.testing.assert_almost_equal(sym_coeff.data[0, 0], 1)