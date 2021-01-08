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

import pytest
import numpy as np
import dask.array as da

from hyperspy.signals import Signal2D

from pyxem.signals import (
    PolarDiffraction2D,
    LazyPolarDiffraction2D,
    Correlation2D,
    Power2D,
)


class TestComputeAndAsLazy2D:
    def test_2d_data_compute(self):
        dask_array = da.random.random((100, 150), chunks=(50, 50))
        s = LazyPolarDiffraction2D(dask_array)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s.compute()
        assert s.__class__ == PolarDiffraction2D
        assert not hasattr(s.data, "compute")
        assert s.axes_manager[0].scale == scale0
        assert s.axes_manager[1].scale == scale1
        assert s.metadata.Test == metadata_string
        assert dask_array.shape == s.data.shape

    def test_4d_data_compute(self):
        dask_array = da.random.random((4, 4, 10, 15), chunks=(1, 1, 10, 15))
        s = LazyPolarDiffraction2D(dask_array)
        s.compute()
        assert s.__class__ == PolarDiffraction2D
        assert dask_array.shape == s.data.shape

    def test_2d_data_as_lazy(self):
        data = np.random.random((100, 150))
        s = PolarDiffraction2D(data)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s_lazy = s.as_lazy()
        assert s_lazy.__class__ == LazyPolarDiffraction2D
        assert hasattr(s_lazy.data, "compute")
        assert s_lazy.axes_manager[0].scale == scale0
        assert s_lazy.axes_manager[1].scale == scale1
        assert s_lazy.metadata.Test == metadata_string
        assert data.shape == s_lazy.data.shape

    def test_4d_data_as_lazy(self):
        data = np.random.random((4, 10, 15))
        s = PolarDiffraction2D(data)
        s_lazy = s.as_lazy()
        assert s_lazy.__class__ == LazyPolarDiffraction2D
        assert data.shape == s_lazy.data.shape


class TestCorrelations:
    @pytest.fixture
    def flat_pattern(self):
        pd = PolarDiffraction2D(data=np.ones(shape=(2, 2, 5, 5)))
        pd.axes_manager.signal_axes[0].scale = 0.5
        pd.axes_manager.signal_axes[0].name = "theta"
        pd.axes_manager.signal_axes[1].scale = 2
        pd.axes_manager.signal_axes[1].name = "k"
        return pd

    def test_correlation_signal(self, flat_pattern):
        ac = flat_pattern.get_angular_correlation()
        assert isinstance(ac, Correlation2D)

    def test_axes_transfer(self, flat_pattern):
        ac = flat_pattern.get_angular_correlation()
        assert (
            ac.axes_manager.signal_axes[0].scale
            == flat_pattern.axes_manager.signal_axes[0].scale
        )
        assert (
            ac.axes_manager.signal_axes[1].scale
            == flat_pattern.axes_manager.signal_axes[1].scale
        )
        assert (
            ac.axes_manager.signal_axes[1].name
            == flat_pattern.axes_manager.signal_axes[1].name
        )

    @pytest.mark.parametrize(
        "mask", [None, np.zeros(shape=(5, 5)), Signal2D(np.zeros(shape=(2, 2, 5, 5)))]
    )
    def test_masking_correlation(self, flat_pattern, mask):
        ap = flat_pattern.get_angular_correlation(mask=mask)
        assert isinstance(ap, Correlation2D)

    def test_correlation_inplace(self, flat_pattern):
        ac = flat_pattern.get_angular_correlation(inplace=True)
        assert ac is None
        assert isinstance(flat_pattern, Correlation2D)

    @pytest.mark.parametrize(
        "mask", [None, np.zeros(shape=(5, 5)), Signal2D(np.zeros(shape=(2, 2, 5, 5)))]
    )
    def test_masking_angular_power(self, flat_pattern, mask):
        ap = flat_pattern.get_angular_power(mask=mask)
        print(ap)
        assert isinstance(ap, Power2D)

    def test_axes_transfer_power(self, flat_pattern):
        ac = flat_pattern.get_angular_power()
        assert ac.axes_manager.signal_axes[0].scale == 1
        assert (
            ac.axes_manager.signal_axes[1].scale
            == flat_pattern.axes_manager.signal_axes[1].scale
        )
        assert (
            ac.axes_manager.signal_axes[1].name
            == flat_pattern.axes_manager.signal_axes[1].name
        )

    def test_power_inplace(self, flat_pattern):
        ac = flat_pattern.get_angular_power(inplace=True)
        assert ac is None
        assert isinstance(flat_pattern, Power2D)


class TestDecomposition:
    def test_decomposition_is_performed(self, diffraction_pattern):
        s = PolarDiffraction2D(diffraction_pattern)
        s.decomposition()
        assert s.learning_results is not None

    def test_decomposition_class_assignment(self, diffraction_pattern):
        s = PolarDiffraction2D(diffraction_pattern)
        s.decomposition()
        assert isinstance(s, PolarDiffraction2D)
