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

from pyxem.signals import Correlation2D, LazyCorrelation2D, Power2D


class TestComputeAndAsLazy2D:
    def test_2d_data_compute(self):
        dask_array = da.random.random((100, 150), chunks=(50, 50))
        s = LazyCorrelation2D(dask_array)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s.compute()
        assert s.__class__ == Correlation2D
        assert not hasattr(s.data, "compute")
        assert s.axes_manager[0].scale == scale0
        assert s.axes_manager[1].scale == scale1
        assert s.metadata.Test == metadata_string
        assert dask_array.shape == s.data.shape

    def test_4d_data_compute(self):
        dask_array = da.random.random((4, 4, 10, 15), chunks=(1, 1, 10, 15))
        s = LazyCorrelation2D(dask_array)
        s.compute()
        assert s.__class__ == Correlation2D
        assert dask_array.shape == s.data.shape

    def test_2d_data_as_lazy(self):
        data = np.random.random((100, 150))
        s = Correlation2D(data)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s_lazy = s.as_lazy()
        assert s_lazy.__class__ == LazyCorrelation2D
        assert hasattr(s_lazy.data, "compute")
        assert s_lazy.axes_manager[0].scale == scale0
        assert s_lazy.axes_manager[1].scale == scale1
        assert s_lazy.metadata.Test == metadata_string
        assert data.shape == s_lazy.data.shape

    def test_4d_data_as_lazy(self):
        data = np.random.random((4, 10, 15))
        s = Correlation2D(data)
        s_lazy = s.as_lazy()
        assert s_lazy.__class__ == LazyCorrelation2D
        assert data.shape == s_lazy.data.shape


class TestGetPower:
    @pytest.fixture
    def flat_pattern(self):
        pd = Correlation2D(data=np.ones(shape=(2, 2, 5, 5)))
        pd.axes_manager.signal_axes[0].scale = 0.5
        pd.axes_manager.signal_axes[0].name = "theta"
        pd.axes_manager.signal_axes[1].scale = 2
        pd.axes_manager.signal_axes[1].name = "k"
        return pd

    def test_power_signal(self, flat_pattern):
        power = flat_pattern.get_angular_power()
        assert isinstance(power, Power2D)

    def test_power_signal_inplace(self, flat_pattern):
        power = flat_pattern.get_angular_power(inplace=True)
        assert isinstance(flat_pattern, Power2D)
        assert power is None

    def test_summed_power_signal(self, flat_pattern):
        power = flat_pattern.get_summed_angular_power(inplace=False)
        assert isinstance(power, Power2D)

    def test_summed_power_signal_inplace(self, flat_pattern):
        power = flat_pattern.get_summed_angular_power(inplace=True)
        assert isinstance(flat_pattern, Power2D)
        assert power is None


class TestDecomposition:
    def test_decomposition_is_performed(self, diffraction_pattern):
        s = Correlation2D(diffraction_pattern)
        s.decomposition()
        assert s.learning_results is not None

    def test_decomposition_class_assignment(self, diffraction_pattern):
        s = Correlation2D(diffraction_pattern)
        s.decomposition()
        assert isinstance(s, Correlation2D)
