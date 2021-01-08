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
import dask.array as da
import pytest
from matplotlib import pyplot as plt

import hyperspy.api as hs

from pyxem.signals import Diffraction1D, LazyDiffraction1D


class TestComputeAndAsLazy1D:
    def test_2d_data_compute(self):
        dask_array = da.random.random((100, 150), chunks=(50, 50))
        s = LazyDiffraction1D(dask_array)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s.compute()
        assert isinstance(s, Diffraction1D)
        assert not hasattr(s.data, "compute")
        assert s.axes_manager[0].scale == scale0
        assert s.axes_manager[1].scale == scale1
        assert s.metadata.Test == metadata_string
        assert dask_array.shape == s.data.shape

    def test_3d_data_compute(self):
        dask_array = da.random.random((4, 10, 15), chunks=(1, 10, 15))
        s = LazyDiffraction1D(dask_array)
        s.compute()
        assert isinstance(s, Diffraction1D)
        assert dask_array.shape == s.data.shape

    def test_2d_data_as_lazy(self):
        data = np.random.random((100, 150))
        s = Diffraction1D(data)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s_lazy = s.as_lazy()
        assert isinstance(s_lazy, LazyDiffraction1D)
        assert hasattr(s_lazy.data, "compute")
        assert s_lazy.axes_manager[0].scale == scale0
        assert s_lazy.axes_manager[1].scale == scale1
        assert s_lazy.metadata.Test == metadata_string
        assert data.shape == s_lazy.data.shape

    def test_3d_data_as_lazy(self):
        data = np.random.random((4, 10, 15))
        s = Diffraction1D(data)
        s_lazy = s.as_lazy()
        assert isinstance(s_lazy, LazyDiffraction1D)
        assert data.shape == s_lazy.data.shape


class TestDecomposition:
    def test_decomposition_is_performed(self, electron_diffraction1d):
        s = Diffraction1D(electron_diffraction1d)
        s.decomposition()
        assert s.learning_results is not None

    def test_decomposition_class_assignment(self, electron_diffraction1d):
        s = Diffraction1D(electron_diffraction1d)
        s.decomposition()
        assert isinstance(s, Diffraction1D)


class TestVirtualImaging:
    # Tests that virtual imaging runs without failure

    def test_plot_integrated_intensity(self, electron_diffraction1d):
        roi = hs.roi.SpanROI(left=1.0, right=2.0)
        plt.ion()  # to make plotting non-blocking
        electron_diffraction1d.plot_integrated_intensity(roi)
        plt.close("all")

    def test_get_integrated_intensity(self, electron_diffraction1d):
        roi = hs.roi.SpanROI(left=1.0, right=2.0)
        vi = electron_diffraction1d.get_integrated_intensity(roi)
        assert vi.data.shape == (2, 2)
        assert vi.axes_manager.signal_dimension == 2
        assert vi.axes_manager.navigation_dimension == 0

    @pytest.mark.parametrize("out_signal_axes", [None, (0, 1), (1, 2), ("x", "y")])
    def test_get_integrated_intensity_stack(
        self, electron_diffraction1d, out_signal_axes
    ):
        s = hs.stack([electron_diffraction1d] * 3)
        s.axes_manager.navigation_axes[0].name = "x"
        s.axes_manager.navigation_axes[1].name = "y"

        roi = hs.roi.SpanROI(left=1.0, right=2.0)
        vi = s.get_integrated_intensity(roi, out_signal_axes)
        assert vi.axes_manager.signal_dimension == 2
        assert vi.axes_manager.navigation_dimension == 1
        if out_signal_axes == (1, 2):
            assert vi.data.shape == (2, 3, 2)
            assert vi.axes_manager.navigation_size == 2
            assert vi.axes_manager.signal_shape == (2, 3)
        else:
            assert vi.data.shape == (3, 2, 2)
            assert vi.axes_manager.navigation_size == 3
            assert vi.axes_manager.signal_shape == (2, 2)
