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
import pyxem as pxm

from hyperspy.signals import Signal1D, Signal2D

from pyxem.signals.xray_diffraction2d import XrayDiffraction2D
from pyxem.signals.xray_diffraction2d import LazyXrayDiffraction2D


def test_init():
    z = np.zeros((2, 2, 2, 2))
    dp = XrayDiffraction2D(z, metadata={'Acquisition_instrument': {'SEM': 'Expensive-SEM'}})


class TestSimpleHyperspy:
    # Tests functions that assign to hyperspy metadata

    def test_set_experimental_parameters(self, diffraction_pattern):
        diffraction_pattern = XrayDiffraction2D(diffraction_pattern)
        diffraction_pattern.set_experimental_parameters(beam_energy=3,
                                                        camera_length=3,
                                                        exposure_time=1)
        assert isinstance(diffraction_pattern, XrayDiffraction2D)

    def test_set_scan_calibration(self, diffraction_pattern):
        diffraction_pattern = XrayDiffraction2D(diffraction_pattern)
        diffraction_pattern.set_scan_calibration(19)
        assert isinstance(diffraction_pattern, XrayDiffraction2D)


class TestComputeAndAsLazyXray2D:

    def test_2d_data_compute(self):
        dask_array = da.random.random((100, 150), chunks=(50, 50))
        s = LazyXrayDiffraction2D(dask_array)
        scale0, scale1, metadata_string = 0.5, 1.5, 'test'
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s.compute()
        assert s.__class__ == XrayDiffraction2D
        assert not hasattr(s.data, 'compute')
        assert s.axes_manager[0].scale == scale0
        assert s.axes_manager[1].scale == scale1
        assert s.metadata.Test == metadata_string
        assert dask_array.shape == s.data.shape

    def test_4d_data_compute(self):
        dask_array = da.random.random((4, 4, 10, 15),
                                      chunks=(1, 1, 10, 15))
        s = LazyXrayDiffraction2D(dask_array)
        s.compute()
        assert s.__class__ == XrayDiffraction2D
        assert dask_array.shape == s.data.shape

    def test_2d_data_as_lazy(self):
        data = np.random.random((100, 150))
        s = XrayDiffraction2D(data)
        scale0, scale1, metadata_string = 0.5, 1.5, 'test'
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s_lazy = s.as_lazy()
        assert s_lazy.__class__ == LazyXrayDiffraction2D
        assert hasattr(s_lazy.data, 'compute')
        assert s_lazy.axes_manager[0].scale == scale0
        assert s_lazy.axes_manager[1].scale == scale1
        assert s_lazy.metadata.Test == metadata_string
        assert data.shape == s_lazy.data.shape

    def test_4d_data_as_lazy(self):
        data = np.random.random((4, 10, 15))
        s = XrayDiffraction2D(data)
        s_lazy = s.as_lazy()
        assert s_lazy.__class__ == LazyXrayDiffraction2D
        assert data.shape == s_lazy.data.shape


class TestDecomposition:
    def test_decomposition_class_assignment(self, diffraction_pattern):
        diffraction_pattern = XrayDiffraction2D(diffraction_pattern)
        diffraction_pattern.decomposition()
        assert isinstance(diffraction_pattern, XrayDiffraction2D)
