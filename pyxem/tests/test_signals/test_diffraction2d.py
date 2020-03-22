# -*- coding: utf-8 -*-
# Copyright 2017-2020 The pyXem developers
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

from pyxem.signals.diffraction2d import Diffraction2D, LazyDiffraction2D
from pyxem.signals.polar_diffraction2d import PolarDiffraction2D
from pyxem.detectors.generic_flat_detector import GenericFlatDetector
from pyxem.signals.diffraction1d import Diffraction1D


class TestComputeAndAsLazy2D:

    def test_2d_data_compute(self):
        dask_array = da.random.random((100, 150), chunks=(50, 50))
        s = LazyDiffraction2D(dask_array)
        scale0, scale1, metadata_string = 0.5, 1.5, 'test'
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s.compute()
        assert isinstance(s, Diffraction2D)
        assert not hasattr(s.data, 'compute')
        assert s.axes_manager[0].scale == scale0
        assert s.axes_manager[1].scale == scale1
        assert s.metadata.Test == metadata_string
        assert dask_array.shape == s.data.shape

    def test_4d_data_compute(self):
        dask_array = da.random.random((4, 4, 10, 15),
                                      chunks=(1, 1, 10, 15))
        s = LazyDiffraction2D(dask_array)
        s.compute()
        assert isinstance(s, Diffraction2D)
        assert dask_array.shape == s.data.shape

    def test_2d_data_as_lazy(self):
        data = np.random.random((100, 150))
        s = Diffraction2D(data)
        scale0, scale1, metadata_string = 0.5, 1.5, 'test'
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s_lazy = s.as_lazy()
        assert isinstance(s_lazy, LazyDiffraction2D)
        assert hasattr(s_lazy.data, 'compute')
        assert s_lazy.axes_manager[0].scale == scale0
        assert s_lazy.axes_manager[1].scale == scale1
        assert s_lazy.metadata.Test == metadata_string
        assert data.shape == s_lazy.data.shape

    def test_4d_data_as_lazy(self):
        data = np.random.random((4, 10, 15))
        s = Diffraction2D(data)
        s_lazy = s.as_lazy()
        assert isinstance(s_lazy, LazyDiffraction2D)
        assert data.shape == s_lazy.data.shape


class TestDecomposition:

    def test_decomposition_is_performed(self, diffraction_pattern):
        s = Diffraction2D(diffraction_pattern)
        s.decomposition()
        assert s.learning_results is not None

    def test_decomposition_class_assignment(self, diffraction_pattern):
        s = Diffraction2D(diffraction_pattern)
        s.decomposition()
        assert isinstance(s, Diffraction2D)


class TestAzimuthalIntegral:

    def test_azimuthal_integral_signal_type(self,
                                            dp_for_azimuthal):
        origin = [3.5, 3.5]
        detector = GenericFlatDetector(8, 8)
        dp_for_azimuthal.metadata.General.title = 'A Title'
        dp_for_azimuthal.axes_manager[0].name = 'x'
        ap = dp_for_azimuthal.get_azimuthal_integral(origin,
                                                     detector=detector,
                                                     detector_distance=1,
                                                     wavelength=1, size_1d=5)

        assert isinstance(ap, Diffraction1D)
        assert dp_for_azimuthal.metadata.General.title == \
            ap.metadata.General.title
        assert dp_for_azimuthal.axes_manager[0].name == \
            ap.axes_manager[0].name

    @pytest.fixture
    def test_dp4D(self):
        dp = Diffraction2D(np.ones((5, 5, 5, 5)))
        return dp

    def test_azimuthal_integral_4D(self, test_dp4D):
        origin = [2, 2]
        detector = GenericFlatDetector(5, 5)
        ap = test_dp4D.get_azimuthal_integral(origin,
                                              detector=detector,
                                              detector_distance=1e6,
                                              wavelength=1, size_1d=4)
        assert isinstance(ap, Diffraction1D)
        assert np.array_equal(ap.data, np.ones((5, 5, 4)))

    @pytest.fixture
    def axes_test_dp(self):
        """Empty diffraction pattern for axes test.
        """
        dp = Diffraction2D(np.zeros((2, 2, 3, 3)))
        return dp

    def test_azimuthal_integral_axes(self, axes_test_dp):
        n_scale = 0.5
        axes_test_dp.axes_manager.navigation_axes[0].scale = n_scale
        axes_test_dp.axes_manager.navigation_axes[1].scale = 2 * n_scale
        name = 'real_space'
        axes_test_dp.axes_manager.navigation_axes[0].name = name
        axes_test_dp.axes_manager.navigation_axes[1].units = name
        units = 'um'
        axes_test_dp.axes_manager.navigation_axes[1].name = units
        axes_test_dp.axes_manager.navigation_axes[0].units = units

        origin = [1, 1]
        detector = GenericFlatDetector(3, 3)
        ap = axes_test_dp.get_azimuthal_integral(origin,
                                                 detector=detector,
                                                 detector_distance=1,
                                                 wavelength=1, size_1d=5)
        rp_scale_x = ap.axes_manager.navigation_axes[0].scale
        rp_scale_y = ap.axes_manager.navigation_axes[1].scale
        rp_units_x = ap.axes_manager.navigation_axes[0].units
        rp_name_x = ap.axes_manager.navigation_axes[0].name
        rp_units_y = ap.axes_manager.navigation_axes[1].units
        rp_name_y = ap.axes_manager.navigation_axes[1].name

        assert n_scale == rp_scale_x
        assert 2 * n_scale == rp_scale_y
        assert units == rp_units_x
        assert name == rp_name_x
        assert name == rp_units_y
        assert units == rp_name_y

    @pytest.mark.parametrize('expected', [
        (np.array(
            [[4.5, 3.73302794, 2.76374221, 1.87174165, 0.83391893, 0.],
             [0.75, 0.46369326, 0.24536559, 0.15187129, 0.06550021, 0.]]
        ))])
    def test_azimuthal_integral_fast(self, dp_for_azimuthal,
                                     expected):
        origin = [3.5, 3.5]
        detector = GenericFlatDetector(8, 8)
        ap = dp_for_azimuthal.get_azimuthal_integral(origin,
                                                     detector=detector,
                                                     detector_distance=1e9,
                                                     wavelength=1, size_1d=6)
        assert np.allclose(ap.data, expected, atol=1e-3)

    def test_azimuthal_integral_slow(self,
                                     dp_for_origin_variation):
        origin = np.array([[[0, 0], [1, 1]], [[1.5, 1.5], [2, 3]]])
        detector = GenericFlatDetector(4, 4)
        ap = dp_for_origin_variation.get_azimuthal_integral(
            origin,
            detector=detector,
            detector_distance=1e9,
            wavelength=1, size_1d=4)
        expected = np.array([[[1.01127149e-07, 4.08790171e-01, 2.93595970e-01, 0.00000000e+00],
                              [2.80096084e-01, 4.43606853e-01, 1.14749573e-01, 0.00000000e+00]],
                             [[6.20952725e-01, 2.99225271e-01, 4.63002026e-02, 0.00000000e+00],
                              [5.00000000e-01, 3.43071640e-01, 1.27089232e-01, 0.00000000e+00]]])
        assert np.allclose(ap.data, expected, atol=1e-5)


class TestPolarReprojection:

    def test_reproject_polar_signal_type(self,
                                         dp_for_azimuthal):
        polar = dp_for_azimuthal.as_polar()
        assert isinstance(polar, PolarDiffraction2D)

    def test_reproject_polar_axes(self,
                                  dp_for_azimuthal):
        polar = dp_for_azimuthal.as_polar()

    def test_reproject_polar_values(self,
                                    dp_for_azimuthal):
        polar = dp_for_azimuthal.as_polar()
