# -*- coding: utf-8 -*-
# Copyright 2016-2020 The pyXem developers
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
import hyperspy.api as hs
from matplotlib import pyplot as plt

from pyxem.signals.diffraction2d import Diffraction2D, LazyDiffraction2D
from pyxem.signals.polar_diffraction2d import PolarDiffraction2D
from pyxem.detectors.generic_flat_detector import GenericFlatDetector
from pyxem.signals.diffraction1d import Diffraction1D


class TestComputeAndAsLazy2D:
    def test_2d_data_compute(self):
        dask_array = da.random.random((100, 150), chunks=(50, 50))
        s = LazyDiffraction2D(dask_array)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s.compute()
        assert isinstance(s, Diffraction2D)
        assert not hasattr(s.data, "compute")
        assert s.axes_manager[0].scale == scale0
        assert s.axes_manager[1].scale == scale1
        assert s.metadata.Test == metadata_string
        assert dask_array.shape == s.data.shape

    def test_4d_data_compute(self):
        dask_array = da.random.random((4, 4, 10, 15), chunks=(1, 1, 10, 15))
        s = LazyDiffraction2D(dask_array)
        s.compute()
        assert isinstance(s, Diffraction2D)
        assert dask_array.shape == s.data.shape

    def test_2d_data_as_lazy(self):
        data = np.random.random((100, 150))
        s = Diffraction2D(data)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s_lazy = s.as_lazy()
        assert isinstance(s_lazy, LazyDiffraction2D)
        assert hasattr(s_lazy.data, "compute")
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


class TestAzimuthalIntegral1d:
    @pytest.fixture
    def ones(self):
        ones_diff = Diffraction2D(data=np.ones(shape=(10, 10)))
        ones_diff.axes_manager.signal_axes[0].scale = 0.1
        ones_diff.axes_manager.signal_axes[1].scale = 0.1
        ones_diff.axes_manager.signal_axes[0].name = "kx"
        ones_diff.axes_manager.signal_axes[1].name = "ky"
        ones_diff.unit = "2th_deg"
        return ones_diff

    def test_unit(self, ones):
        dif = Diffraction2D(data=[[1, 1], [1, 1]])
        assert dif.unit is None
        dif.unit = "!23"
        assert dif.unit is None

    def test_unit_set(self, ones):
        assert ones.unit == "2th_deg"

    @pytest.mark.parametrize(
        "unit", ["q_nm^-1", "q_A^-1", "k_nm^-1", "k_A^-1", "2th_deg", "2th_rad"]
    )
    def test_1d_azimuthal_integral_fast_2th_units(self, ones, unit):
        ones.unit = unit
        az = ones.get_azimuthal_integral1d(
            npt_rad=10, wavelength=1e-9, correctSolidAngle=False
        )
        np.testing.assert_array_equal(az.data[0:8], np.ones(8))

    def test_1d_azimuthal_integral_inplace(self, ones):
        az = ones.get_azimuthal_integral1d(
            npt_rad=10, correctSolidAngle=False, inplace=True,
        )
        assert isinstance(ones, Diffraction1D)
        np.testing.assert_array_equal(ones.data[0:8], np.ones((8)))
        assert az is None

    def test_1d_azimuthal_integral_fast_slicing(self, ones):
        ones.unit = "2th_rad"
        az = ones.get_azimuthal_integral1d(
            npt_rad=10,
            center=(5.5, 5.5),
            method="BBox",
            correctSolidAngle=False,
            radial_range=[0.0, 1.0],
        )
        np.testing.assert_array_equal(az.data[0:7], np.ones(7))

    @pytest.mark.parametrize(
        "unit", ["q_nm^-1", "q_A^-1", "k_nm^-1", "k_A^-1", "2th_deg", "2th_rad"]
    )
    def test_1d_axes_continuity(self, ones, unit):
        ones.unit = unit
        az1 = ones.get_azimuthal_integral1d(
            wavelength=1e-9,
            npt_rad=10,
            center=(5.5, 5.5),
            radial_range=[0.0, 1.0],
            method="splitpixel",
        )
        assert np.allclose(az1.axes_manager.signal_axes[0].scale, 0.1)

    @pytest.mark.parametrize("radial_range", [None, [0.0, 1.0]])
    @pytest.mark.parametrize("azimuth_range", [None, [-np.pi, np.pi]])
    @pytest.mark.parametrize("center", [None, [9, 9]])
    @pytest.mark.parametrize("affine", [None, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    def test_1d_integration(
        self, ones, radial_range, azimuth_range, center, affine,
    ):
        az = ones.get_azimuthal_integral1d(
            npt_rad=10,
            method="BBox",
            wavelength=1e-9,
            radial_range=radial_range,
            azimuth_range=azimuth_range,
            center=center,
            affine=affine,
            correctSolidAngle=False,
            inplace=False,
        )
        assert isinstance(az, Diffraction1D)

    def test_1d_azimuthal_integral_slow(self, ones):
        from hyperspy.signals import BaseSignal

        aff = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
        aff_bs = BaseSignal(data=aff)
        ones.get_azimuthal_integral1d(
            npt_rad=10,
            method="BBox",
            wavelength=1e-9,
            correctSolidAngle=False,
            affine=aff_bs,
        )

    def test_1d_azimuthal_integral_slow_shifted_center(self, ones):
        from hyperspy.signals import BaseSignal

        aff = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
        aff_bs = BaseSignal(data=aff)
        center = [1, 1]
        center_bs = BaseSignal(data=center)
        ones.get_azimuthal_integral1d(
            npt_rad=10,
            method="BBox",
            wavelength=1e-9,
            correctSolidAngle=False,
            affine=aff_bs,
            center=center_bs,
        )

    def test_1d_azimuthal_integral_slow_mask(self, ones):
        from hyperspy.signals import BaseSignal

        aff = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
        aff_bs = BaseSignal(data=aff)
        center = [1, 1]
        center_bs = BaseSignal(data=center)
        mask = np.zeros((10, 10))
        mask_bs = BaseSignal(data=mask)
        ones.get_azimuthal_integral1d(
            npt_rad=10,
            method="BBox",
            wavelength=1e-9,
            correctSolidAngle=False,
            affine=aff_bs,
            center=center_bs,
            mask=mask_bs,
        )

    def test_1d_azimuthal_integral_pyfai(self, ones):
        from pyFAI.detectors import Detector

        d = Detector(pixel1=1e-4, pixel2=1e-4)
        ones.get_azimuthal_integral1d(
            npt_rad=10,
            detector=d,
            detector_dist=1,
            method="BBox",
            wavelength=1e-9,
            correctSolidAngle=False,
            unit="q_nm^-1",
        )

    def test_1d_azimuthal_integral_failure(self, ones):
        ones.unit = "k_nm^-1"
        integration = ones.get_azimuthal_integral1d(npt_rad=10)
        assert integration is None


class TestAzimuthalIntegral2d:
    @pytest.fixture
    def ones(self):
        ones_diff = Diffraction2D(data=np.ones(shape=(10, 10)))
        ones_diff.axes_manager.signal_axes[0].scale = 0.1
        ones_diff.axes_manager.signal_axes[1].scale = 0.1
        ones_diff.axes_manager.signal_axes[0].name = "kx"
        ones_diff.axes_manager.signal_axes[1].name = "ky"
        ones_diff.unit = "2th_deg"
        return ones_diff

    def test_2d_azimuthal_integral_fast(self, ones):
        az = ones.get_azimuthal_integral2d(
            npt_rad=10, npt_azim=10, method="BBox", correctSolidAngle=False
        )
        np.testing.assert_array_equal(az.data[0:8, :], np.ones((8, 10)))

    def test_2d_azimuthal_integral_fast_slicing(self, ones):
        az1 = ones.get_azimuthal_integral2d(
            npt_rad=10,
            npt_azim=10,
            center=(5.5, 5.5),
            radial_range=[0.0, 1.0],
            method="splitpixel",
            correctSolidAngle=False,
        )
        np.testing.assert_array_equal(az1.data[8:, :], np.zeros(shape=(2, 10)))

    @pytest.mark.parametrize(
        "unit", ["q_nm^-1", "q_A^-1", "k_nm^-1", "k_A^-1", "2th_deg", "2th_rad"]
    )
    def test_2d_axes_continuity(self, ones, unit):
        ones.unit = unit
        az1 = ones.get_azimuthal_integral2d(
            wavelength=1e-9,
            npt_rad=10,
            npt_azim=20,
            center=(5.5, 5.5),
            radial_range=[0.0, 1.0],
            method="splitpixel",
        )
        assert np.allclose(az1.axes_manager.signal_axes[1].scale, 0.1)

    def test_2d_azimuthal_integral_inplace(self, ones):
        az = ones.get_azimuthal_integral2d(
            npt_rad=10,
            npt_azim=10,
            correctSolidAngle=False,
            inplace=True,
            method="BBox",
        )
        assert isinstance(ones, PolarDiffraction2D)
        np.testing.assert_array_equal(ones.data[0:8, :], np.ones((8, 10)))
        assert az is None

    @pytest.mark.parametrize("radial_range", [None, [0, 1.0]])
    @pytest.mark.parametrize("azimuth_range", [None, [-np.pi, 0]])
    @pytest.mark.parametrize("correctSolidAngle", [True, False])
    @pytest.mark.parametrize("center", [None, [7, 7]])
    @pytest.mark.parametrize("affine", [None, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    def test_2d_azimuthal_integral_fast_params(
        self, ones, radial_range, azimuth_range, correctSolidAngle, center, affine
    ):
        az = ones.get_azimuthal_integral2d(
            npt_rad=10,
            npt_azim=10,
            wavelength=1e-9,
            radial_range=radial_range,
            azimuth_range=azimuth_range,
            correctSolidAngle=correctSolidAngle,
            center=center,
            affine=affine,
            method="BBox",
        )
        assert isinstance(az, PolarDiffraction2D)

    def test_2d_azimuthal_integral_failure(self, ones):
        ones.unit = "k_nm^-1"
        integration = ones.get_azimuthal_integral2d(npt_rad=10)
        assert integration is None

    def test_2d_azimuthal_integral_slow(self, ones):
        from hyperspy.signals import BaseSignal

        aff = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
        aff_bs = BaseSignal(data=aff)
        ones.get_azimuthal_integral2d(
            npt_rad=10,
            npt_azim=10,
            method="BBox",
            wavelength=1e-9,
            correctSolidAngle=False,
            affine=aff_bs,
        )

    def test_2d_azimuthal_integral_slow_shifted_center(self, ones):
        from hyperspy.signals import BaseSignal

        aff = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
        aff_bs = BaseSignal(data=aff)
        center = [1, 1]
        center_bs = BaseSignal(data=center)
        ones.get_azimuthal_integral2d(
            npt_rad=10,
            npt_azim=10,
            method="BBox",
            wavelength=1e-9,
            correctSolidAngle=False,
            affine=aff_bs,
            center=center_bs,
        )

    def test_2d_azimuthal_integral_slow_mask(self, ones):
        from hyperspy.signals import BaseSignal

        aff = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
        aff_bs = BaseSignal(data=aff)
        center = [1, 1]
        center_bs = BaseSignal(data=center)
        mask = np.zeros((10, 10))
        mask_bs = BaseSignal(data=mask)
        ones.get_azimuthal_integral2d(
            npt_rad=10,
            npt_azim=10,
            method="BBox",
            wavelength=1e-9,
            correctSolidAngle=False,
            affine=aff_bs,
            center=center_bs,
            mask=mask_bs,
        )

    def test_2d_azimuthal_integral_pyfai(self, ones):
        from pyFAI.detectors import Detector

        d = Detector(pixel1=1e-4, pixel2=1e-4)
        ones.get_azimuthal_integral2d(
            npt_rad=10,
            detector=d,
            detector_dist=1,
            method="BBox",
            wavelength=1e-9,
            correctSolidAngle=False,
            unit="q_nm^-1",
        )


class TestVirtualImaging:
    # Tests that virtual imaging runs without failure

    @pytest.mark.parametrize("stack", [False])
    def test_plot_integrated_intensity(self, stack, diffraction_pattern):
        if stack:
            diffraction_pattern = hs.stack([diffraction_pattern] * 3)
        roi = hs.roi.CircleROI(3, 3, 5)
        plt.ion()  # to make plotting non-blocking
        diffraction_pattern.plot_integrated_intensity(roi)
        plt.close("all")

    def test_get_integrated_intensity(self, diffraction_pattern):
        roi = hs.roi.CircleROI(3, 3, 5)
        vi = diffraction_pattern.get_integrated_intensity(roi)
        assert vi.data.shape == (2, 2)
        assert vi.axes_manager.signal_dimension == 2
        assert vi.axes_manager.navigation_dimension == 0

    @pytest.mark.parametrize("out_signal_axes", [None, (0, 1), (1, 2), ("x", "y")])
    def test_get_integrated_intensity_stack(self, diffraction_pattern, out_signal_axes):
        s = hs.stack([diffraction_pattern] * 3)
        s.axes_manager.navigation_axes[0].name = "x"
        s.axes_manager.navigation_axes[1].name = "y"

        roi = hs.roi.CircleROI(3, 3, 5)
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

    def test_get_integrated_intensity_out_signal_axes(self, diffraction_pattern):
        s = hs.stack([diffraction_pattern] * 3)
        roi = hs.roi.CircleROI(3, 3, 5)
        vi = s.get_integrated_intensity(roi, out_signal_axes=(0, 1, 2))
        assert vi.axes_manager.signal_dimension == 3
        assert vi.axes_manager.navigation_dimension == 0
        assert vi.metadata.General.title == "Integrated intensity"
        assert (
            vi.metadata.Diffraction.intergrated_range
            == "CircleROI(cx=3, cy=3, r=5) of Stack of "
        )

    def test_get_integrated_intensity_error(
        self, diffraction_pattern, out_signal_axes=(0, 1, 2)
    ):
        roi = hs.roi.CircleROI(3, 3, 5)
        with pytest.raises(ValueError):
            _ = diffraction_pattern.get_integrated_intensity(roi, out_signal_axes)

    def test_get_integrated_intensity_linescan(self, diffraction_pattern):
        s = diffraction_pattern.inav[0, :]
        s.metadata.General.title = ""
        roi = hs.roi.CircleROI(3, 3, 5)
        vi = s.get_integrated_intensity(roi)
        assert vi.data.shape == (2,)
        assert vi.axes_manager.signal_dimension == 1
        assert vi.axes_manager.navigation_dimension == 0
        assert vi.metadata.Diffraction.intergrated_range == "CircleROI(cx=3, cy=3, r=5)"
