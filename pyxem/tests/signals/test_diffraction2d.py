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
    def test_1d_azimuthal_integral_2th_units(self, ones, unit):
        ones.unit = unit
        ones.set_ai(wavelength=1e-9)
        az = ones.get_azimuthal_integral1d(
            npt=10, wavelength=1e-9, correctSolidAngle=False
        )
        np.testing.assert_array_equal(az.data[0:8], np.ones(8))

    def test_1d_azimuthal_integral_inplace(self, ones):
        ones.set_ai()
        az = ones.get_azimuthal_integral1d(
            npt=10, correctSolidAngle=False, inplace=True,
        )
        assert isinstance(ones, Diffraction1D)
        np.testing.assert_array_equal(ones.data[0:8], np.ones((8)))
        assert az is None

    def test_1d_azimuthal_integral_slicing(self, ones):
        ones.unit = "2th_rad"
        ones.set_ai(center=(5.5, 5.5))
        az = ones.get_azimuthal_integral1d(
            npt=10, method="BBox", correctSolidAngle=False, radial_range=[0.0, 1.0],
        )
        np.testing.assert_array_equal(az.data[0:7], np.ones(7))

    @pytest.mark.parametrize(
        "unit", ["q_nm^-1", "q_A^-1", "k_nm^-1", "k_A^-1", "2th_deg", "2th_rad"]
    )
    def test_1d_axes_continuity(self, ones, unit):
        ones.unit = unit
        ones.set_ai(center=(5.5, 5.5), wavelength=1e-9)
        az1 = ones.get_azimuthal_integral1d(
            npt=10, radial_range=[0.0, 1.0], method="splitpixel",
        )
        assert np.allclose(az1.axes_manager.signal_axes[0].scale, 0.1)

    @pytest.mark.parametrize("radial_range", [None, [0.0, 1.0]])
    @pytest.mark.parametrize("azimuth_range", [None, [-np.pi, np.pi]])
    @pytest.mark.parametrize("center", [None, [9, 9]])
    @pytest.mark.parametrize("affine", [None, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    def test_1d_integration(
        self, ones, radial_range, azimuth_range, center, affine,
    ):
        ones.set_ai(center=center, affine=affine, radial_range=radial_range)
        az = ones.get_azimuthal_integral1d(
            npt=10,
            method="BBox",
            radial_range=radial_range,
            azimuth_range=azimuth_range,
            correctSolidAngle=False,
            inplace=False,
        )
        assert isinstance(az, Diffraction1D)

    def test_1d_azimuthal_integral_mask(self, ones):
        from hyperspy.signals import BaseSignal

        aff = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
        center = [1, 1]
        mask = np.zeros((10, 10))
        mask_bs = BaseSignal(data=mask)
        ones.set_ai(center=center, affine=aff)
        ones.get_azimuthal_integral1d(
            npt=10,
            method="BBox",
            wavelength=1e-9,
            correctSolidAngle=False,
            mask=mask_bs,
        )

    def test_1d_azimuthal_integral_sum(self, ones):
        ones.set_ai()
        integration = ones.get_azimuthal_integral1d(
            npt=5, radial_range=[0, 0.5], sum=True
        )
        # 5^2*pi = 78.5
        np.testing.assert_almost_equal(integration.data.sum(), 78.5, decimal=0)


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

    def test_2d_azimuthal_integral(self, ones):
        ones.set_ai()
        az = ones.get_azimuthal_integral2d(
            npt=10, npt_azim=10, method="BBox", correctSolidAngle=False
        )
        np.testing.assert_array_equal(az.data[0:8, :], np.ones((8, 10)))

    def test_2d_azimuthal_integral_fast_slicing(self, ones):
        ones.set_ai(center=(5.5, 5.5))
        az1 = ones.get_azimuthal_integral2d(
            npt=10,
            npt_azim=10,
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
        ones.set_ai(wavelength=1e-9, center=(5.5, 5.5))
        az1 = ones.get_azimuthal_integral2d(
            npt=10, npt_azim=20, radial_range=[0.0, 1.0], method="splitpixel",
        )
        assert np.allclose(az1.axes_manager.signal_axes[1].scale, 0.1)

    def test_2d_azimuthal_integral_inplace(self, ones):
        ones.set_ai()
        az = ones.get_azimuthal_integral2d(
            npt=10, npt_azim=10, correctSolidAngle=False, inplace=True, method="BBox",
        )
        assert isinstance(ones, PolarDiffraction2D)
        np.testing.assert_array_equal(ones.data[0:8, :], np.ones((8, 10)))
        assert az is None

    @pytest.mark.parametrize("radial_range", [None, [0, 1.0]])
    @pytest.mark.parametrize("azimuth_range", [None, [-np.pi, 0]])
    @pytest.mark.parametrize("correctSolidAngle", [True, False])
    @pytest.mark.parametrize("center", [None, [7, 7]])
    @pytest.mark.parametrize("affine", [None, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    def test_2d_azimuthal_integral_params(
        self, ones, radial_range, azimuth_range, correctSolidAngle, center, affine
    ):
        ones.set_ai(
            center=center, affine=affine, radial_range=radial_range, wavelength=1e-9
        )
        az = ones.get_azimuthal_integral2d(
            npt=10,
            npt_azim=10,
            radial_range=radial_range,
            azimuth_range=azimuth_range,
            correctSolidAngle=correctSolidAngle,
            method="BBox",
        )
        assert isinstance(az, PolarDiffraction2D)

    def test_2d_azimuthal_integral_mask_iterate(self, ones):
        from hyperspy.signals import BaseSignal

        aff = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
        center = [1, 1]
        ones.set_ai(center=center, affine=aff, wavelength=1e-9)
        mask = np.zeros((10, 10))
        mask_bs = BaseSignal(data=mask)
        ones.get_azimuthal_integral2d(
            npt=10, npt_azim=10, method="BBox", correctSolidAngle=False, mask=mask_bs,
        )

    def test_2d_azimuthal_integral_sum(self, ones):
        ones.set_ai()
        integration = ones.get_azimuthal_integral2d(
            npt=10, npt_azim=15, radial_range=[0, 0.5], sum=True
        )
        #  mostly correct except for at the very center where things get weird...
        mask = np.ones((10, 10), dtype=bool)
        mask[0:5] = False
        integration2 = ones.get_azimuthal_integral2d(
            npt=10, npt_azim=15, radial_range=[0, 0.5], sum=True, mask=mask
        )
        print(integration2.sum((-1, -2)).data)


class TestPyFAIIntegration:
    @pytest.fixture
    def ones(self):
        ones_diff = Diffraction2D(data=np.ones(shape=(10, 10)))
        ones_diff.axes_manager.signal_axes[0].scale = 0.1
        ones_diff.axes_manager.signal_axes[1].scale = 0.1
        ones_diff.axes_manager.signal_axes[0].name = "kx"
        ones_diff.axes_manager.signal_axes[1].name = "ky"
        ones_diff.unit = "q_nm^-1"
        return ones_diff

    def test_integrate_radial(self, ones):
        ones.set_ai(center=(5.5, 5.5), wavelength=1e-9)
        integration = ones.get_radial_integral(
            npt=10, npt_rad=100, method="BBox", correctSolidAngle=False,
        )
        np.testing.assert_array_equal(integration, np.ones(10))
        integration = ones.get_radial_integral(
            npt=10, npt_rad=100, method="BBox", correctSolidAngle=False, sum=True,
        )
        integration = ones.get_radial_integral(
            npt=10, npt_rad=100, method="BBox", correctSolidAngle=False, inplace=True
        )
        np.testing.assert_array_equal(ones, np.ones(10))
        assert integration is None

    def test_integrate_med_filter(self, ones):
        ones.set_ai(center=(5.5, 5.5), wavelength=1e-9)
        integration = ones.get_medfilt1d(
            npt_rad=10, npt_azim=100, method="BBox", correctSolidAngle=False
        )
        np.testing.assert_array_equal(integration, np.ones(10))
        integration = ones.get_medfilt1d(
            npt_rad=10,
            npt_azim=100,
            method="BBox",
            correctSolidAngle=False,
            inplace=True,
        )
        np.testing.assert_array_equal(ones, np.ones(10))
        assert integration is None

    def test_integrate_sigma_clip(self, ones):
        ones.set_ai(center=(5.5, 5.5), wavelength=1e-9)
        integration = ones.sigma_clip(
            npt_rad=10, npt_azim=100, method="BBox", correctSolidAngle=False
        )
        np.testing.assert_array_equal(integration, np.ones(10))
        integration = ones.sigma_clip(
            npt_rad=10,
            npt_azim=100,
            method="BBox",
            correctSolidAngle=False,
            inplace=True,
        )
        np.testing.assert_array_equal(ones, np.ones(10))
        assert integration is None


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


class TestAzimuthalIntegrator:
    # Tests the setting of a Azimutal Integrator:
    @pytest.fixture
    def ones(self):
        ones_diff = Diffraction2D(data=np.ones(shape=(10, 10)))
        ones_diff.axes_manager.signal_axes[0].scale = 0.1
        ones_diff.axes_manager.signal_axes[1].scale = 0.1
        ones_diff.axes_manager.signal_axes[0].name = "kx"
        ones_diff.axes_manager.signal_axes[1].name = "ky"
        ones_diff.unit = "2th_deg"
        return ones_diff

    def test_set_ai_fail(self, ones):
        ones.unit = "k_nm^-1"
        ai = ones.set_ai()
        assert ai is None

    def test_return_ai_fail(self, ones):
        ai = ones.ai
        assert ai is None


class TestGetDirectBeamPosition:
    def setup_method(self):
        px, py, dx, dy = 15, 10, 20, 26
        x_pos_list = np.linspace(dx / 2 - 5, dx / 2 + 5, px, dtype=np.int16)
        y_pos_list = np.linspace(dy / 2 - 7, dy / 2 + 7, py, dtype=np.int16)
        data = np.zeros((py, px, dy, dx), dtype=np.uint16)
        for ix in range(data.shape[1]):
            for iy in range(data.shape[0]):
                x_pos, y_pos = x_pos_list[ix], y_pos_list[iy]
                data[iy, ix, y_pos, x_pos] = 10
        s = Diffraction2D(data)
        self.s = s
        self.dx, self.dy = dx, dy
        self.x_pos_list, self.y_pos_list = x_pos_list, y_pos_list

    def test_blur(self):
        dx, dy = self.dx, self.dy
        s, x_pos_list, y_pos_list = self.s, self.x_pos_list, self.y_pos_list
        s_shift = s.get_direct_beam_position(method="blur", sigma=1)
        assert s.axes_manager.navigation_shape == s_shift.axes_manager.navigation_shape
        assert (-(x_pos_list - dx / 2) == s_shift.isig[0].data[0]).all()
        assert (-(y_pos_list - dy / 2) == s_shift.isig[1].data[:, 0]).all()

    def test_interpolate(self):
        dx, dy = self.dx, self.dy
        s, x_pos_list, y_pos_list = self.s, self.x_pos_list, self.y_pos_list
        s_shift = s.get_direct_beam_position(
            method="interpolate",
            sigma=1,
            upsample_factor=2,
            kind="nearest",
        )
        assert s.axes_manager.navigation_shape == s_shift.axes_manager.navigation_shape
        assert (-(x_pos_list - dx / 2) == s_shift.isig[0].data[0]).all()
        assert (-(y_pos_list - dy / 2) == s_shift.isig[1].data[:, 0]).all()

    def test_cross_correlate(self):
        s = self.s
        s_shift = s.get_direct_beam_position(
            method="cross_correlate", radius_start=0, radius_finish=1
        )

    def test_lazy_result_none_non_lazy_signal(self):
        s = self.s
        s_shift = s.get_direct_beam_position(method="blur", sigma=1)
        assert not s_shift._lazy

    def test_lazy_result_none_lazy_signal(self):
        s = self.s.as_lazy()
        s_shift = s.get_direct_beam_position(method="blur", sigma=1)
        assert s_shift._lazy

    def test_lazy_result(self):
        s = self.s
        s_shift = s.get_direct_beam_position(method="blur", sigma=1, lazy_result=True)
        assert hasattr(s_shift.data, "compute")
        s_shift.compute()

    def test_lazy_input_non_lazy_result(self):
        s = LazyDiffraction2D(da.from_array(self.s.data))
        s_shift = s.get_direct_beam_position(method="blur", sigma=1, lazy_result=False)
        assert not hasattr(s_shift.data, "compute")
        assert not hasattr(s_shift, "compute")

    def test_lazy_input_lazy_result(self):
        s = LazyDiffraction2D(da.from_array(self.s.data))
        s_shift = s.get_direct_beam_position(method="blur", sigma=1, lazy_result=True)
        assert hasattr(s_shift.data, "compute")
        s_shift.compute()


class TestCenterDirectBeam:
    def setup_method(self):
        data = np.zeros((8, 6, 20, 16), dtype=np.int16)
        x_pos_list = np.random.randint(8 - 2, 8 + 2, 6, dtype=np.int16)
        x_pos_list[x_pos_list == 8] = 9
        y_pos_list = np.random.randint(10 - 2, 10 + 2, 8, dtype=np.int16)
        for ix in range(len(x_pos_list)):
            for iy in range(len(y_pos_list)):
                data[iy, ix, y_pos_list[iy], x_pos_list[ix]] = 9
        s = Diffraction2D(data)
        s.axes_manager[0].scale = 0.5
        s.axes_manager[1].scale = 0.6
        s.axes_manager[2].scale = 3
        s.axes_manager[3].scale = 4
        s_lazy = s.as_lazy()
        self.s = s
        self.s_lazy = s_lazy
        self.x_pos_list = x_pos_list
        self.y_pos_list = y_pos_list

    def test_non_lazy(self):
        s = self.s
        s.center_direct_beam(method="blur", sigma=1)
        assert s._lazy is False
        assert (s.data[:, :, 10, 8] == 9).all()
        # Make sure only the pixel we expect to change, has actually changed
        s.data[:, :, 10, 8] = 0
        assert not s.data.any()

    def test_non_lazy_lazy_result(self):
        s = self.s
        s.center_direct_beam(method="blur", sigma=1, lazy_result=True)
        assert s._lazy is True
        s.compute()
        assert (s.data[:, :, 10, 8] == 9).all()
        s.data[:, :, 10, 8] = 0
        assert not s.data.any()

    def test_lazy(self):
        s_lazy = self.s_lazy
        s_lazy.center_direct_beam(method="blur", sigma=1)
        assert s_lazy._lazy is True
        s_lazy.compute()
        assert (s_lazy.data[:, :, 10, 8] == 9).all()
        s_lazy.data[:, :, 10, 8] = 0
        assert not s_lazy.data.any()

    def test_lazy_not_lazy_result(self):
        s_lazy = self.s_lazy
        s_lazy.center_direct_beam(method="blur", sigma=1, lazy_result=False)
        assert s_lazy._lazy is False
        assert (s_lazy.data[:, :, 10, 8] == 9).all()
        s_lazy.data[:, :, 10, 8] = 0
        assert not s_lazy.data.any()

    def test_return_shifts_non_lazy(self):
        s = self.s
        s_shifts = s.center_direct_beam(method="blur", sigma=1, return_shifts=True)
        assert s_shifts._lazy is False
        nav_dim = s.axes_manager.navigation_dimension
        assert nav_dim == s_shifts.axes_manager.navigation_dimension
        x_pos_list, y_pos_list = self.x_pos_list, self.y_pos_list
        assert ((8 - x_pos_list) == s_shifts.isig[0].data[0]).all()
        assert ((10 - y_pos_list) == s_shifts.isig[1].data[:, 0]).all()

    def test_return_shifts_lazy(self):
        s_lazy = self.s_lazy
        s_shifts = s_lazy.center_direct_beam(method="blur", sigma=1, return_shifts=True)
        assert s_shifts._lazy is True
        s_shifts.compute()
        x_pos_list, y_pos_list = self.x_pos_list, self.y_pos_list
        assert ((8 - x_pos_list) == s_shifts.isig[0].data[0]).all()
        assert ((10 - y_pos_list) == s_shifts.isig[1].data[:, 0]).all()

    def test_return_axes_manager(self):
        s = self.s
        s.center_direct_beam(method="blur", sigma=1)
        assert s.axes_manager[0].scale == 0.5
        assert s.axes_manager[1].scale == 0.6
        assert s.axes_manager[2].scale == 3
        assert s.axes_manager[3].scale == 4

    def test_shifts_input(self):
        s = self.s
        s_shifts = s.get_direct_beam_position(method="blur", sigma=1, lazy_result=False)
        s.center_direct_beam(shifts=s_shifts)
        assert (s.data[:, :, 10, 8] == 9).all()
        s.data[:, :, 10, 8] = 0
        assert not s.data.any()

    def test_shifts_input_lazy(self):
        s = self.s
        s_shifts = s.get_direct_beam_position(method="blur", sigma=1, lazy_result=True)
        s.center_direct_beam(shifts=s_shifts)
        assert (s.data[:, :, 10, 8] == 9).all()
        s.data[:, :, 10, 8] = 0
        assert not s.data.any()

    def test_subpixel(self):
        s = self.s
        s_shifts = s.get_direct_beam_position(method="blur", sigma=1)
        s_shifts += 0.5
        s.change_dtype("float32")
        s.center_direct_beam(shifts=s_shifts, subpixel=True)
        assert (s.data[:, :, 10:12, 8:10] == 9 / 4).all()
        s.data[:, :, 10:12, 8:10] = 0.0
        assert not s.data.any()

    def test_not_subpixel(self):
        s = self.s
        s_shifts = s.get_direct_beam_position(method="blur", sigma=1)
        s_shifts += 0.3
        s.change_dtype("float32")
        s.center_direct_beam(shifts=s_shifts, subpixel=False)
        assert (s.data[:, :, 10, 8] == 9).all()
        s.data[:, :, 10, 8] = 0.0
        assert not s.data.any()

    @pytest.mark.parametrize(
        "shape", [(20, 20), (10, 20, 20), (8, 10, 20, 20), (6, 8, 10, 20, 20)]
    )
    def test_different_dimensions(self, shape):
        s = Diffraction2D(np.random.randint(0, 256, size=shape))
        s.center_direct_beam(method="blur", sigma=1)
        assert s.data.shape == shape

    def test_half_square_width(self):
        s = self.s.isig[:, 2:-2]
        s.data[:, :, 1, -1] = 1000
        s1 = s.deepcopy()
        s.center_direct_beam(method="blur", sigma=1)
        assert (s.data[:, :, 8, 8] == 1000).all()
        s1.center_direct_beam(method="blur", sigma=1, half_square_width=5)
        assert (s1.data[:, :, 8, 8] == 9).all()

    def test_align_kwargs(self):
        s = self.s
        s.data += 1
        s1 = s.deepcopy()
        s.center_direct_beam(method="blur", sigma=1)
        assert (s.data == 0).any()
        s1.center_direct_beam(method="blur", sigma=1, align_kwargs={"mode": "wrap"})
        assert not (s1.data == 0).any()

    def test_method_interpolate(self):
        s = self.s
        s.center_direct_beam(method="interpolate", sigma=1, upsample_factor=10, kind=1)

    def test_method_cross_correlate(self):
        s = self.s
        s.center_direct_beam(method="cross_correlate", radius_start=0, radius_finish=2)

    def test_parameter_both_method_and_shifts(self):
        s = self.s
        with pytest.raises(ValueError):
            s.center_direct_beam(method="blur", sigma=1, shifts=np.ones((8, 6, 2)))

    def test_parameter_neither_method_and_shifts(self):
        s = self.s
        with pytest.raises(ValueError):
            s.center_direct_beam()


class TestMakeProbeNavigation:
    def test_fast(self):
        s = Diffraction2D(np.ones((6, 5, 12, 10)))
        s.make_probe_navigation(method="fast")
        s_nav = s._navigator_probe
        assert s.axes_manager.navigation_shape == s_nav.axes_manager.signal_shape
        assert np.all(s_nav.data == 1)

    def test_slow(self):
        s = Diffraction2D(np.ones((5, 5, 12, 10)))
        s.make_probe_navigation(method="slow")
        s_nav = s._navigator_probe
        assert s.axes_manager.navigation_shape == s_nav.axes_manager.signal_shape
        assert np.all(s_nav.data == 120)

    def test_fast_lazy(self):
        s = LazyDiffraction2D(da.ones((6, 4, 60, 50), chunks=(2, 2, 10, 10)))
        s.make_probe_navigation(method="fast")
        s_nav = s._navigator_probe
        assert s.axes_manager.navigation_shape == s_nav.axes_manager.signal_shape
        assert np.all(s_nav.data == 100)

    def test_slow_lazy(self):
        s = LazyDiffraction2D(da.ones((6, 4, 60, 50), chunks=(2, 2, 10, 10)))
        s.make_probe_navigation(method="slow")
        s_nav = s._navigator_probe
        assert s.axes_manager.navigation_shape == s_nav.axes_manager.signal_shape
        assert np.all(s_nav.data == 3000)

    def test_very_asymmetric_size(self):
        s = Diffraction2D(np.ones((50, 2, 120, 10)))
        s.make_probe_navigation(method="fast")
        s_nav = s._navigator_probe
        assert s.axes_manager.navigation_shape == s_nav.axes_manager.signal_shape

    def test_very_asymmetric_size_lazy(self):
        s = LazyDiffraction2D(da.ones((50, 2, 120, 10), chunks=(2, 2, 5, 5)))
        s.make_probe_navigation(method="fast")
        s_nav = s._navigator_probe
        assert s.axes_manager.navigation_shape == s_nav.axes_manager.signal_shape

    @pytest.mark.parametrize("shape", [(2, 10, 10), (3, 2, 10, 10)])
    def test_different_shapes(self, shape):
        s = Diffraction2D(np.ones(shape))
        s.make_probe_navigation(method="fast")
        s_nav = s._navigator_probe
        assert s.axes_manager.navigation_shape == s_nav.axes_manager.signal_shape

    @pytest.mark.parametrize("shape", [(10, 20), (2, 3, 4, 5, 6), (2, 3, 4, 5, 6, 7)])
    def test_wrong_navigation_dimensions(self, shape):
        s = Diffraction2D(np.ones(shape))
        with pytest.raises(ValueError):
            s.make_probe_navigation(method="fast")


class TestPlotNavigator:
    @pytest.mark.parametrize(
        "shape", [(9, 8), (5, 9, 8), (4, 5, 9, 8), (8, 4, 5, 9, 8), (9, 8, 4, 5, 9, 8)]
    )
    def test_non_lazy(self, shape):
        s = Diffraction2D(np.random.randint(0, 256, shape), dtype=np.uint8)
        plt.ion()  # To make plotting non-blocking
        s.plot()
        s.plot()
        plt.close("all")

    @pytest.mark.parametrize(
        "shape", [(9, 8), (5, 9, 8), (4, 5, 9, 8), (8, 4, 5, 9, 8), (9, 8, 4, 5, 9, 8)]
    )
    def test_lazy(self, shape):
        s = LazyDiffraction2D(da.random.randint(0, 256, shape), dtype=np.uint8)
        plt.ion()  # To make plotting non-blocking
        s.plot()
        s.plot()
        plt.close("all")

    def test_navigator_kwarg(self):
        s = Diffraction2D(np.random.randint(0, 256, (8, 9, 10, 30), dtype=np.uint8))
        plt.ion()  # To make plotting non-blocking
        s_nav = Diffraction2D(np.zeros((8, 9)))
        s.plot(navigator=s_nav)
        plt.close("all")

    def test_wrong_navigator_shape_kwarg(self):
        s = Diffraction2D(np.random.randint(0, 256, (8, 9, 10, 30), dtype=np.uint8))
        plt.ion()  # To make plotting non-blocking
        s_nav = Diffraction2D(np.zeros((2, 19)))
        s._navigator_probe = s_nav
        with pytest.raises(ValueError):
            s.plot()
