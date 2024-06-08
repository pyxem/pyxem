# -*- coding: utf-8 -*-
# Copyright 2016-2024 The pyXem developers
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
from numpy.random import randint
import dask.array as da
from skimage import morphology

from hyperspy.signals import Signal2D, BaseSignal

import pyxem.signals
from pyxem.signals import Diffraction2D, LazyDiffraction2D
import pyxem.utils.ransac_ellipse_tools as ret
from pyxem.data.dummy_data import make_diffraction_test_data as mdtd
import pyxem.data.dummy_data as dd


@pytest.mark.slow
class TestDiffraction2D:
    def test_create(self):
        array0 = np.zeros(shape=(10, 10, 10, 10))
        s0 = Diffraction2D(array0)
        assert array0.shape == s0.axes_manager.shape

        # This should fail due to Diffraction2D inheriting
        # signal2D, i.e. the data has to be at least
        # 2-dimensions
        with pytest.raises(ValueError):
            Diffraction2D(np.zeros(10))

        array1 = np.zeros(shape=(10, 10))
        s1 = Diffraction2D(array1)
        assert array1.shape == s1.axes_manager.shape


@pytest.mark.slow
class TestPlotting:
    def test_simple_plot(self):
        s = Diffraction2D(np.zeros(shape=(3, 4, 6, 10)))
        s.plot()

    def test_navigation_signal_plot(self):
        s = Diffraction2D(np.zeros(shape=(3, 4, 6, 10)))
        s_nav = Signal2D(np.zeros((3, 4)))
        s.navigation_signal = s_nav
        s.plot()

    def test_navigation_signal_plot_argument(self):
        s = Diffraction2D(np.zeros(shape=(3, 4, 6, 10)))
        s_nav = Signal2D(np.zeros((3, 4)))
        s.plot(navigator=s_nav)


class TestDiffraction2DFlipDiffraction:
    def test_flip_x(self):
        array = np.zeros(shape=(3, 4, 6, 10))
        array[:, :, :, 5:] = 1
        s = Diffraction2D(array)
        assert (s.data[:, :, :, 5:] == 1).all()
        s_flip = s.flip_diffraction_x()
        s.metadata.General.title = "Test"
        assert s_flip.metadata == s.metadata
        assert s_flip.metadata.General.title == "Test"
        assert (s_flip.data[:, :, :, 5:] == 0).all()
        assert (s_flip.data[:, :, :, :5] == 1).all()

    def test_flip_y(self):
        array = np.zeros(shape=(3, 4, 6, 10))
        array[:, :, 3:, :] = 1
        s = Diffraction2D(array)
        assert (s.data[:, :, 3:, :] == 1).all()
        s.metadata.General.title = "Test"
        s_flip = s.flip_diffraction_y()
        assert s_flip.metadata == s.metadata
        assert s_flip.metadata.General.title == "Test"
        assert (s_flip.data[:, :, 3:, :] == 0).all()
        assert (s_flip.data[:, :, :3, :] == 1).all()


class TestAddPeakArrayAsMarkers:
    def test_simple(self):
        s = Diffraction2D(np.zeros((2, 3, 100, 100)))
        peak_array = np.empty((2, 3), dtype=object)
        for index in np.ndindex(peak_array.shape):
            islice = np.s_[index]
            peak_array[islice] = np.random.randint(20, 80, (100, 2))
        s.add_peak_array_as_markers(peak_array)

    def test_color(self):
        s = Diffraction2D(np.zeros((2, 3, 100, 100)))
        peak_array = np.empty((2, 3), dtype=object)
        for index in np.ndindex(peak_array.shape):
            islice = np.s_[index]
            peak_array[islice] = np.random.randint(20, 80, (100, 2))
        s.add_peak_array_as_markers(peak_array, color=("blue",))
        marker0 = list(s.metadata.Markers)[0][1]
        assert marker0.kwargs["color"] == ("blue",)

    def test_peak_array(self):
        s = Diffraction2D(np.zeros((2, 3, 100, 100)))
        peak_array = np.empty((3, 2), dtype=object)
        for index in np.ndindex(peak_array.shape):
            islice = np.s_[index]
            peak_array[islice] = np.random.randint(20, 80, (100, 2))
        pk = BaseSignal(peak_array)
        s.add_peak_array_as_markers(pk)

    def test_peak_array_error(self):
        s = Diffraction2D(np.zeros((2, 3, 100, 100)))
        with pytest.raises(TypeError):
            s.add_peak_array_as_markers([1, 2, 3, 4])

    def test_size(self):
        s = Diffraction2D(np.zeros((2, 3, 100, 100)))
        peak_array = np.empty((2, 3), dtype=object)
        for index in np.ndindex(peak_array.shape):
            islice = np.s_[index]
            peak_array[islice] = np.random.randint(20, 80, (100, 2))
        s.add_peak_array_as_markers(peak_array, sizes=(13,))
        marker = list(s.metadata.Markers)[0][1]
        assert marker.kwargs["sizes"] == (13,)

    def test_3d_nav_dims(self):
        s = Diffraction2D(np.zeros((2, 3, 4, 100, 100)))
        peak_array = np.empty((2, 3, 4), dtype=object)
        for index in np.ndindex(peak_array.shape):
            islice = np.s_[index]
            peak_array[islice] = np.random.randint(20, 80, (100, 2))
        s.add_peak_array_as_markers(peak_array)
        marker = list(s.metadata.Markers)[0][1]
        assert marker.kwargs["offsets"][()].shape == (4, 3, 2)

    def test_1d_nav_dims(self):
        nav_dim = 3
        s = Diffraction2D(np.zeros((nav_dim, 100, 100)))
        peak_array = np.empty(nav_dim, dtype=object)
        for index in np.ndindex(peak_array.shape):
            islice = np.s_[index]
            peak_array[islice] = np.random.randint(20, 80, (100, 2))
        s.add_peak_array_as_markers(peak_array)
        marker = list(s.metadata.Markers)[0][1]
        assert marker.kwargs["offsets"].shape == (3,)

    def test_0d_nav_dims(self):
        s = Diffraction2D(np.zeros((100, 100)))
        peak_array = np.random.randint(20, 80, size=(100, 2))
        s.add_peak_array_as_markers(peak_array)
        marker = list(s.metadata.Markers)[0][1]
        assert marker.kwargs["offsets"].dtype != object


class TestAddEllipseArrayAsMarkers:
    def test_simple(self):
        s, parray = dd.get_simple_ellipse_signal_peak_array(seed=15)
        ellipse_array, inlier_array = ret.get_ellipse_model_ransac(
            parray,
            xf=95,
            yf=95,
            rf_lim=20,
            semi_len_min=40,
            semi_len_max=100,
            semi_len_ratio_lim=5,
            max_trials=50,
        )
        s.add_ellipse_array_as_markers(
            ellipse_array, inlier_array=inlier_array.T, peak_array=parray.T
        )

    @pytest.mark.flaky(reruns=2)
    def test_only_ellipse_array(self):
        s, parray = dd.get_simple_ellipse_signal_peak_array(seed=15)
        s1 = s.deepcopy()
        ellipse_array, inlier_array = ret.get_ellipse_model_ransac(
            parray,
            xf=95,
            yf=95,
            rf_lim=20,
            semi_len_min=40,
            semi_len_max=100,
            semi_len_ratio_lim=5,
            max_trials=50,
        )
        s.add_ellipse_array_as_markers(ellipse_array)
        s1.add_ellipse_array_as_markers(
            ellipse_array, inlier_array=inlier_array.T, peak_array=parray.T
        )

    def test_wrong_input_dimensions(self):
        s = Diffraction2D(np.zeros((2, 5, 5)))
        with pytest.raises(ValueError):
            s.add_ellipse_array_as_markers(ellipse_array=None)


class TestDiffraction2DThresholdAndMask:
    @pytest.mark.parametrize("shape", [(9, 9, 9, 9), (4, 8, 6, 3), (6, 3, 2, 5)])
    def test_no_change(self, shape):
        s = Diffraction2D(np.zeros(shape))
        s1 = s.threshold_and_mask()
        assert (s.data == s1.data).all()

    @pytest.mark.parametrize("mask", [(3, 5, 1), (6, 8, 1), (4, 6, 1)])
    def test_mask(self, mask):
        s = Diffraction2D(np.ones((10, 10, 10, 10)))
        s1 = s.threshold_and_mask(mask=mask)
        slice0 = np.s_[:, :, mask[1] - mask[2] : mask[1] + mask[2] + 1, mask[0]]
        assert (s1.data[slice0] == 1.0).all()
        slice1 = np.s_[:, :, mask[1], mask[0] - mask[2] : mask[0] + mask[2] + 1]
        assert (s1.data[slice1] == 1.0).all()
        s1.data[slice0] = 0
        s1.data[slice1] = 0
        assert (s1.data == 0).all()

    @pytest.mark.parametrize("x, y", [(3, 5), (7, 5), (5, 2)])
    def test_threshold(self, x, y):
        s = Diffraction2D(np.random.randint(0, 10, size=(10, 10, 10, 10)))
        s.data[:, :, x, y] = 1000000
        s1 = s.threshold_and_mask(threshold=1)
        assert (s1.data[:, :, x, y] == 1.0).all()
        s1.data[:, :, x, y] = 0
        assert (s1.data == 0).all()

    def test_threshold_mask(self):
        s = Diffraction2D(np.zeros((12, 11, 13, 10)))
        s.data[:, :, 1, 2] = 1000000
        s.data[:, :, 8, 6] = 10
        s1 = s.threshold_and_mask(threshold=1)
        assert (s1.data[:, :, 1, 2] == 1.0).all()
        s1.data[:, :, 1, 2] = 0
        assert (s1.data == 0).all()

        s2 = s.threshold_and_mask(threshold=1, mask=(6, 8, 1))
        assert (s2.data[:, :, 8, 6] == 1.0).all()
        s2.data[:, :, 8, 6] = 0
        assert (s2.data == 0).all()

    def test_lazy_exception(self):
        s = dd.get_disk_shift_simple_test_signal(lazy=True)
        with pytest.raises(NotImplementedError):
            s.threshold_and_mask()


class TestDiffraction2DAngleSector:
    def test_get_angle_sector_mask_simple(self):
        array = np.zeros((10, 10, 10, 10))
        array[:, :, 0:5, 0:5] = 1
        s = Diffraction2D(array)
        s.axes_manager.signal_axes[0].offset = -4.5
        s.axes_manager.signal_axes[1].offset = -4.5
        mask = s.angular_mask(0.0, 0.5 * np.pi)
        assert mask[:, :, 0:5, 0:5].all()
        assert not mask[:, :, 5:, :].any()
        assert not mask[:, :, :, 5:].any()

    def test_com_angle_sector_mask(self):
        x, y = 4, 7
        array = np.zeros((5, 4, 10, 20))
        array[:, :, y, x] = 1
        s = Diffraction2D(array)
        s_bs = s.get_direct_beam_position(method="center_of_mass")
        s.angular_mask(
            0.0,
            0.5 * np.pi,
            centre_x_array=s_bs.isig[0].data,
            centre_y_array=s_bs.isig[1].data,
        )


class TestAngularSliceRadialAverage:
    def test_deprecated_method(self):
        s = Diffraction2D(np.zeros((2, 2, 10, 10)))
        with pytest.raises(Exception):
            s.angular_slice_radial_integration()


class TestDiffraction2DTemplateMatchDisk:
    def test_simple(self):
        s = Diffraction2D(np.random.randint(100, size=(5, 5, 20, 20)))
        s_template = s.template_match_disk()
        assert s.data.shape == s_template.data.shape
        assert not s_template._lazy

    def test_disk_radius(self):
        s = LazyDiffraction2D(
            da.random.randint(100, size=(5, 5, 30, 30), chunks=(1, 1, 10, 10))
        )
        s_template0 = s.template_match_disk(disk_r=2, lazy_output=False)
        s_template1 = s.template_match_disk(disk_r=4, lazy_output=False)
        assert s.data.shape == s_template0.data.shape
        assert s.data.shape == s_template1.data.shape
        assert not (s_template0.data == s_template1.data).all()

    @pytest.mark.parametrize("mode", ["constant", "reflect"])
    def test_template_circular(self, mode):
        s = Diffraction2D(np.random.randint(100, size=(5, 5, 20, 20)))
        s_template = s.template_match_disk(circular_background=True, mode=mode)
        assert s.data.shape == s_template.data.shape
        assert not s_template._lazy


class TestDiffraction2DTemplateMatchRing:
    def test_simple(self):
        s = Diffraction2D(np.random.randint(100, size=(5, 5, 20, 20)))
        s_template = s.template_match_ring(r_inner=3, r_outer=5, lazy_output=True)
        assert s.data.shape == s_template.data.shape
        assert s_template._lazy is True

    def test_wrong_input(self):
        s = Diffraction2D(np.random.randint(100, size=(5, 5, 20, 20)))
        with pytest.raises(ValueError):
            s.template_match_ring(r_inner=5, r_outer=3)
        with pytest.raises(ValueError):
            s.template_match_ring(r_inner=3, r_outer=3)


class TestDiffraction2DTemplate:
    def test_square_and_disk(self):
        s = Diffraction2D(np.zeros((2, 2, 100, 100)))

        square_image = np.zeros((9, 9))
        square_image[2:-2, 2:-2] = 1
        s.data[:, :, 20:29, 40:49] = square_image

        disk = morphology.disk(4, s.data.dtype)
        s.data[:, :, 60:69, 50:59] = disk

        s_st = s.template_match(square_image)
        s_dt = s.template_match(disk)

        st_ind = np.unravel_index(np.argmax(s_st.data, axis=None), s_st.data.shape)[-2:]
        dt_ind = np.unravel_index(np.argmax(s_dt.data, axis=None), s_dt.data.shape)[-2:]

        assert st_ind == (24, 44)
        assert dt_ind == (64, 54)
        assert s.data.shape == s_st.data.shape
        assert s.data.shape == s_dt.data.shape


# Remove in 1.0.0 Release
class TestDiffraction2DTemplateWithBinaryImage:
    def test_square_and_disk(self):
        s = Diffraction2D(np.zeros((2, 2, 100, 100)))

        square_image = np.zeros((9, 9))
        square_image[2:-2, 2:-2] = 1
        s.data[:, :, 20:29, 40:49] = square_image

        disk = morphology.disk(4, s.data.dtype)
        s.data[:, :, 60:69, 50:59] = disk

        s_st = s.template_match_with_binary_image(square_image, lazy_result=False)
        s_dt = s.template_match_with_binary_image(disk, lazy_result=False)

        st_ind = np.unravel_index(np.argmax(s_st.data, axis=None), s_st.data.shape)[-2:]
        dt_ind = np.unravel_index(np.argmax(s_dt.data, axis=None), s_dt.data.shape)[-2:]

        assert st_ind == (24, 44)
        assert dt_ind == (64, 54)
        assert s.data.shape == s_st.data.shape
        assert s.data.shape == s_dt.data.shape


class TestDiffraction2DRotateDiffraction:
    def test_rotate_diffraction_keep_shape(self):
        shape = (7, 5, 4, 15)
        s = Diffraction2D(np.zeros(shape))
        s_rot = s.rotate_diffraction(angle=45)
        assert s.axes_manager.shape == s_rot.axes_manager.shape

        s_lazy = LazyDiffraction2D(da.zeros(shape, chunks=(1, 1, 1, 1)))
        s_rot_lazy = s_lazy.rotate_diffraction(angle=45)
        assert s_lazy.axes_manager.shape == s_rot_lazy.axes_manager.shape

    def test_rotate_diffraction_values(self):
        data = np.zeros((10, 5, 12, 14))
        data[:, :, 6:, 7:] = 1
        s = Diffraction2D(data)
        s_rot = s.rotate_diffraction(angle=180)
        np.testing.assert_almost_equal(
            s.data[0, 0, :6, :7], np.zeros_like(s.data[0, 0, :6, :7])
        )
        np.testing.assert_almost_equal(
            s_rot.data[0, 0, :6, :7], np.ones_like(s_rot.data[0, 0, :6, :7])
        )
        s_rot.data[:, :, :6, :7] = 0
        np.testing.assert_almost_equal(s_rot.data, np.zeros_like(s.data))


class TestDiffraction2DShiftDiffraction:
    @pytest.mark.parametrize("shift_x,shift_y", [(2, 5), (-6, -1), (2, -4)])
    def test_single_shift(self, shift_x, shift_y):
        s = Diffraction2D(np.zeros((10, 10, 30, 40)))
        x, y = 20, 10
        s.data[:, :, y, x] = 1
        s_shift = s.shift_diffraction(shift_x=shift_x, shift_y=shift_y)
        assert s_shift.data[0, 0, y - shift_y, x - shift_x] == 1
        s_shift.data[:, :, y - shift_y, x - shift_x] = 0
        assert s_shift.data.sum() == 0

    @pytest.mark.parametrize("centre_x,centre_y", [(25, 25), (30, 20)])
    def test_random_shifts(self, centre_x, centre_y):
        y, x = np.mgrid[20:30:7j, 20:30:5j]
        s = mdtd.generate_4d_data(
            probe_size_x=5,
            probe_size_y=7,
            disk_x=x,
            disk_y=y,
            disk_r=1,
            blur=True,
            ring_x=None,
        )
        s_bs = s.get_direct_beam_position(method="center_of_mass")
        s_shift = s.shift_diffraction(
            shift_x=-s_bs.isig[0].data, shift_y=-s_bs.isig[1].data
        )
        s_shift_c = s_shift.get_direct_beam_position(method="center_of_mass")
        assert s_shift_c.data == pytest.approx(0.0)

    def test_inplace(self):
        s = Diffraction2D(np.zeros((10, 10, 30, 40)))
        x, y, shift_x, shift_y = 20, 10, 4, -3
        s.data[:, :, y, x] = 1
        s.shift_diffraction(shift_x=shift_x, shift_y=shift_y, inplace=True)
        assert s.data[0, 0, y - shift_y, x - shift_x] == 1
        s.data[:, :, y - shift_y, x - shift_x] = 0
        assert s.data.sum() == 0

    def test_lazy(self):
        data = np.zeros((10, 10, 30, 30))
        x, y = 20, 10
        data[:, :, y, x] += 1
        data = da.from_array(data, chunks=(5, 5, 5, 5))
        s = LazyDiffraction2D(data)
        shift_x, shift_y = 4, 3
        s_shift = s.shift_diffraction(shift_x=shift_x, shift_y=shift_y)
        s_shift.compute()
        assert s_shift.data[0, 0, y - shift_y, x - shift_x] == 1
        s_shift.data[:, :, y - shift_y, x - shift_x] = 0
        assert s_shift.data.sum() == 0


class TestComputeAndAsLazy:
    def test_2d_data_compute(self):
        dask_array = da.random.random((100, 150), chunks=(50, 50))
        s = LazyDiffraction2D(dask_array)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s.compute()
        assert s.__class__ == Diffraction2D
        assert not hasattr(s.data, "compute")
        assert s.axes_manager[0].scale == scale0
        assert s.axes_manager[1].scale == scale1
        assert s.metadata.Test == metadata_string
        assert dask_array.shape == s.data.shape

    def test_5d_data_compute(self):
        dask_array = da.random.random((2, 3, 4, 10, 15), chunks=(1, 1, 1, 10, 15))
        s = LazyDiffraction2D(dask_array)
        s.compute()
        assert s.__class__ == Diffraction2D
        assert dask_array.shape == s.data.shape

    def test_2d_data_as_lazy(self):
        data = np.random.random((100, 150))
        s = Diffraction2D(data)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s_lazy = s.as_lazy()
        assert s_lazy.__class__ == LazyDiffraction2D
        assert hasattr(s_lazy.data, "compute")
        assert s_lazy.axes_manager[0].scale == scale0
        assert s_lazy.axes_manager[1].scale == scale1
        assert s_lazy.metadata.Test == metadata_string
        assert data.shape == s_lazy.data.shape

    def test_5d_data_as_lazy(self):
        data = np.random.random((2, 3, 4, 10, 15))
        s = Diffraction2D(data)
        s_lazy = s.as_lazy()
        assert s_lazy.__class__ == LazyDiffraction2D
        assert data.shape == s_lazy.data.shape
