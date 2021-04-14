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

import pytest
import numpy as np
import dask.array as da

from hyperspy.signals import Signal2D

from pyxem.signals import Diffraction2D
import pyxem.utils.pixelated_stem_tools as pst


class TestRadialIntegrationDaskArray:
    def test_simple(self):
        dask_array = da.zeros((10, 10, 15, 15), chunks=(5, 5, 5, 5))
        centre_x, centre_y = np.ones((2, 100)) * 7.5
        data = pst._radial_average_dask_array(
            dask_array,
            return_sig_size=11,
            centre_x=centre_x,
            centre_y=centre_y,
            normalize=False,
            show_progressbar=False,
        )
        assert data.shape == (10, 10, 11)
        assert (data == 0.0).all()

    def test_different_size(self):
        dask_array = da.zeros((5, 10, 12, 15), chunks=(5, 5, 5, 5))
        centre_x, centre_y = np.ones((2, 100)) * 7.5
        data = pst._radial_average_dask_array(
            dask_array,
            return_sig_size=11,
            centre_x=centre_x,
            centre_y=centre_y,
            normalize=False,
            show_progressbar=False,
        )
        assert data.shape == (5, 10, 11)
        assert (data == 0.0).all()

    def test_mask(self):
        numpy_array = np.zeros((10, 10, 30, 30))
        numpy_array[:, :, 0, 0] = 1000
        numpy_array[:, :, -1, -1] = 1
        dask_array = da.from_array(numpy_array, chunks=(5, 5, 5, 5))
        centre_x, centre_y = np.ones((2, 100)) * 15
        data = pst._radial_average_dask_array(
            dask_array,
            return_sig_size=22,
            centre_x=centre_x,
            centre_y=centre_y,
            normalize=False,
            show_progressbar=False,
        )
        assert data.shape == (10, 10, 22)
        assert (data != 0.0).any()
        mask = pst._make_circular_mask(15, 15, 30, 30, 15)
        data = pst._radial_average_dask_array(
            dask_array,
            return_sig_size=22,
            centre_x=centre_x,
            centre_y=centre_y,
            normalize=False,
            mask_array=mask,
            show_progressbar=False,
        )
        assert data.shape == (10, 10, 22)
        assert (data == 0.0).all()


class TestThresholdAndMaskSingleFrame:
    def test_no_data_change(self):
        im = np.random.random((52, 43))
        im_out = pst._threshold_and_mask_single_frame(im)
        assert (im == im_out).all()

    def test_mask_all_zeros(self):
        im = np.zeros((12, 45))
        im[2, 10] = 10
        mask = pst._make_circular_mask(6, 31, 45, 12, 2)
        im_out = pst._threshold_and_mask_single_frame(im, mask=mask)
        assert (im_out == 0).all()

    def test_mask_not_all_zeros(self):
        im = np.zeros((12, 45))
        im[7, 31] = 10
        im[2, 43] = 5
        mask = pst._make_circular_mask(32, 7, 45, 12, 2)
        im_out = pst._threshold_and_mask_single_frame(im, mask=mask)
        assert im_out[7, 31] == 10
        im_out[7, 31] = 0
        assert (im_out == 0).all()

    def test_threshold(self):
        im = np.ones((12, 45))
        im[7, 12] = 10000000
        im_out = pst._threshold_and_mask_single_frame(im, threshold=2)
        assert im_out[7, 12] == 1
        im_out[7, 12] = 0
        assert (im_out == 0).all()


class TestPixelatedTools:
    def test_find_longest_distance_manual(self):
        # These values are tested manually, against knowns results,
        # to make sure everything works fine.
        imX, imY = 10, 10
        centre_list = ((0, 0), (imX, 0), (0, imY), (imX, imY))
        distance = 14
        for cX, cY in centre_list:
            dist = pst._find_longest_distance(imX, imY, cX, cY, cX, cY)
            assert dist == distance

        centre_list = ((1, 1), (imX - 1, 1), (1, imY - 1), (imX - 1, imY - 1))
        distance = 12
        for cX, cY in centre_list:
            dist = pst._find_longest_distance(imX, imY, cX, cY, cX, cY)
            assert dist == distance

        imX, imY = 10, 5
        centre_list = ((0, 0), (imX, 0), (0, imY), (imX, imY))
        distance = 11
        for cX, cY in centre_list:
            dist = pst._find_longest_distance(imX, imY, cX, cY, cX, cY)
            assert dist == distance

        imX, imY = 10, 10
        cX_min, cX_max, cY_min, cY_max = 1, 2, 2, 3
        distance = 12
        dist = pst._find_longest_distance(imX, imY, cX_min, cX_max, cY_min, cY_max)
        assert dist == distance

    def test_find_longest_distance_all(self):
        imX, imY = 100, 100
        x_array, y_array = np.mgrid[0:10, 0:10]
        for x, y in zip(x_array.flatten(), y_array.flatten()):
            distance = int(((imX - x) ** 2 + (imY - y) ** 2) ** 0.5)
            dist = pst._find_longest_distance(imX, imY, x, y, x, y)
            assert dist == distance

        x_array, y_array = np.mgrid[90:100, 90:100]
        for x, y in zip(x_array.flatten(), y_array.flatten()):
            distance = int((x ** 2 + y ** 2) ** 0.5)
            dist = pst._find_longest_distance(imX, imY, x, y, x, y)
            assert dist == distance

        x_array, y_array = np.mgrid[0:10, 90:100]
        for x, y in zip(x_array.flatten(), y_array.flatten()):
            distance = int(((imX - x) ** 2 + y ** 2) ** 0.5)
            dist = pst._find_longest_distance(imX, imY, x, y, x, y)
            assert dist == distance

        x_array, y_array = np.mgrid[90:100, 0:10]
        for x, y in zip(x_array.flatten(), y_array.flatten()):
            distance = int((x ** 2 + (imY - y) ** 2) ** 0.5)
            dist = pst._find_longest_distance(imX, imY, x, y, x, y)
            assert dist == distance

    def test_make_centre_array_from_signal(self):
        s = Signal2D(np.ones((5, 10, 20, 7)))
        sa = s.axes_manager.signal_axes
        offset_x = sa[0].offset
        offset_y = sa[1].offset
        mask = pst._make_centre_array_from_signal(s)
        assert mask[0].shape[::-1] == s.axes_manager.navigation_shape
        assert mask[1].shape[::-1] == s.axes_manager.navigation_shape
        assert (offset_x == mask[0]).all()
        assert (offset_y == mask[1]).all()

        offset0_x, offset0_y = -3, -2
        sa[0].offset, sa[1].offset = offset0_x, offset0_y
        mask = pst._make_centre_array_from_signal(s)
        assert (-offset0_x == mask[0]).all()
        assert (-offset0_y == mask[1]).all()


class TestShiftSingleFrame:
    @pytest.mark.parametrize("x,y", [(1, 1), (-2, 2), (-2, 5), (-4, -3)])
    def test_simple_shift(self, x, y):
        im = np.zeros((20, 20))
        x0, y0 = 10, 12
        im[x0, y0] = 1
        # Note that the shifts are switched when calling the function,
        # to stay consistent with the HyperSpy axis ordering
        im_shift = pst._shift_single_frame(im=im, shift_x=y, shift_y=x)
        assert im_shift[x0 - x, y0 - y] == 1
        assert im_shift.sum() == 1
        im_shift[x0 - x, y0 - y] = 0
        assert (im_shift == 0.0).all()

    @pytest.mark.parametrize("x,y", [(0.5, 1), (-4, 2.5), (-6, 1.5), (-3.5, -4)])
    def test_single_half_shift(self, x, y):
        im = np.zeros((20, 20))
        x0, y0 = 10, 12
        im[x0, y0] = 1
        # Note that the shifts are switched when calling the function,
        # to stay consistent with the HyperSpy axis ordering
        im_shift = pst._shift_single_frame(im=im, shift_x=y, shift_y=x)
        assert im_shift[x0 - int(x), y0 - int(y)] == 0.5
        assert im_shift.max() == 0.5
        assert im_shift.sum() == 1

    @pytest.mark.parametrize(
        "x,y", [(0.5, 1.5), (-4.5, 2.5), (-6.5, 1.5), (-3.5, -4.5)]
    )
    def test_two_half_shift(self, x, y):
        im = np.zeros((20, 20))
        x0, y0 = 10, 12
        im[x0, y0] = 1
        # Note that the shifts are switched when calling the function,
        # to stay consistent with the HyperSpy axis ordering
        im_shift = pst._shift_single_frame(im=im, shift_x=y, shift_y=x)
        assert im_shift[x0 - int(x), y0 - int(y)] == 0.25
        assert im_shift.max() == 0.25
        assert im_shift.sum() == 1

    def test_interpolation_order(self):
        im = np.arange(0, 100).reshape((10, 10))
        x, y = 2.5, -3.5
        im_shift0 = pst._shift_single_frame(
            im=im, shift_x=x, shift_y=y, interpolation_order=0
        )
        im_shift1 = pst._shift_single_frame(
            im=im, shift_x=x, shift_y=y, interpolation_order=1
        )
        im_shift2 = pst._shift_single_frame(
            im=im, shift_x=x, shift_y=y, interpolation_order=2
        )
        assert not (im_shift0 == im_shift1).all()
        assert not (im_shift1 == im_shift2).all()
        assert not (im_shift0 == im_shift2).all()


class TestGetAngleSectorMask:
    def test_0d(self):
        data_shape = (10, 8)
        data = np.ones(data_shape) * 100.0
        s = Signal2D(data)
        mask0 = pst._get_angle_sector_mask(s, angle0=0.0, angle1=np.pi / 2)
        np.testing.assert_array_equal(mask0, np.zeros_like(mask0, dtype=bool))
        assert mask0.shape == data_shape

        s.axes_manager.signal_axes[0].offset = -5
        s.axes_manager.signal_axes[1].offset = -5
        mask1 = pst._get_angle_sector_mask(s, angle0=0.0, angle1=np.pi / 2)
        assert mask1[:5, :5].all()
        mask1[:5, :5] = False
        np.testing.assert_array_equal(mask1, np.zeros_like(mask1, dtype=bool))

        s.axes_manager.signal_axes[0].offset = -15
        s.axes_manager.signal_axes[1].offset = -15
        mask2 = pst._get_angle_sector_mask(s, angle0=0.0, angle1=np.pi / 2)
        assert mask2.all()

        s.axes_manager.signal_axes[0].offset = -15
        s.axes_manager.signal_axes[1].offset = -15
        mask3 = pst._get_angle_sector_mask(s, angle0=np.pi * 3 / 2, angle1=np.pi * 2)
        assert not mask3.any()

    def test_0d_com(self):
        data_shape = (10, 8)
        data = np.zeros(data_shape)
        s = Diffraction2D(data)
        s.data[4, 5] = 5.0
        s_com = s.center_of_mass()
        pst._get_angle_sector_mask(
            s,
            centre_x_array=s_com.inav[0].data,
            centre_y_array=s_com.inav[1].data,
            angle0=np.pi * 3 / 2,
            angle1=np.pi * 2,
        )

    def test_1d(self):
        data_shape = (5, 7, 10)
        data = np.ones(data_shape) * 100.0
        s = Signal2D(data)
        mask0 = pst._get_angle_sector_mask(s, angle0=0.0, angle1=np.pi / 2)
        np.testing.assert_array_equal(mask0, np.zeros_like(mask0, dtype=bool))
        assert mask0.shape == data_shape

        s.axes_manager.signal_axes[0].offset = -5
        s.axes_manager.signal_axes[1].offset = -4
        mask1 = pst._get_angle_sector_mask(s, angle0=0.0, angle1=np.pi / 2)
        assert mask1[:, :4, :5].all()
        mask1[:, :4, :5] = False
        np.testing.assert_array_equal(mask1, np.zeros_like(mask1, dtype=bool))

    def test_2d(self):
        data_shape = (3, 5, 7, 10)
        data = np.ones(data_shape) * 100.0
        s = Signal2D(data)
        mask0 = pst._get_angle_sector_mask(s, angle0=np.pi, angle1=2 * np.pi)
        assert mask0.shape == data_shape

    def test_3d(self):
        data_shape = (5, 3, 5, 7, 10)
        data = np.ones(data_shape) * 100.0
        s = Signal2D(data)
        mask0 = pst._get_angle_sector_mask(s, angle0=np.pi, angle1=2 * np.pi)
        assert mask0.shape == data_shape

    def test_centre_xy(self):
        data_shape = (3, 5, 7, 10)
        data = np.ones(data_shape) * 100.0
        s = Signal2D(data)
        nav_shape = s.axes_manager.navigation_shape
        centre_x, centre_y = np.ones(nav_shape[::-1]) * 5, np.ones(nav_shape[::-1]) * 9
        mask0 = pst._get_angle_sector_mask(
            s,
            angle0=np.pi,
            angle1=2 * np.pi,
            centre_x_array=centre_x,
            centre_y_array=centre_y,
        )
        assert mask0.shape == data_shape

    def test_bad_angles(self):
        s = Signal2D(np.zeros((100, 100)))
        with pytest.raises(ValueError):
            pst._get_angle_sector_mask(s, angle0=2, angle1=-1)

    def test_angles_across_pi(self):
        s = Signal2D(np.zeros((100, 100)))
        s.axes_manager[0].offset, s.axes_manager[1].offset = -49.5, -49.5
        mask0 = pst._get_angle_sector_mask(s, 0.5 * np.pi, 1.5 * np.pi)
        assert np.invert(mask0[:, 0:50]).all()
        assert mask0[:, 50:].all()

        mask1 = pst._get_angle_sector_mask(s, 2.5 * np.pi, 3.5 * np.pi)
        assert np.invert(mask1[:, 0:50]).all()
        assert mask1[:, 50:].all()

        mask2 = pst._get_angle_sector_mask(s, 4.5 * np.pi, 5.5 * np.pi)
        assert np.invert(mask2[:, 0:50]).all()
        assert mask2[:, 50:].all()

        mask3 = pst._get_angle_sector_mask(s, -3.5 * np.pi, -2.5 * np.pi)
        assert np.invert(mask3[:, 0:50]).all()
        assert mask3[:, 50:].all()

    def test_angles_across_zero(self):
        s = Signal2D(np.zeros((100, 100)))
        s.axes_manager[0].offset, s.axes_manager[1].offset = -50, -50
        mask0 = pst._get_angle_sector_mask(s, -0.5 * np.pi, 0.5 * np.pi)
        assert mask0[:, 0:50].all()
        assert np.invert(mask0[:, 50:]).all()

        mask1 = pst._get_angle_sector_mask(s, 1.5 * np.pi, 2.5 * np.pi)
        assert mask1[:, 0:50].all()
        assert np.invert(mask1[:, 50:]).all()

        mask2 = pst._get_angle_sector_mask(s, 3.5 * np.pi, 4.5 * np.pi)
        assert mask2[:, 0:50].all()
        assert np.invert(mask2[:, 50:]).all()

        mask3 = pst._get_angle_sector_mask(s, -4.5 * np.pi, -3.5 * np.pi)
        assert mask3[:, 0:50].all()
        assert np.invert(mask3[:, 50:]).all()

    def test_angles_more_than_2pi(self):
        s = Signal2D(np.zeros((100, 100)))
        s.axes_manager[0].offset, s.axes_manager[1].offset = -50, -50
        mask0 = pst._get_angle_sector_mask(s, 0.1 * np.pi, 4 * np.pi)
        assert mask0.all()

        mask1 = pst._get_angle_sector_mask(s, -1 * np.pi, 2.1 * np.pi)
        assert mask1.all()


class TestAxesManagerMetadataCopying:
    def setup_method(self):
        s = Signal2D(np.zeros((50, 50)))
        s.axes_manager.signal_axes[0].offset = 10
        s.axes_manager.signal_axes[1].offset = 20
        s.axes_manager.signal_axes[0].scale = 0.5
        s.axes_manager.signal_axes[1].scale = 0.3
        s.axes_manager.signal_axes[0].name = "axes0"
        s.axes_manager.signal_axes[1].name = "axes1"
        s.axes_manager.signal_axes[0].units = "unit0"
        s.axes_manager.signal_axes[1].units = "unit1"
        self.s = s

    def test_copy_axes_manager(self):
        s = self.s
        s_new = Signal2D(np.zeros((50, 50)))
        pst._copy_signal2d_axes_manager_metadata(s, s_new)
        sa_ori = s.axes_manager.signal_axes
        sa_new = s_new.axes_manager.signal_axes
        assert sa_ori[0].offset == sa_new[0].offset
        assert sa_ori[1].offset == sa_new[1].offset
        assert sa_ori[0].scale == sa_new[0].scale
        assert sa_ori[1].scale == sa_new[1].scale
        assert sa_ori[0].name == sa_new[0].name
        assert sa_ori[1].name == sa_new[1].name
        assert sa_ori[0].units == sa_new[0].units
        assert sa_ori[1].units == sa_new[1].units

    def test_copy_axes_object(self):
        s = self.s
        s_new = Signal2D(np.zeros((50, 50)))
        ax_o = s.axes_manager[0]
        ax_n = s_new.axes_manager[0]
        pst._copy_axes_object_metadata(ax_o, ax_n)
        assert ax_o.offset == ax_n.offset
        assert ax_o.scale == ax_n.scale
        assert ax_o.name == ax_n.name
        assert ax_o.units == ax_n.units

    def test_copy_all_axes_manager(self):
        s = Signal2D(np.zeros((5, 40, 50)))
        s_new = Signal2D(np.zeros_like(s.data))
        sa_ori = s.axes_manager
        sa_ori[0].scale = 0.2
        sa_ori[1].scale = 2.2
        sa_ori[2].scale = 0.02
        sa_ori[0].offset = 321
        sa_ori[1].offset = -232
        sa_ori[2].offset = 32
        pst._copy_signal_all_axes_metadata(s, s_new)
        sa_new = s_new.axes_manager
        assert sa_ori[0].scale == sa_new[0].scale
        assert sa_ori[1].scale == sa_new[1].scale
        assert sa_ori[2].scale == sa_new[2].scale
        assert sa_ori[0].offset == sa_new[0].offset
        assert sa_ori[1].offset == sa_new[1].offset
        assert sa_ori[2].offset == sa_new[2].offset

    def test_copy_all_axes_manager_wrong_shape(self):
        s0 = Signal2D(np.zeros((5, 40, 50)))
        s1 = Signal2D(np.zeros((6, 40, 50)))
        s2 = Signal2D(np.zeros((2, 5, 40, 50)))
        with pytest.raises(ValueError):
            pst._copy_signal_all_axes_metadata(s0, s1)
        with pytest.raises(ValueError):
            pst._copy_signal_all_axes_metadata(s0, s2)
