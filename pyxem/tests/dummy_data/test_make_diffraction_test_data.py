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
from pytest import approx
import numpy as np
from hyperspy.signals import Signal2D
from pyxem.dummy_data import make_diffraction_test_data as mdtd


class TestMakeTestData:
    def test_init(self):
        mdtd.MakeTestData(default=True)
        mdtd.MakeTestData(size_x=1, size_y=10, scale=0.05)
        mdtd.MakeTestData(
            size_x=200,
            size_y=100,
            scale=2,
            default=False,
            blur=False,
            blur_sigma=3.0,
            downscale=False,
        )

    def test_zero_signal(self):
        test_data_1 = mdtd.MakeTestData(default=False)
        assert (test_data_1.signal.data == 0.0).all()

        test_data_2 = mdtd.MakeTestData(default=True)
        test_data_2.set_signal_zero()
        assert (test_data_2.signal.data == 0.0).all()

    def test_repr(self):
        test = mdtd.MakeTestData(size_x=250)
        repr_string = test.__repr__()
        assert str(250) in repr_string


def test_circle_repr():
    xx, yy = np.zeros((2, 10, 10))
    circle = mdtd.Circle(xx, yy, x0=7, y0=5, r=3, intensity=2, scale=1)
    repr_string = circle.__repr__()
    assert str(7) in repr_string


class TestMakeDiffractionTestDataDisks:
    def test_simple_disks(self):
        test0 = mdtd.MakeTestData(
            size_x=100, size_y=100, default=False, blur=False, downscale=False
        )
        test0.add_disk(50, 50, 1, 20)

        assert (test0.signal.isig[49:52, 49:52].data == 20).all()
        test0.signal.data[49:52, 49:52] = 0
        assert not test0.signal.data.any()

        test1 = mdtd.MakeTestData(
            size_x=100, size_y=100, default=False, blur=False, downscale=False
        )
        test1.add_disk(20, 40, 2, 30)
        assert (test1.signal.isig[18:23, 38:43].data == 30).all()
        test1.signal.data[38:43, 18:23] = 0
        assert not test1.signal.data.any()

        test2 = mdtd.MakeTestData(
            size_x=80, size_y=120, default=False, blur=False, downscale=False
        )
        test2.add_disk(50, 40, 1, 10)
        test2.add_disk(20, 30, 2, 30)
        assert (test2.signal.isig[49:52, 39:42].data == 10).all()
        assert (test2.signal.isig[19:22, 29:32].data == 30).all()
        test2.signal.data[39:42, 49:52] = 0
        test2.signal.data[28:33, 18:23] = 0
        assert not test2.signal.data.any()

        test3 = mdtd.MakeTestData(
            size_x=150, size_y=50, default=False, blur=False, downscale=False
        )
        test3.add_disk(200, 400, 2, 30)
        assert not test3.signal.data.any()

        test4 = mdtd.MakeTestData(
            size_x=150, size_y=50, default=False, blur=False, downscale=False
        )
        test4.add_disk(50, 50, 500, 500)
        assert (test4.signal.data == 500).all()

        test5 = mdtd.MakeTestData(
            size_x=100, size_y=200, default=False, blur=False, downscale=True
        )
        test5.add_disk(50, 50, 500, 10)
        assert (test5.signal.data == 10).all()

        test6 = mdtd.MakeTestData(
            size_x=100, size_y=200, default=False, blur=True, downscale=False
        )
        test6.add_disk(50, 30, 500, 10)
        test6_ref = np.full_like(test6.signal.data, 10.0)
        np.testing.assert_allclose(test6.signal.data, test6_ref, rtol=1e-05)

    def test_large_disk(self):
        test_data_1 = mdtd.MakeTestData(size_x=10, size_y=10, scale=0.01, default=False)
        test_data_1.add_disk(x0=5, y0=5, r=20, intensity=100)
        assert (test_data_1.signal.data > 0.0).all()

    def test_repr(self):
        test = mdtd.MakeTestData()
        test.add_disk(x0=70)
        repr_string = test.z_list[0].__repr__()
        assert str(70) in repr_string


class TestMakeDiffractionTestDataRing:
    def test_ring_inner_radius(self):
        r, lw = 20, 2
        x0, y0, scale = 50, 50, 1
        test_data = mdtd.MakeTestData(
            size_x=100, size_y=100, default=False, blur=False, downscale=False
        )
        test_data.add_ring(x0=x0, y0=y0, r=r, intensity=10, lw_pix=lw)
        r_inner = 20 - 2.5 * scale

        s_h0 = test_data.signal.isig[x0, : y0 + 1]
        s_h0_edge = s_h0.axes_manager[0].index2value(s_h0.data[::-1].argmax())
        r_h0 = s_h0_edge - 0.5 * scale
        assert r_h0 == r_inner

        s_h1 = test_data.signal.isig[x0, y0:]
        s_h1_edge = s_h1.axes_manager[0].index2value(s_h1.data.argmax())
        r_h1 = s_h1_edge - 0.5 * scale - 50
        assert r_h1 == r_inner

        s_h2 = test_data.signal.isig[: x0 + 1, y0]
        s_h2_edge = s_h2.axes_manager[0].index2value(s_h2.data[::-1].argmax())
        r_h2 = s_h2_edge - 0.5 * scale
        assert r_h2 == r_inner

        s_h3 = test_data.signal.isig[x0:, y0]
        s_h3_edge = s_h3.axes_manager[0].index2value(s_h3.data.argmax())
        r_h3 = s_h3_edge - 0.5 * scale - 50
        assert r_h3 == r_inner

    def test_ring_outer_radius(self):
        r, lw = 20, 1
        x0, y0, scale = 0, 0, 1
        ring_1 = mdtd.MakeTestData(
            size_x=100, size_y=100, default=False, blur=False, downscale=False
        )
        ring_1.add_ring(x0=x0, y0=y0, r=r, intensity=10, lw_pix=lw)
        s_h0 = ring_1.signal.isig[x0, :]
        s_h0_edge = s_h0.axes_manager[0].index2value(s_h0.data[::-1].argmax())
        r_h0 = s_h0.data.size - s_h0_edge - 0.5 * scale
        r_out = r + lw + 0.5 * scale
        assert r_h0 == r_out
        s_h1 = ring_1.signal.isig[:, y0]
        s_h1_edge = s_h1.axes_manager[0].index2value(s_h1.data[::-1].argmax())
        r_h1 = s_h1.data.size - s_h1_edge - 0.5 * scale
        assert r_h1 == r_out

        r, lw = 20, 1
        x0, y0, scale = 100, 100, 1
        ring_2 = mdtd.MakeTestData(
            size_x=100, size_y=100, default=False, blur=False, downscale=False
        )
        ring_2.add_ring(x0=x0, y0=y0, r=r, intensity=10, lw_pix=lw)
        s_h2 = ring_2.signal.isig[-1, :]
        s_h2_edge = s_h2.axes_manager[0].index2value(s_h2.data.argmax())
        r_h2 = 100 - (s_h2_edge - 0.5 * scale)
        r_out = r + lw + 0.5 * scale
        assert r_h2 == r_out
        s_h3 = ring_2.signal.isig[:, -1]
        s_h3_edge = s_h3.axes_manager[0].index2value(s_h3.data.argmax())
        r_h3 = 100 - (s_h3_edge - 0.5 * scale)
        assert r_h3 == r_out

    def test_ring_radius1(self):
        intensity = 20
        test = mdtd.MakeTestData(
            size_x=100, size_y=100, default=False, blur=False, downscale=False
        )
        test.add_ring(x0=50, y0=50, r=1, intensity=intensity, lw_pix=0)

        assert (test.signal.isig[50, 50].data == 0.0).all()
        test.signal.data[50, 50] = intensity
        assert (test.signal.isig[49:52, 49:52].data == intensity).all()
        test.signal.data[49:52, 49:52] = 0.0
        assert not test.signal.data.any()

    def test_ring_radius2(self):
        intensity = 10
        test = mdtd.MakeTestData(
            size_x=100, size_y=100, default=False, blur=False, downscale=False
        )
        test.add_ring(x0=50, y0=50, r=2, intensity=intensity, lw_pix=0)

        assert (test.signal.isig[49:52, 49:52].data == 0.0).all()
        test.signal.data[49:52, 49:52] = intensity
        assert (test.signal.isig[48:53, 48:53].data == intensity).all()
        test.signal.data[48:53, 48:53] = 0.0
        assert not test.signal.data.any()

    def test_ring_radius_outside_image(self):
        test = mdtd.MakeTestData(
            size_x=100, size_y=100, default=False, blur=False, downscale=False
        )
        test.add_ring(x0=50, y0=50, r=300, intensity=19, lw_pix=1)
        assert not test.signal.data.any()

    def test_ring_rectangle_image(self):
        intensity = 10
        test = mdtd.MakeTestData(
            size_x=100, size_y=50, default=False, blur=False, downscale=False
        )
        test.add_ring(x0=50, y0=25, r=1, intensity=intensity, lw_pix=0)

        assert (test.signal.isig[50, 25].data == 0.0).all()
        test.signal.data[25, 50] = intensity
        assert (test.signal.isig[49:52, 24:27].data == intensity).all()
        test.signal.data[24:27, 49:52] = 0.0
        assert not test.signal.data.any()

    def test_ring_cover_whole_image(self):
        intensity = 10.0
        test = mdtd.MakeTestData(
            size_x=50, size_y=100, default=False, blur=False, downscale=False
        )
        test.add_ring(x0=25, y0=200, r=150, intensity=intensity, lw_pix=100)
        assert (test.signal.data == intensity).all()

    def test_ring_position(self):
        x0, y0, r = 40.0, 60.0, 20
        test = mdtd.MakeTestData(
            size_x=100, size_y=100, default=False, blur=False, downscale=False
        )
        test.add_ring(x0=x0, y0=y0, r=r, intensity=20, lw_pix=0)
        s_h0 = test.signal.isig[x0, 0.0:y0]
        max_h0 = s_h0.axes_manager[0].index2value(s_h0.data.argmax())
        assert max_h0 == y0 - r
        s_h1 = test.signal.isig[x0, y0:]
        max_h1 = s_h1.axes_manager[0].index2value(s_h1.data.argmax())
        assert max_h1 == y0 + r

        s_v0 = test.signal.isig[0:x0, y0]
        max_v0 = s_v0.axes_manager[0].index2value(s_v0.data.argmax())
        assert max_v0 == x0 - r
        s_v1 = test.signal.isig[x0:, y0]
        max_v1 = s_v1.axes_manager[0].index2value(s_v1.data.argmax())
        assert max_v1 == x0 + r

    def test_ring_position_blur(self):
        x0, y0, r = 50, 50, 15
        test = mdtd.MakeTestData(
            size_x=100, size_y=100, default=False, blur=True, downscale=False
        )
        test.add_ring(x0=x0, y0=y0, r=r, intensity=20, lw_pix=1)
        s_h0 = test.signal.isig[x0, 0:y0]
        max_h0 = s_h0.axes_manager[0].index2value(s_h0.data.argmax())
        assert max_h0 == y0 - r
        s_h1 = test.signal.isig[x0, y0:]
        max_h1 = s_h1.axes_manager[0].index2value(s_h1.data.argmax())
        assert max_h1 == y0 + r

        s_v0 = test.signal.isig[0:x0, y0]
        max_v0 = s_v0.axes_manager[0].index2value(s_v0.data.argmax())
        assert max_v0 == x0 - r
        s_v1 = test.signal.isig[x0:, y0]
        max_v1 = s_v1.axes_manager[0].index2value(s_v1.data.argmax())
        assert max_v1 == x0 + r

    def test_ring_position_blur_lw(self):
        x0, y0, r = 50, 50, 15
        test = mdtd.MakeTestData(
            size_x=100, size_y=100, default=False, blur=True, downscale=False
        )
        test.add_ring(x0=x0, y0=y0, r=r, intensity=20, lw_pix=1)
        s_h0 = test.signal.isig[x0, 0:y0]
        max_h0 = s_h0.axes_manager[0].index2value(s_h0.data.argmax())
        assert max_h0 == y0 - r
        s_h1 = test.signal.isig[x0, y0:]
        max_h1 = s_h1.axes_manager[0].index2value(s_h1.data.argmax())
        assert max_h1 == y0 + r

        s_v0 = test.signal.isig[0:x0, y0]
        max_v0 = s_v0.axes_manager[0].index2value(s_v0.data.argmax())
        assert max_v0 == x0 - r
        s_v1 = test.signal.isig[x0:, y0]
        max_v1 = s_v1.axes_manager[0].index2value(s_v1.data.argmax())
        assert max_v1 == x0 + r

    def test_ring_position_downscale(self):
        x0, y0, r = 50, 50, 15
        test = mdtd.MakeTestData(
            size_x=100, size_y=100, default=False, blur=False, downscale=True
        )
        test.add_ring(x0=x0, y0=y0, r=r, intensity=20, lw_pix=0)
        s_h0 = test.signal.isig[x0, 0:y0]
        max_h0 = s_h0.axes_manager[0].index2value(s_h0.data.argmax())
        assert max_h0 == y0 - r
        s_h1 = test.signal.isig[x0, y0:]
        max_h1 = s_h1.axes_manager[0].index2value(s_h1.data.argmax())
        assert max_h1 == y0 + r

        s_v0 = test.signal.isig[0:x0, y0]
        max_v0 = s_v0.axes_manager[0].index2value(s_v0.data.argmax())
        assert max_v0 == x0 - r
        s_v1 = test.signal.isig[x0:, y0]
        max_v1 = s_v1.axes_manager[0].index2value(s_v1.data.argmax())
        assert max_v1 == x0 + r

    def test_ring_position_downscale_and_blur(self):
        x0, y0, r = 50, 50, 18
        test = mdtd.MakeTestData(
            size_x=100, size_y=100, default=False, blur=True, downscale=True
        )
        test.add_ring(x0=x0, y0=y0, r=r, intensity=20, lw_pix=1)
        s_h0 = test.signal.isig[x0, 0:y0]
        max_h0 = s_h0.axes_manager[0].index2value(s_h0.data.argmax())
        assert max_h0 == y0 - r
        s_h1 = test.signal.isig[x0, y0:]
        max_h1 = s_h1.axes_manager[0].index2value(s_h1.data.argmax())
        assert max_h1 == y0 + r

        s_v0 = test.signal.isig[0:x0, y0]
        max_v0 = s_v0.axes_manager[0].index2value(s_v0.data.argmax())
        assert max_v0 == x0 - r
        s_v1 = test.signal.isig[x0:, y0]
        max_v1 = s_v1.axes_manager[0].index2value(s_v1.data.argmax())
        assert max_v1 == x0 + r

    def test_repr(self):
        test = mdtd.MakeTestData()
        test.add_ring(x0=70)
        repr_string = test.z_list[0].__repr__()
        assert str(70) in repr_string

    def test_bad_input_too_wide_ring(self):
        test = mdtd.MakeTestData()
        with pytest.raises(ValueError):
            test.add_ring(r=5, lw_pix=10)


class TestMakeDiffractionTestDataDisksEllipse:
    def test_disk_cover_all(self):
        test = mdtd.MakeTestData(
            size_x=120, size_y=100, default=False, blur=False, downscale=False
        )
        test.add_disk_ellipse(x0=50, y0=50, semi_len0=100, semi_len1=130, intensity=2)
        assert (test.signal.data == 2).all()

    def test_with_downscale_blur_default(self):
        test = mdtd.MakeTestData(size_x=120, size_y=100)
        test.add_disk_ellipse(x0=50, y0=50, semi_len0=10, semi_len1=13, intensity=2)

    def test_disk_cover_nothing(self):
        test = mdtd.MakeTestData(
            size_x=120, size_y=100, default=False, blur=False, downscale=False
        )
        test.add_disk_ellipse(x0=350, y0=50, semi_len0=10, semi_len1=10, intensity=2)
        assert not test.signal.data.any()

    def test_disk_semi_len0_very_small(self):
        test = mdtd.MakeTestData(
            size_x=120, size_y=100, default=False, blur=False, downscale=False
        )
        x0, y0 = 50, 40
        intensity = 2
        test.add_disk_ellipse(
            x0=x0, y0=y0, semi_len0=0.1, semi_len1=1000, intensity=intensity, rotation=0
        )
        data = test.signal.data
        assert (data[:, x0] == intensity).all()
        data[:, x0] = 0
        assert not data.any()

    def test_disk_semi_len1_very_small(self):
        test = mdtd.MakeTestData(
            size_x=120, size_y=100, default=False, blur=False, downscale=False
        )
        x0, y0 = 50, 40
        intensity = 2
        test.add_disk_ellipse(
            x0=x0, y0=y0, semi_len0=1000, semi_len1=0.1, intensity=intensity, rotation=0
        )
        data = test.signal.data
        assert (data[y0, :] == intensity).all()
        data[y0, :] = 0
        assert not data.any()

    def test_disk_semi_len1_very_small_rot_90(self):
        test = mdtd.MakeTestData(
            size_x=120, size_y=100, default=False, blur=False, downscale=False
        )
        x0, y0 = 50, 40
        intensity = 2
        test.add_disk_ellipse(
            x0=x0,
            y0=y0,
            semi_len0=1000,
            semi_len1=0.1,
            intensity=intensity,
            rotation=np.pi / 2,
        )
        data = test.signal.data
        assert (data[:, x0] == intensity).all()
        data[:, x0] = 0
        assert not data.any()

    def test_disk_semi_len1_very_small_180(self):
        test = mdtd.MakeTestData(
            size_x=120, size_y=100, default=False, blur=False, downscale=False
        )
        x0, y0 = 50, 40
        intensity = 2
        test.add_disk_ellipse(
            x0=x0,
            y0=y0,
            semi_len0=1000,
            semi_len1=0.1,
            intensity=intensity,
            rotation=np.pi,
        )
        data = test.signal.data
        assert (data[y0, :] == intensity).all()
        data[y0, :] = 0
        assert not data.any()

    def test_repr(self):
        test = mdtd.MakeTestData()
        test.add_disk_ellipse(x0=70)
        repr_string = test.z_list[0].__repr__()
        assert str(70) in repr_string


class TestMakeDiffractionTestDataDisksRing:
    def test_downscale(self):
        test0 = mdtd.MakeTestData(
            size_x=120, size_y=100, downscale=True, blur=False, default=False
        )
        test0.add_ring_ellipse(x0=50, y0=50, semi_len0=8, semi_len1=13, intensity=2)
        test1 = mdtd.MakeTestData(
            size_x=120, size_y=100, downscale=False, blur=False, default=False
        )
        test1.add_ring_ellipse(x0=50, y0=50, semi_len0=8, semi_len1=13, intensity=2)
        assert not (test0.signal.data == test1.signal.data).all()

    def test_blur(self):
        test0 = mdtd.MakeTestData(
            size_x=120, size_y=100, downscale=False, blur=True, default=False
        )
        test0.add_ring_ellipse(x0=50, y0=50, semi_len0=8, semi_len1=13, intensity=2)
        test1 = mdtd.MakeTestData(
            size_x=120, size_y=100, downscale=False, blur=False, default=False
        )
        test1.add_ring_ellipse(x0=50, y0=50, semi_len0=8, semi_len1=13, intensity=2)
        assert not (test0.signal.data == test1.signal.data).all()

    def test_very_large_radius_no_cover(self):
        test = mdtd.MakeTestData(size_x=120, size_y=100)
        test.add_ring_ellipse(x0=50, y0=50, semi_len0=200, semi_len1=159, intensity=2)
        assert not test.signal.data.any()

    def test_lw_r(self):
        test0 = mdtd.MakeTestData(size_x=120, size_y=100)
        test0.add_ring_ellipse(x0=50, y0=50, semi_len0=20, semi_len1=30)
        test1 = mdtd.MakeTestData(size_x=120, size_y=100)
        test1.add_ring_ellipse(x0=50, y0=50, semi_len0=20, semi_len1=30, lw_r=3)
        assert test0.signal.data.sum() < test1.signal.data.sum()

    def test_rotation(self):
        test0 = mdtd.MakeTestData(size_x=120, size_y=100)
        test0.add_ring_ellipse(x0=50, y0=50, semi_len0=20, semi_len1=30)
        test1 = mdtd.MakeTestData(size_x=120, size_y=100)
        test1.add_ring_ellipse(x0=50, y0=50, semi_len0=20, semi_len1=30, rotation=1)
        assert not (test0.signal.data == test1.signal.data).all()

    def test_repr(self):
        test = mdtd.MakeTestData()
        test.add_ring_ellipse(x0=70)
        repr_string = test.z_list[0].__repr__()
        assert str(70) in repr_string


class TestGenerate4dData:
    def test_simple0(self):
        mdtd.generate_4d_data()

    def test_all_arguments(self):
        s = mdtd.generate_4d_data(
            probe_size_x=10,
            probe_size_y=10,
            image_size_x=50,
            image_size_y=50,
            disk_x=20,
            disk_y=20,
            disk_r=5,
            disk_I=30,
            ring_x=None,
            ring_e_x=None,
            blur=True,
            blur_sigma=1,
            downscale=True,
            add_noise=True,
            noise_amplitude=2,
        )
        assert s.axes_manager.shape == (10, 10, 50, 50)

    def test_different_size(self):
        s = mdtd.generate_4d_data(
            probe_size_x=5,
            probe_size_y=7,
            ring_x=None,
            ring_e_x=None,
            image_size_x=30,
            image_size_y=50,
        )
        ax = s.axes_manager
        assert ax.navigation_dimension == 2
        assert ax.signal_dimension == 2
        assert ax.navigation_shape == (5, 7)
        assert ax.signal_shape == (30, 50)

    def test_disk_outside_image(self):
        s = mdtd.generate_4d_data(
            probe_size_x=6,
            probe_size_y=4,
            image_size_x=40,
            image_size_y=40,
            ring_x=None,
            ring_e_x=None,
            disk_x=1000,
            disk_y=1000,
            disk_r=5,
        )
        assert (s.data == 0).all()

    def test_disk_cover_whole_image(self):
        s = mdtd.generate_4d_data(
            probe_size_x=6,
            probe_size_y=4,
            image_size_x=20,
            image_size_y=20,
            ring_x=None,
            ring_e_x=None,
            disk_x=10,
            disk_y=10,
            disk_r=40,
            disk_I=50,
            blur=False,
            downscale=False,
        )
        assert (s.data == 50).all()

    def test_disk_position_array(self):
        ps_x, ps_y, intensity = 4, 7, 30
        disk_x = np.random.randint(5, 35, size=(ps_y, ps_x))
        disk_y = np.random.randint(5, 45, size=(ps_y, ps_x))
        s = mdtd.generate_4d_data(
            probe_size_x=ps_x,
            probe_size_y=ps_y,
            image_size_x=40,
            image_size_y=50,
            ring_x=None,
            ring_e_x=None,
            disk_x=disk_x,
            disk_y=disk_y,
            disk_r=1,
            disk_I=intensity,
            blur=False,
            downscale=False,
        )
        for x in range(ps_x):
            for y in range(ps_y):
                cX, cY = disk_x[y, x], disk_y[y, x]
                sl = np.s_[cY - 1 : cY + 2, cX - 1 : cX + 2]
                im = s.inav[x, y].data[:]
                assert (im[sl] == intensity).all()
                im[sl] = 0
                assert not im.any()

    def test_disk_ring_outside_image(self):
        s = mdtd.generate_4d_data(
            probe_size_x=6,
            probe_size_y=4,
            image_size_x=40,
            image_size_y=40,
            disk_x=1000,
            disk_y=1000,
            disk_r=5,
            ring_x=1000,
            ring_y=1000,
            ring_r=10,
            ring_e_x=None,
        )
        assert (s.data == 0).all()

    def test_ring_center(self):
        x, y = 40, 51
        s = mdtd.generate_4d_data(
            probe_size_x=4,
            probe_size_y=5,
            image_size_x=120,
            image_size_y=100,
            disk_x=x,
            disk_y=y,
            disk_r=10,
            disk_I=0,
            ring_x=x,
            ring_y=y,
            ring_r=30,
            ring_I=5,
            ring_e_x=None,
            blur=False,
            downscale=False,
        )
        s_com = s.center_of_mass()
        assert (s_com.inav[0].data == x).all()
        assert (s_com.inav[1].data == y).all()

    def test_ring_ellipse_center(self):
        x, y = 40, 51
        s = mdtd.generate_4d_data(
            probe_size_x=4,
            probe_size_y=5,
            image_size_x=120,
            image_size_y=100,
            disk_x=None,
            ring_x=None,
            ring_e_x=x,
            ring_e_y=y,
            blur=False,
            downscale=False,
        )
        s_com = s.center_of_mass()
        assert (s_com.inav[0].data == x).all()
        assert (s_com.inav[1].data == y).all()

    def test_input_numpy_array(self):
        size = (20, 10)
        disk_x = np.random.randint(5, 35, size=size)
        disk_y = np.random.randint(5, 45, size=size)
        disk_r = np.random.randint(5, 9, size=size)
        disk_I = np.random.randint(50, 100, size=size)
        ring_x = np.random.randint(5, 35, size=size)
        ring_y = np.random.randint(5, 45, size=size)
        ring_r = np.random.randint(10, 15, size=size)
        ring_I = np.random.randint(1, 30, size=size)
        ring_lw = np.random.randint(1, 5, size=size)
        ring_e_x = np.random.randint(20, 30, size)
        ring_e_y = np.random.randint(20, 30, size)
        ring_e_semi_len0 = np.random.randint(10, 20, size)
        ring_e_semi_len1 = np.random.randint(10, 20, size)
        ring_e_r = np.random.random(size) * np.pi
        ring_e_lw = np.random.randint(1, 3, size)

        mdtd.generate_4d_data(
            probe_size_x=10,
            probe_size_y=20,
            image_size_x=40,
            image_size_y=50,
            disk_x=disk_x,
            disk_y=disk_y,
            disk_I=disk_I,
            disk_r=disk_r,
            ring_x=ring_x,
            ring_y=ring_y,
            ring_r=ring_r,
            ring_I=ring_I,
            ring_lw=ring_lw,
            ring_e_x=ring_e_x,
            ring_e_y=ring_e_y,
            ring_e_semi_len0=ring_e_semi_len0,
            ring_e_semi_len1=ring_e_semi_len1,
            ring_e_r=ring_e_r,
            ring_e_lw=ring_e_lw,
        )


class TestGetEllipticalRing:
    def test_simple(self):
        s = Signal2D(np.zeros((110, 200)))
        x, y, semi_len0, semi_len1, rotation = 60, 70, 12, 9, 0.2
        xx, yy = np.meshgrid(s.axes_manager[0].axis, s.axes_manager[1].axis)
        ellipse_image = mdtd._get_elliptical_ring(
            xx, yy, x, y, semi_len0, semi_len1, rotation, 3
        )
        assert ellipse_image.any()

    def test_cover_no_signal1(self):
        s = Signal2D(np.zeros((50, 56)))
        x, y, semi_len0, semi_len1, rotation = 523, 620, 20, 30, 2.0
        xx, yy = np.meshgrid(s.axes_manager[0].axis, s.axes_manager[1].axis)
        ellipse_image = mdtd._get_elliptical_ring(
            xx, yy, x, y, semi_len0, semi_len1, rotation
        )
        assert not ellipse_image.any()

    def test_cover_no_signal2(self):
        s = Signal2D(np.zeros((50, 56)))
        x, y, semi_len0, semi_len1, rotation = 20, 20, 90, 130, 2.0
        xx, yy = np.meshgrid(s.axes_manager[0].axis, s.axes_manager[1].axis)
        ellipse_image = mdtd._get_elliptical_ring(
            xx, yy, x, y, semi_len0, semi_len1, rotation
        )
        assert not ellipse_image.any()


class TestGetEllipticalMask:
    def test_simple(self):
        s = Signal2D(np.zeros((110, 200)))
        xx, yy = np.meshgrid(s.axes_manager[0].axis, s.axes_manager[1].axis)
        x, y, semi_len0, semi_len1, rotation = 60, 70, 12, 9, 0.2
        ellipse_image = mdtd._get_elliptical_disk(
            xx, yy, x, y, semi_len0, semi_len1, rotation
        )
        assert s.data.shape == ellipse_image.shape

    def test_cover_all_signal(self):
        s = Signal2D(np.zeros((50, 56)))
        xx, yy = np.meshgrid(s.axes_manager[0].axis, s.axes_manager[1].axis)
        x, y, semi_len0, semi_len1, rotation = 23, 20, 200, 200, 0.0
        ellipse_image = mdtd._get_elliptical_disk(
            xx, yy, x, y, semi_len0, semi_len1, rotation
        )
        assert s.data.shape == ellipse_image.shape
        assert ellipse_image.all()

    def test_cover_no_signal(self):
        s = Signal2D(np.zeros((50, 56)))
        xx, yy = np.meshgrid(s.axes_manager[0].axis, s.axes_manager[1].axis)
        x, y, semi_len0, semi_len1, rotation = 523, 620, 20, 30, 2.0
        ellipse_image = mdtd._get_elliptical_disk(
            xx, yy, x, y, semi_len0, semi_len1, rotation
        )
        assert not ellipse_image.any()


class TestMake4dPeakArrayTestData:
    def test_simple(self):
        xf, yf = np.ones((2, 3)), np.ones((2, 3))
        semi0, semi1, rot = np.ones((2, 3)), np.ones((2, 3)), np.ones((2, 3))
        peak_array = mdtd._make_4d_peak_array_test_data(xf, yf, semi0, semi1, rot)
        assert peak_array.shape == xf.shape

    def test_xf_yf(self):
        xf = np.random.randint(10, 20, size=(2, 3))
        yf = np.random.randint(39, 60, size=(2, 3))
        semi0, semi1, rot = np.ones((2, 3)), np.ones((2, 3)), np.ones((2, 3))
        peak_array = mdtd._make_4d_peak_array_test_data(
            xf, yf, semi0, semi1, rot, nt=1000
        )
        for iy, ix in np.ndindex(xf.shape):
            x_pos_mean = peak_array[iy, ix][:, 1].mean()
            assert approx(x_pos_mean, abs=0.01) == xf[iy, ix]
            y_pos_mean = peak_array[iy, ix][:, 0].mean()
            assert approx(y_pos_mean, abs=0.01) == yf[iy, ix]

    def test_semi_lengths(self):
        semi0 = np.random.randint(10, 20, size=(2, 3))
        semi1 = np.random.randint(10, 20, size=(2, 3))
        xf, yf, rot = np.ones((2, 3)) * 10, np.ones((2, 3)), np.zeros((2, 3))
        peak_array0 = mdtd._make_4d_peak_array_test_data(
            xf, yf, semi0, semi0, rot, nt=1000
        )
        peak_array1 = mdtd._make_4d_peak_array_test_data(
            xf, yf, semi1, semi1, rot, nt=1000
        )
        for iy, ix in np.ndindex(xf.shape):
            semi0_max = semi0[iy, ix] + xf[iy, ix]
            assert approx(peak_array0[iy, ix][:, 1].max()) == semi0_max
            semi1_max = semi1[iy, ix] + yf[iy, ix]
            assert approx(peak_array1[iy, ix][:, 0].max()) == semi1_max

    def test_rotation(self):
        rot0, rot1 = np.zeros((2, 3)), np.ones((2, 3)) * np.pi / 2
        xf, yf = np.ones((2, 3)), np.ones((2, 3))
        semi0, semi1 = np.ones((2, 3)) * 5, np.ones((2, 3))
        pa0 = mdtd._make_4d_peak_array_test_data(xf, yf, semi0, semi1, rot0, nt=1000)
        pa1 = mdtd._make_4d_peak_array_test_data(xf, yf, semi0, semi1, rot1, nt=1000)
        assert not np.array_equal(pa0, pa1)

    def test_nt(self):
        p = np.ones((2, 3))
        nt0, nt1 = 100, 200
        pa0 = mdtd._make_4d_peak_array_test_data(p, p, p, p, p, nt=nt0)
        pa1 = mdtd._make_4d_peak_array_test_data(p, p, p, p, p, nt=nt1)
        assert len(pa0[0, 0][:, 0]) == nt0
        assert len(pa1[0, 0][:, 0]) == nt1


class TestDiffractionTestImage:
    def test_simple(self):
        mdtd.DiffractionTestImage()

    def test_repr(self):
        di = mdtd.DiffractionTestImage(image_y=111)
        repr_string = di.__repr__()
        assert str(111) in repr_string

    def test_plot(self):
        di = mdtd.DiffractionTestImage()
        di.plot()

    def test_get_signal(self):
        di = mdtd.DiffractionTestImage()
        s = di.get_signal()
        assert hasattr(s, "plot")

    def test_copy(self):
        di0 = mdtd.DiffractionTestImage(intensity_noise=False)
        di0.add_cubic_disks(20, 20)
        di1 = di0.copy()
        data0 = di0.get_diffraction_test_image()
        data1 = di1.get_diffraction_test_image()
        assert (data0 == data1).all()

    def test_get_diffraction_test_image(self):
        di = mdtd.DiffractionTestImage()
        data = di.get_diffraction_test_image()
        assert hasattr(data, "__array__")

    def test_get_diffraction_test_image_dtype(self):
        di = mdtd.DiffractionTestImage()
        data = di.get_diffraction_test_image(dtype=np.float32)
        assert data.dtype == np.float32
        data = di.get_diffraction_test_image(dtype=np.float64)
        assert data.dtype == np.float64

    def test_add_disk(self):
        di = mdtd.DiffractionTestImage(
            diff_intensity_reduction=False, intensity_noise=False, blur=0
        )
        di.add_disk(10, 40, intensity=3.0)
        di.add_disk(128, 128, intensity=4.0)
        data = di.get_diffraction_test_image()
        assert data[40, 10] == 3.0
        assert data[128, 128] == 4.0

    def test_add_cubic_disks_n(self):
        di = mdtd.DiffractionTestImage(
            diff_intensity_reduction=False, intensity_noise=False, blur=0
        )
        di0 = di.copy()
        di1 = di.copy()
        di0.add_cubic_disks(20, 20, n=1)
        di1.add_cubic_disks(20, 20, n=2)
        data0 = di0.get_diffraction_test_image()
        data1 = di1.get_diffraction_test_image()
        assert data0.sum() < data1.sum()

    def test_add_cubic_disks_intensity(self):
        di = mdtd.DiffractionTestImage(
            diff_intensity_reduction=False,
            intensity_noise=False,
            blur=0,
            image_x=100,
            image_y=100,
        )
        di = di.copy()
        di.add_cubic_disks(20, 20, intensity=25.0, n=1)
        data = di.get_diffraction_test_image()
        assert data[30, 30] == 25.0

    def test_add_background_lorentz(self):
        di = mdtd.DiffractionTestImage(
            diff_intensity_reduction=False, intensity_noise=False, blur=0
        )
        data0 = di.get_diffraction_test_image()
        di.add_background_lorentz(intensity=10.0, width=5.0)
        data1 = di.get_diffraction_test_image()
        assert data0[50, 50] == 0.0
        assert data1[50, 50] != 0.0
        di.add_background_lorentz(intensity=10.0, width=20.0)
        data2 = di.get_diffraction_test_image()
        assert data2[50, 50] != data1[50, 50]
        di.add_background_lorentz(intensity=90.0, width=20.0)
        data3 = di.get_diffraction_test_image()
        assert data3[50, 50] > data2[50, 50]

    def test_image_x_y(self):
        x, y = 100, 100
        di = mdtd.DiffractionTestImage(image_x=x, image_y=y)
        data = di.get_diffraction_test_image()
        assert data.shape == (100, 100)
        s = di.get_signal()
        assert s.axes_manager.signal_shape == (100, 100)

    def test_disk_r(self):
        di = mdtd.DiffractionTestImage(
            image_x=50, image_y=50, disk_r=10, intensity_noise=False
        )
        di.add_disk(25, 25, intensity=2.0)
        data = di.get_diffraction_test_image()
        assert data[0, 0] == 0
        assert data[-1, 0] == 0
        assert data[0, -1] == 0
        assert data[-1, -1] == 0
        di.disk_r = 40
        data = di.get_diffraction_test_image()
        assert data[0, 0] == 2.0

    def test_disk_bad_input(self):
        di = mdtd.DiffractionTestImage()
        with pytest.raises(ValueError):
            di.add_disk(1.5, 5)
        with pytest.raises(ValueError):
            di.add_disk(5, 1.5)

    def test_rotation(self):
        di = mdtd.DiffractionTestImage(
            diff_intensity_reduction=False, intensity_noise=False, blur=0
        )
        di.add_disk(10, 10, intensity=2.0)
        data0 = di.get_diffraction_test_image()
        di.rotation = 180
        data1 = di.get_diffraction_test_image()
        assert data0[10, 10] == 2.0
        assert data1[-10, -10] == 2.0

    def test_blur(self):
        di = mdtd.DiffractionTestImage(
            diff_intensity_reduction=False, intensity_noise=False, blur=0
        )
        di.add_disk(10, 10, intensity=2.0)
        data0 = di.get_diffraction_test_image()
        di.blur = 3.0
        data1 = di.get_diffraction_test_image()
        assert data0[10, 10] == 2.0
        assert data1[10, 10] != 2.0

    def test_diff_intensity_reduction(self):
        di = mdtd.DiffractionTestImage(
            diff_intensity_reduction=False, intensity_noise=False, blur=0
        )
        di.add_disk(10, 10, intensity=2.0)
        data0 = di.get_diffraction_test_image()
        di.diff_intensity_reduction = 1.0
        data1 = di.get_diffraction_test_image()
        assert data0[10, 10] == 2.0
        assert data1[10, 10] != 2.0


class TestDiffractionTestDataset:
    def test_simple(self):
        mdtd.DiffractionTestDataset()

    def test_repr(self):
        dtd = mdtd.DiffractionTestDataset(probe_x=21)
        repr_string = dtd.__repr__()
        assert str(21) in repr_string

    def test_dataset_size(self):
        dtd = mdtd.DiffractionTestDataset(
            probe_x=5, probe_y=15, detector_x=101, detector_y=200
        )
        assert dtd.data.shape == (5, 15, 101, 200)

    def test_plot(self):
        dtd = mdtd.DiffractionTestDataset()
        dtd.plot()

    def test_get_signal(self):
        dtd = mdtd.DiffractionTestDataset()
        s = dtd.get_signal()
        assert hasattr(s, "plot")

    def test_add_diffraction_image(self):
        dtd = mdtd.DiffractionTestDataset(noise=False)
        di = mdtd.DiffractionTestImage(intensity_noise=False)
        di.add_cubic_disks(20, 20, n=1)
        data = di.get_diffraction_test_image()
        dtd.add_diffraction_image(di)
        for ix, iy in np.ndindex(dtd.data.shape[:2]):
            assert (dtd.data[ix, iy] == data).all()
