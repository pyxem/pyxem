import numpy as np
from hyperspy.signals import Signal2D
import fpd_data_processing.make_diffraction_test_data as mdtd


class TestMakeDiffractionTestData:

    def test_init(self):
        mdtd.MakeTestData(default=True)
        mdtd.MakeTestData(size_x=1, size_y=10, scale=0.05)
        mdtd.MakeTestData(
                size_x=200, size_y=100, scale=2, default=False, blur=False,
                blur_sigma=3., downscale=False)

    def test_zero_signal(self):
        test_data_1 = mdtd.MakeTestData(default=False)
        assert (test_data_1.signal.data == 0.).all()

        test_data_2 = mdtd.MakeTestData(default=True)
        test_data_2.set_signal_zero()
        assert (test_data_2.signal.data == 0.).all()


class TestMakeDiffractionTestDataDisks:

    def test_simple_disks(self):
        test0 = mdtd.MakeTestData(
                size_x=100, size_y=100,
                default=False, blur=False, downscale=False)
        test0.add_disk(50, 50, 1, 20)

        assert (test0.signal.isig[49:52, 49:52].data == 20).all()
        test0.signal.data[49:52, 49:52] = 0
        assert not test0.signal.data.any()

        test1 = mdtd.MakeTestData(
                size_x=100, size_y=100,
                default=False, blur=False, downscale=False)
        test1.add_disk(20, 40, 2, 30)
        assert (test1.signal.isig[18:23, 38:43].data == 30).all()
        test1.signal.data[38:43, 18:23] = 0
        assert not test1.signal.data.any()

        test2 = mdtd.MakeTestData(
                size_x=80, size_y=120,
                default=False, blur=False, downscale=False)
        test2.add_disk(50, 40, 1, 10)
        test2.add_disk(20, 30, 2, 30)
        assert (test2.signal.isig[49:52, 39:42].data == 10).all()
        assert (test2.signal.isig[19:22, 29:32].data == 30).all()
        test2.signal.data[39:42, 49:52] = 0
        test2.signal.data[28:33, 18:23] = 0
        assert not test2.signal.data.any()

        test3 = mdtd.MakeTestData(
                size_x=150, size_y=50,
                default=False, blur=False, downscale=False)
        test3.add_disk(200, 400, 2, 30)
        assert not test3.signal.data.any()

        test4 = mdtd.MakeTestData(
                size_x=150, size_y=50,
                default=False, blur=False, downscale=False)
        test4.add_disk(50, 50, 500, 500)
        assert (test4.signal.data == 500).all()

        test5 = mdtd.MakeTestData(
                size_x=100, size_y=200,
                default=False, blur=False, downscale=True)
        test5.add_disk(50, 50, 500, 10)
        assert (test5.signal.data == 10).all()

        test6 = mdtd.MakeTestData(
                size_x=100, size_y=200,
                default=False, blur=True, downscale=False)
        test6.add_disk(50, 30, 500, 10)
        test6_ref = np.full_like(test6.signal.data, 10.)
        np.testing.assert_allclose(test6.signal.data, test6_ref, rtol=1e-05)

    def test_large_disk(self):
        test_data_1 = mdtd.MakeTestData(
                size_x=10, size_y=10,
                scale=0.01, default=False)
        test_data_1.add_disk(x0=5, y0=5, r=20, intensity=100)
        assert (test_data_1.signal.data > 0.).all()


class TestMakeDiffractionTestDataRing:

    def test_ring_inner_radius(self):
        r, lw = 20, 2
        x0, y0, scale = 50, 50, 1
        test_data = mdtd.MakeTestData(
                size_x=100, size_y=100, default=False,
                blur=False, downscale=False)
        test_data.add_ring(x0=x0, y0=y0, r=r, intensity=10, lw_pix=lw)
        r_inner = 20 - 2.5*scale

        s_h0 = test_data.signal.isig[x0, :y0+1]
        s_h0_edge = s_h0.axes_manager[0].index2value(s_h0.data[::-1].argmax())
        r_h0 = s_h0_edge - 0.5*scale
        assert r_h0 == r_inner

        s_h1 = test_data.signal.isig[x0, y0:]
        s_h1_edge = s_h1.axes_manager[0].index2value(s_h1.data.argmax())
        r_h1 = s_h1_edge - 0.5*scale - 50
        assert r_h1 == r_inner

        s_h2 = test_data.signal.isig[:x0+1, y0]
        s_h2_edge = s_h2.axes_manager[0].index2value(s_h2.data[::-1].argmax())
        r_h2 = s_h2_edge - 0.5*scale
        assert r_h2 == r_inner

        s_h3 = test_data.signal.isig[x0:, y0]
        s_h3_edge = s_h3.axes_manager[0].index2value(s_h3.data.argmax())
        r_h3 = s_h3_edge - 0.5*scale - 50
        assert r_h3 == r_inner

    def test_ring_outer_radius(self):
        r, lw = 20, 1
        x0, y0, scale = 0, 0, 1
        ring_1 = mdtd.MakeTestData(
                size_x=100, size_y=100, default=False,
                blur=False, downscale=False)
        ring_1.add_ring(x0=x0, y0=y0, r=r, intensity=10, lw_pix=lw)
        s_h0 = ring_1.signal.isig[x0, :]
        s_h0_edge = s_h0.axes_manager[0].index2value(s_h0.data[::-1].argmax())
        r_h0 = s_h0.data.size - s_h0_edge - 0.5*scale
        r_out = r + lw + 0.5*scale
        assert r_h0 == r_out
        s_h1 = ring_1.signal.isig[:, y0]
        s_h1_edge = s_h1.axes_manager[0].index2value(s_h1.data[::-1].argmax())
        r_h1 = s_h1.data.size - s_h1_edge - 0.5*scale
        assert r_h1 == r_out

        r, lw = 20, 1
        x0, y0, scale = 100, 100, 1
        ring_2 = mdtd.MakeTestData(
                size_x=100, size_y=100, default=False,
                blur=False, downscale=False)
        ring_2.add_ring(x0=x0, y0=y0, r=r, intensity=10, lw_pix=lw)
        s_h2 = ring_2.signal.isig[-1, :]
        s_h2_edge = s_h2.axes_manager[0].index2value(s_h2.data.argmax())
        r_h2 = 100 - (s_h2_edge - 0.5*scale)
        r_out = r + lw + 0.5*scale
        assert r_h2 == r_out
        s_h3 = ring_2.signal.isig[:, -1]
        s_h3_edge = s_h3.axes_manager[0].index2value(s_h3.data.argmax())
        r_h3 = 100 - (s_h3_edge - 0.5*scale)
        assert r_h3 == r_out

    def test_ring_radius1(self):
        intensity = 20
        test = mdtd.MakeTestData(
                size_x=100, size_y=100,
                default=False, blur=False, downscale=False)
        test.add_ring(x0=50, y0=50, r=1, intensity=intensity, lw_pix=0)

        assert (test.signal.isig[50, 50].data == 0.).all()
        test.signal.data[50, 50] = intensity
        assert (test.signal.isig[49:52, 49:52].data == intensity).all()
        test.signal.data[49:52, 49:52] = 0.
        assert not test.signal.data.any()

    def test_ring_radius2(self):
        intensity = 10
        test = mdtd.MakeTestData(
                size_x=100, size_y=100,
                default=False, blur=False, downscale=False)
        test.add_ring(x0=50, y0=50, r=2, intensity=intensity, lw_pix=0)

        assert (test.signal.isig[49:52, 49:52].data == 0.).all()
        test.signal.data[49:52, 49:52] = intensity
        assert (test.signal.isig[48:53, 48:53].data == intensity).all()
        test.signal.data[48:53, 48:53] = 0.
        assert not test.signal.data.any()

    def test_ring_radius_outside_image(self):
        test = mdtd.MakeTestData(
                size_x=100, size_y=100,
                default=False, blur=False, downscale=False)
        test.add_ring(x0=50, y0=50, r=300, intensity=19, lw_pix=1)
        assert not test.signal.data.any()

    def test_ring_rectangle_image(self):
        intensity = 10
        test = mdtd.MakeTestData(
                size_x=100, size_y=50,
                default=False, blur=False, downscale=False)
        test.add_ring(x0=50, y0=25, r=1, intensity=intensity, lw_pix=0)

        assert (test.signal.isig[50, 25].data == 0.).all()
        test.signal.data[25, 50] = intensity
        assert (test.signal.isig[49:52, 24:27].data == intensity).all()
        test.signal.data[24:27, 49:52] = 0.
        assert not test.signal.data.any()

    def test_ring_cover_whole_image(self):
        intensity = 10.
        test = mdtd.MakeTestData(
                size_x=50, size_y=100,
                default=False, blur=False, downscale=False)
        test.add_ring(x0=25, y0=200, r=150, intensity=intensity, lw_pix=100)
        assert (test.signal.data == intensity).all()

    def test_ring_position(self):
        x0, y0, r = 40., 60., 20
        test = mdtd.MakeTestData(
                size_x=100, size_y=100,
                default=False, blur=False, downscale=False)
        test.add_ring(x0=x0, y0=y0, r=r, intensity=20, lw_pix=0)
        s_h0 = test.signal.isig[x0, 0.:y0]
        max_h0 = s_h0.axes_manager[0].index2value(s_h0.data.argmax())
        assert max_h0 == y0-r
        s_h1 = test.signal.isig[x0, y0:]
        max_h1 = s_h1.axes_manager[0].index2value(s_h1.data.argmax())
        assert max_h1 == y0+r

        s_v0 = test.signal.isig[0:x0, y0]
        max_v0 = s_v0.axes_manager[0].index2value(s_v0.data.argmax())
        assert max_v0 == x0-r
        s_v1 = test.signal.isig[x0:, y0]
        max_v1 = s_v1.axes_manager[0].index2value(s_v1.data.argmax())
        assert max_v1 == x0+r

    def test_ring_position_blur(self):
        x0, y0, r = 50, 50, 15
        test = mdtd.MakeTestData(
                size_x=100, size_y=100,
                default=False, blur=True, downscale=False)
        test.add_ring(x0=x0, y0=y0, r=r, intensity=20, lw_pix=1)
        s_h0 = test.signal.isig[x0, 0:y0]
        max_h0 = s_h0.axes_manager[0].index2value(s_h0.data.argmax())
        assert max_h0 == y0-r
        s_h1 = test.signal.isig[x0, y0:]
        max_h1 = s_h1.axes_manager[0].index2value(s_h1.data.argmax())
        assert max_h1 == y0+r

        s_v0 = test.signal.isig[0:x0, y0]
        max_v0 = s_v0.axes_manager[0].index2value(s_v0.data.argmax())
        assert max_v0 == x0-r
        s_v1 = test.signal.isig[x0:, y0]
        max_v1 = s_v1.axes_manager[0].index2value(s_v1.data.argmax())
        assert max_v1 == x0+r

    def test_ring_position_blur_lw(self):
        x0, y0, r = 50, 50, 15
        test = mdtd.MakeTestData(
                size_x=100, size_y=100,
                default=False, blur=True, downscale=False)
        test.add_ring(x0=x0, y0=y0, r=r, intensity=20, lw_pix=1)
        s_h0 = test.signal.isig[x0, 0:y0]
        max_h0 = s_h0.axes_manager[0].index2value(s_h0.data.argmax())
        assert max_h0 == y0-r
        s_h1 = test.signal.isig[x0, y0:]
        max_h1 = s_h1.axes_manager[0].index2value(s_h1.data.argmax())
        assert max_h1 == y0+r

        s_v0 = test.signal.isig[0:x0, y0]
        max_v0 = s_v0.axes_manager[0].index2value(s_v0.data.argmax())
        assert max_v0 == x0-r
        s_v1 = test.signal.isig[x0:, y0]
        max_v1 = s_v1.axes_manager[0].index2value(s_v1.data.argmax())
        assert max_v1 == x0+r

    def test_ring_position_downscale(self):
        x0, y0, r = 50, 50, 15
        test = mdtd.MakeTestData(
                size_x=100, size_y=100,
                default=False, blur=False, downscale=True)
        test.add_ring(x0=x0, y0=y0, r=r, intensity=20, lw_pix=0)
        s_h0 = test.signal.isig[x0, 0:y0]
        max_h0 = s_h0.axes_manager[0].index2value(s_h0.data.argmax())
        assert max_h0 == y0-r
        s_h1 = test.signal.isig[x0, y0:]
        max_h1 = s_h1.axes_manager[0].index2value(s_h1.data.argmax())
        assert max_h1 == y0+r

        s_v0 = test.signal.isig[0:x0, y0]
        max_v0 = s_v0.axes_manager[0].index2value(s_v0.data.argmax())
        assert max_v0 == x0-r
        s_v1 = test.signal.isig[x0:, y0]
        max_v1 = s_v1.axes_manager[0].index2value(s_v1.data.argmax())
        assert max_v1 == x0+r

    def test_ring_position_downscale_and_blur(self):
        x0, y0, r = 50, 50, 18
        test = mdtd.MakeTestData(
                size_x=100, size_y=100,
                default=False, blur=True, downscale=True)
        test.add_ring(x0=x0, y0=y0, r=r, intensity=20, lw_pix=1)
        s_h0 = test.signal.isig[x0, 0:y0]
        max_h0 = s_h0.axes_manager[0].index2value(s_h0.data.argmax())
        assert max_h0 == y0-r
        s_h1 = test.signal.isig[x0, y0:]
        max_h1 = s_h1.axes_manager[0].index2value(s_h1.data.argmax())
        assert max_h1 == y0+r

        s_v0 = test.signal.isig[0:x0, y0]
        max_v0 = s_v0.axes_manager[0].index2value(s_v0.data.argmax())
        assert max_v0 == x0-r
        s_v1 = test.signal.isig[x0:, y0]
        max_v1 = s_v1.axes_manager[0].index2value(s_v1.data.argmax())
        assert max_v1 == x0+r


class TestGenerate4dData:

    def test_simple0(self):
        mdtd.generate_4d_data()

    def test_all_arguments(self):
        s = mdtd.generate_4d_data(
                probe_size_x=10, probe_size_y=10,
                image_size_x=50, image_size_y=50,
                disk_x=20, disk_y=20, disk_r=5, disk_I=30,
                ring_x=None, blur=True, blur_sigma=1,
                downscale=True, add_noise=True,
                noise_amplitude=2)
        assert s.axes_manager.shape == (10, 10, 50, 50)

    def test_different_size(self):
        s = mdtd.generate_4d_data(
                probe_size_x=5, probe_size_y=7, ring_x=None,
                image_size_x=30, image_size_y=50)
        ax = s.axes_manager
        assert ax.navigation_dimension == 2
        assert ax.signal_dimension == 2
        assert ax.navigation_shape == (5, 7)
        assert ax.signal_shape == (30, 50)

    def test_disk_outside_image(self):
        s = mdtd.generate_4d_data(
                probe_size_x=6, probe_size_y=4,
                image_size_x=40, image_size_y=40,
                ring_x=None, disk_x=1000, disk_y=1000, disk_r=5)
        assert (s.data == 0).all()

    def test_disk_cover_whole_image(self):
        s = mdtd.generate_4d_data(
                probe_size_x=6, probe_size_y=4,
                image_size_x=20, image_size_y=20,
                ring_x=None, disk_x=10, disk_y=10, disk_r=40, disk_I=50,
                blur=False, downscale=False)
        assert (s.data == 50).all()

    def test_disk_position_array(self):
        ps_x, ps_y, intensity = 4, 7, 30
        disk_x = np.random.randint(5, 35, size=(ps_y, ps_x))
        disk_y = np.random.randint(5, 45, size=(ps_y, ps_x))
        s = mdtd.generate_4d_data(
                probe_size_x=ps_x, probe_size_y=ps_y,
                image_size_x=40, image_size_y=50,
                ring_x=None, disk_x=disk_x, disk_y=disk_y, disk_r=1,
                disk_I=intensity, blur=False, downscale=False)
        for x in range(ps_x):
            for y in range(ps_y):
                cX, cY = disk_x[y, x], disk_y[y, x]
                sl = np.s_[cY-1:cY+2, cX-1:cX+2]
                im = s.inav[x, y].data[:]
                assert (im[sl] == intensity).all()
                im[sl] = 0
                assert not im.any()

    def test_disk_ring_outside_image(self):
        s = mdtd.generate_4d_data(
                probe_size_x=6, probe_size_y=4,
                image_size_x=40, image_size_y=40,
                disk_x=1000, disk_y=1000, disk_r=5,
                ring_x=1000, ring_y=1000, ring_r=10)
        assert (s.data == 0).all()

    def test_ring_center(self):
        x, y = 40, 51
        s = mdtd.generate_4d_data(
                probe_size_x=4, probe_size_y=5,
                image_size_x=120, image_size_y=100,
                disk_x=x, disk_y=y, disk_r=10, disk_I=0,
                ring_x=x, ring_y=y, ring_r=30, ring_I=5,
                blur=False, downscale=False)
        s_com = s.center_of_mass()
        assert (s_com.inav[0].data == x).all()
        assert (s_com.inav[1].data == y).all()

    def test_input_numpy_array(self):
        disk_x = np.random.randint(5, 35, size=(20, 10))
        disk_y = np.random.randint(5, 45, size=(20, 10))
        disk_r = np.random.randint(5, 9, size=(20, 10))
        disk_I = np.random.randint(50, 100, size=(20, 10))
        ring_x = np.random.randint(5, 35, size=(20, 10))
        ring_y = np.random.randint(5, 45, size=(20, 10))
        ring_r = np.random.randint(10, 15, size=(20, 10))
        ring_I = np.random.randint(1, 30, size=(20, 10))
        ring_lw = np.random.randint(1, 5, size=(20, 10))
        mdtd.generate_4d_data(
                probe_size_x=10, probe_size_y=20,
                image_size_x=40, image_size_y=50,
                disk_x=disk_x, disk_y=disk_y, disk_I=disk_I, disk_r=disk_r,
                ring_x=ring_x, ring_y=ring_y, ring_r=ring_r,
                ring_I=ring_I, ring_lw=ring_lw)


class TestGetEllipticalRing:

    def test_simple(self):
        s = Signal2D(np.zeros((110, 200)))
        x, y, semi_len0, semi_len1, rotation = 60, 70, 12, 9, 0.2
        ellipse_image = mdtd._get_elliptical_ring(
                s, x, y, semi_len0, semi_len1, rotation, 3)
        assert ellipse_image.any()

    def test_cover_no_signal1(self):
        s = Signal2D(np.zeros((50, 56)))
        x, y, semi_len0, semi_len1, rotation = 523, 620, 20, 30, 2.0
        ellipse_image = mdtd._get_elliptical_ring(
                s, x, y, semi_len0, semi_len1, rotation)
        assert not ellipse_image.any()

    def test_cover_no_signal2(self):
        s = Signal2D(np.zeros((50, 56)))
        x, y, semi_len0, semi_len1, rotation = 20, 20, 90, 130, 2.0
        ellipse_image = mdtd._get_elliptical_ring(
                s, x, y, semi_len0, semi_len1, rotation)
        assert not ellipse_image.any()


class TestGetEllipticalMask:

    def test_simple(self):
        s = Signal2D(np.zeros((110, 200)))
        x, y, semi_len0, semi_len1, rotation = 60, 70, 12, 9, 0.2
        ellipse_image = mdtd._get_elliptical_mask(
                s, x, y, semi_len0, semi_len1, rotation)
        assert s.data.shape == ellipse_image.shape

    def test_cover_all_signal(self):
        s = Signal2D(np.zeros((50, 56)))
        x, y, semi_len0, semi_len1, rotation = 23, 20, 200, 200, 0.0
        ellipse_image = mdtd._get_elliptical_mask(
                s, x, y, semi_len0, semi_len1, rotation)
        assert s.data.shape == ellipse_image.shape
        assert ellipse_image.all()

    def test_cover_no_signal(self):
        s = Signal2D(np.zeros((50, 56)))
        x, y, semi_len0, semi_len1, rotation = 523, 620, 20, 30, 2.0
        ellipse_image = mdtd._get_elliptical_mask(
                s, x, y, semi_len0, semi_len1, rotation)
        assert not ellipse_image.any()
