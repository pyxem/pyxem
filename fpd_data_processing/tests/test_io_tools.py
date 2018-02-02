import os
import unittest
import numpy as np
from tempfile import TemporaryDirectory
import hyperspy.api as hs
import fpd_data_processing.api as fp
import fpd_data_processing.io_tools as it


my_path = os.path.dirname(__file__)


class test_dpcsignal_io(unittest.TestCase):

    def setUp(self):
        self.tmpdir = TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_load_basesignal(self):
        filename = os.path.join(
                my_path, "test_data", "dpcbasesignal_test.hdf5")
        fp.load_dpc_signal(filename)

    def test_load_signal1d(self):
        filename = os.path.join(
                my_path, "test_data", "dpcsignal1d_test.hdf5")
        fp.load_dpc_signal(filename)

    def test_load_signal2d(self):
        filename = os.path.join(
                my_path, "test_data", "dpcsignal2d_test.hdf5")
        fp.load_dpc_signal(filename)

    def test_load_signal2d_too_many_nav_dim(self):
        filename = os.path.join(
                my_path, "test_data", "dpcsignal2d_test_too_many_nav_dim.hdf5")
        with self.assertRaises(Exception):
            fp.load_dpc_signal(filename)

    def test_load_basesignal_too_many_signal_dim(self):
        filename = os.path.join(
                my_path,
                "test_data",
                "dpcbasesignal_test_too_many_signal_dims.hdf5")
        with self.assertRaises(NotImplementedError):
            fp.load_dpc_signal(filename)

    def test_retain_metadata(self):
        s = fp.DPCSignal2D(np.ones((2, 10, 5)))
        s.metadata.General.title = "test_data"
        filename = os.path.join(self.tmpdir.name, 'test_metadata.hspy')
        s.save(filename)
        s_load = fp.load_dpc_signal(filename)
        self.assertEqual(s_load.metadata.General.title, "test_data")

    def test_retain_axes_manager(self):
        s = fp.DPCSignal2D(np.ones((2, 10, 5)))
        s_sa0 = s.axes_manager.signal_axes[0]
        s_sa1 = s.axes_manager.signal_axes[1]
        s_sa0.offset, s_sa1.offset, s_sa0.scale, s_sa1.scale = 20, 10, 0.2, 0.3
        s_sa0.units, s_sa1.units, s_sa0.name, s_sa1.name = "a", "b", "e", "f"
        filename = os.path.join(self.tmpdir.name, 'test_axes_manager.hspy')
        s.save(filename)
        s_load = fp.load_dpc_signal(filename)
        self.assertEqual(s_load.axes_manager[1].offset, 20)
        self.assertEqual(s_load.axes_manager[2].offset, 10)
        self.assertEqual(s_load.axes_manager[1].scale, 0.2)
        self.assertEqual(s_load.axes_manager[2].scale, 0.3)
        self.assertEqual(s_load.axes_manager[1].units, "a")
        self.assertEqual(s_load.axes_manager[2].units, "b")
        self.assertEqual(s_load.axes_manager[1].name, "e")
        self.assertEqual(s_load.axes_manager[2].name, "f")


class test_pixelatedstem_signal_io(unittest.TestCase):

    def test_load_hspy_signal(self):
        # Has shape (2, 5, 4, 3)
        filename = os.path.join(
                my_path, "test_data", "pixelated_stem_test.hdf5")
        s = fp.load_fpd_signal(filename)
        self.assertEqual(s.axes_manager.shape, (2, 5, 4, 3))

    def test_load_hspy_signal_generated(self):
        shape = (7, 6, 3, 5)
        tmpdir = TemporaryDirectory()
        filename = os.path.join(
                tmpdir.name, "test.hdf5")
        s = fp.PixelatedSTEM(np.zeros(shape))
        s.save(filename)

        sl = fp.load_fpd_signal(filename, lazy=False)
        self.assertEqual(
                sl.axes_manager.shape,
                (shape[1], shape[0], shape[3], shape[2]))
        tmpdir.cleanup()

    def test_load_hspy_signal_generated_lazy(self):
        shape = (3, 5, 7, 9)
        tmpdir = TemporaryDirectory()
        filename = os.path.join(
                tmpdir.name, "test_lazy.hdf5")
        s = fp.PixelatedSTEM(np.zeros(shape))
        s.save(filename)

        sl = fp.load_fpd_signal(filename, lazy=True)
        self.assertEqual(
                sl.axes_manager.shape,
                (shape[1], shape[0], shape[3], shape[2]))
        tmpdir.cleanup()

    def test_load_fpd_signal(self):
        # Dataset has known size (2, 2, 256, 256)
        filename = os.path.join(
                my_path, "test_data", "fpd_file_test.hdf5")
        s = fp.load_fpd_signal(filename)
        self.assertEqual(s.axes_manager.shape, (2, 2, 256, 256))

        s = fp.load_fpd_signal(filename, lazy=True)
        self.assertEqual(s.axes_manager.shape, (2, 2, 256, 256))

    def test_navigation_signal(self):
        # Dataset has known size (2, 2, 256, 256)
        filename = os.path.join(
                my_path, "test_data", "fpd_file_test.hdf5")
        s = fp.load_fpd_signal(filename)
        self.assertEqual(s.axes_manager.shape, (2, 2, 256, 256))

        s_nav0 = hs.signals.Signal2D(np.zeros((2, 2)))
        s = fp.load_fpd_signal(filename, lazy=True, navigation_signal=s_nav0)
        s.plot()

        s_nav1 = hs.signals.Signal2D(np.zeros((2, 4)))
        with self.assertRaises(ValueError):
            s = fp.load_fpd_signal(
                    filename, lazy=True, navigation_signal=s_nav1)


class test_signal_to_pixelated_stem(unittest.TestCase):

    def test_conserve_signal_axes_metadata(self):
        x_nav, y_nav, x_sig, y_sig = 9, 8, 5, 7
        x_nav_scale, y_nav_scale, x_sig_scale, y_sig_scale = 0.5, 0.2, 1.2, 3.2
        x_nav_off, y_nav_off, x_sig_off, y_sig_off = 30, 12, 76, 32
        x_nav_name, y_nav_name, x_sig_name, y_sig_name = "nX", "nY", "sX", "sY"
        x_nav_unit, y_nav_unit, x_sig_unit, y_sig_unit = "u1", "u2", "u3", "u4"
        title = "test_title"
        data = np.random.random((y_nav, x_nav, y_sig, x_sig))

        s = hs.signals.Signal2D(data)
        s.metadata.General.title = title
        am = s.axes_manager
        am[0].scale, am[0].offset = x_nav_scale, x_nav_off
        am[1].scale, am[1].offset = y_nav_scale, y_nav_off
        am[2].scale, am[2].offset = x_sig_scale, x_sig_off
        am[3].scale, am[3].offset = y_sig_scale, y_sig_off
        am[0].name, am[0].units = x_nav_name, x_nav_unit
        am[1].name, am[1].units = y_nav_name, y_nav_unit
        am[2].name, am[2].units = x_sig_name, x_sig_unit
        am[3].name, am[3].units = y_sig_name, y_sig_unit

        s1 = it.signal_to_pixelated_stem(s)

        self.assertTrue((data == s1.data).all())
        self.assertEqual(s1.metadata.General.title, title)
        self.assertEqual(s1.axes_manager.shape, (x_nav, y_nav, x_sig, y_sig))
        self.assertEqual(s1.axes_manager[0].scale, x_nav_scale)
        self.assertEqual(s1.axes_manager[1].scale, y_nav_scale)
        self.assertEqual(s1.axes_manager[2].scale, x_sig_scale)
        self.assertEqual(s1.axes_manager[3].scale, y_sig_scale)

        self.assertEqual(s1.axes_manager[0].offset, x_nav_off)
        self.assertEqual(s1.axes_manager[1].offset, y_nav_off)
        self.assertEqual(s1.axes_manager[2].offset, x_sig_off)
        self.assertEqual(s1.axes_manager[3].offset, y_sig_off)

        self.assertEqual(s1.axes_manager[0].name, x_nav_name)
        self.assertEqual(s1.axes_manager[1].name, y_nav_name)
        self.assertEqual(s1.axes_manager[2].name, x_sig_name)
        self.assertEqual(s1.axes_manager[3].name, y_sig_name)

        self.assertEqual(s1.axes_manager[0].units, x_nav_unit)
        self.assertEqual(s1.axes_manager[1].units, y_nav_unit)
        self.assertEqual(s1.axes_manager[2].units, x_sig_unit)
        self.assertEqual(s1.axes_manager[3].units, y_sig_unit)
