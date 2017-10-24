import os
import unittest
import numpy as np
from tempfile import TemporaryDirectory
import fpd_data_processing.api as fp


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
