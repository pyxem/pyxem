import os
import unittest
import numpy as np
from tempfile import TemporaryDirectory
import fpd_data_processing.api as fp


my_path = os.path.dirname(__file__)


class test_dpcsignal_io(unittest.TestCase):

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
                my_path, "test_data", "dpcsignal1d_test.hdf5")
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

    def test_load_fpd_signal_lazy(self):
        # Dataset has known size (2, 2, 256, 256)
        filename = os.path.join(
                my_path, "test_data", "fpd_file_test.hdf5")
        s = fp.load_fpd_signal(filename, lazy=True)
        self.assertEqual(s.axes_manager.shape, (2, 2, 256, 256))
