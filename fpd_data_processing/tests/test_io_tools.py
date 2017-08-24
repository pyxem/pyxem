import os
import unittest
import fpd_data_processing.api as fdp

my_path = os.path.dirname(__file__)


class test_dpcsignal_io(unittest.TestCase):

    def test_load_basesignal(self):
        filename = os.path.join(
                my_path, "test_data", "dpcbasesignal_test.hdf5")
        s = fdp.load_dpc_signal(filename)

    def test_load_signal1d(self):
        filename = os.path.join(
                my_path, "test_data", "dpcsignal1d_test.hdf5")
        s = fdp.load_dpc_signal(filename)

    def test_load_signal2d(self):
        filename = os.path.join(
                my_path, "test_data", "dpcsignal1d_test.hdf5")
        s = fdp.load_dpc_signal(filename)

    def test_load_signal2d_too_many_nav_dim(self):
        filename = os.path.join(
                my_path, "test_data", "dpcsignal2d_test_too_many_nav_dim.hdf5")
        with self.assertRaises(Exception):
            s = fdp.load_dpc_signal(filename)

    def test_load_basesignal_too_many_signal_dim(self):
        filename = os.path.join(
                my_path,
                "test_data",
                "dpcbasesignal_test_too_many_signal_dims.hdf5")
        with self.assertRaises(NotImplementedError):
            s = fdp.load_dpc_signal(filename)
