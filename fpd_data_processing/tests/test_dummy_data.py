import unittest
import fpd_data_processing.dummy_data as dd


class test_dummy_data_module(unittest.TestCase):

    def test_simple_disk_shift(self):
        s = dd.get_disk_shift_simple_test_signal()
        s.plot()

    def test_simple_holz_signal(self):
        s = dd.get_holz_simple_test_signal()
        s.plot()

    def test_single_ring_diffraction_signal(self):
        s = dd.get_single_ring_diffraction_signal()
        s.plot()

    def test_get_simple_dpc_signal(self):
        s = dd.get_simple_dpc_signal()
        s.plot()

    def test_get_holz_heterostructure_test_signal(self):
        s = dd.get_holz_heterostructure_test_signal()
        s.plot()

    def test_get_stripe_pattern_dpc_signal(self):
        s = dd.get_stripe_pattern_dpc_signal()
        s.plot()

    def test_get_square_dpc_signal(self):
        s = dd.get_square_dpc_signal()
        s.plot()
