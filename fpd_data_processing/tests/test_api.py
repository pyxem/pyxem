import unittest
import numpy as np
import fpd_data_processing.api as fdp

class test_api(unittest.TestCase):

    def test_pixelated_stem(self):
        s = fdp.PixelatedSTEM(np.ones((10, 5, 3, 6)))
        self.assertEqual(s.axes_manager.signal_shape, (6, 3))
        self.assertEqual(s.axes_manager.navigation_shape, (5, 10))

    def test_dpcbasesignal(self):
        s = fdp.DPCBaseSignal(np.ones((2))).T
        self.assertEqual(s.axes_manager.signal_shape, ())
        self.assertEqual(s.axes_manager.navigation_shape, (2, ))

    def test_dpcsignal1d(self):
        s = fdp.DPCSignal1D(np.ones((2, 10)))
        self.assertEqual(s.axes_manager.signal_shape, (10, ))
        self.assertEqual(s.axes_manager.navigation_shape, (2, ))

    def test_dpcsignal2d(self):
        s = fdp.DPCSignal2D(np.ones((2, 10, 15)))
        self.assertEqual(s.axes_manager.signal_shape, (15, 10))
        self.assertEqual(s.axes_manager.navigation_shape, (2, ))
