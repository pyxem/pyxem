import numpy as np
import pixstem.api as ps


class TestApi:

    def test_pixelated_stem(self):
        s = ps.PixelatedSTEM(np.ones((10, 5, 3, 6)))
        assert s.axes_manager.signal_shape == (6, 3)
        assert s.axes_manager.navigation_shape == (5, 10)

    def test_dpcbasesignal(self):
        s = ps.DPCBaseSignal(np.ones((2))).T
        assert s.axes_manager.signal_shape == ()
        assert s.axes_manager.navigation_shape == (2, )

    def test_dpcsignal1d(self):
        s = ps.DPCSignal1D(np.ones((2, 10)))
        assert s.axes_manager.signal_shape == (10, )
        assert s.axes_manager.navigation_shape == (2, )

    def test_dpcsignal2d(self):
        s = ps.DPCSignal2D(np.ones((2, 10, 15)))
        assert s.axes_manager.signal_shape == (15, 10)
        assert s.axes_manager.navigation_shape == (2, )
