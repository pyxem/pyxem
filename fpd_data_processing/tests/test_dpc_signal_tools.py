import numpy as np
import fpd_data_processing.pixelated_stem_tools as pst


class TestGetRgbPhaseMagnitudeArray:

    def test_simple(self):
        phase = np.zeros((50, 50))
        magnitude = np.zeros((50, 50))
        rgb_array = pst._get_rgb_phase_magnitude_array(phase, magnitude)
        assert (rgb_array == 0.).all()

    def test_magnitude_zero(self):
        phase = np.random.random((50, 50))
        magnitude = np.zeros((50, 50))
        rgb_array = pst._get_rgb_phase_magnitude_array(phase, magnitude)
        assert (rgb_array == 0.).all()

    def test_all_same(self):
        phase = np.ones((50, 50))
        magnitude = np.ones((50, 50))
        rgb_array = pst._get_rgb_phase_magnitude_array(phase, magnitude)
        assert (rgb_array == rgb_array[0][0]).all()


class TestGetRgbPhaseArray:

    def test_all_same0(self):
        phase = np.zeros((50, 50))
        rgb_array = pst._get_rgb_phase_array(phase)
        assert (rgb_array == rgb_array[0][0]).all()

    def test_all_same1(self):
        phase = np.ones((50, 50))
        rgb_array = pst._get_rgb_phase_array(phase)
        assert (rgb_array == rgb_array[0][0]).all()


class TestFindPhase:

    def test_simple(self):
        phase = np.zeros((50, 50))
        phase0 = pst._find_phase(phase)
        assert (phase0 == 0.).all()

    def test_rotation(self):
        phase = np.zeros((50, 50))
        phase0 = pst._find_phase(phase, rotation=90)
        assert (phase0 == np.pi/2).all()
        phase1 = pst._find_phase(phase, rotation=45)
        assert (phase1 == np.pi/4).all()
        phase2 = pst._find_phase(phase, rotation=180)
        assert (phase2 == np.pi).all()
        phase3 = pst._find_phase(phase, rotation=360)
        assert (phase3 == 0).all()
        phase4 = pst._find_phase(phase, rotation=-90)
        assert (phase4 == 3*np.pi/2).all()

    def test_max_phase(self):
        phase = (np.ones((50, 50))*np.pi*0.5) + np.pi
        phase0 = pst._find_phase(phase, max_phase=np.pi)
        assert (phase0 == np.pi/2).all()


class TestMakeBivariateHistogram:

    def test_single_x(self):
        size = 100
        x, y = np.ones(size), np.zeros(size)
        s = pst._make_bivariate_histogram(x, y)
        hist_iX = s.axes_manager[0].value2index(1.)
        hist_iY = s.axes_manager[1].value2index(0.)
        assert s.data[hist_iY, hist_iX] == size
        s.data[hist_iY, hist_iX] = 0
        assert not s.data.any()

    def test_single_negative_x(self):
        size = 100
        x, y = -np.ones(size), np.zeros(size)
        s = pst._make_bivariate_histogram(x, y)
        hist_iX = s.axes_manager[0].value2index(-1)
        hist_iY = s.axes_manager[1].value2index(0)
        assert s.data[hist_iY, hist_iX] == size
        s.data[hist_iY, hist_iX] = 0
        assert not s.data.any()

    def test_single_negative_x_y(self):
        size = 100
        x, y = -np.ones(size), np.ones(size)
        s = pst._make_bivariate_histogram(x, y)
        hist_iX = s.axes_manager[0].value2index(-1)
        hist_iY = s.axes_manager[1].value2index(1)
        assert s.data[hist_iY, hist_iX] == size
        s.data[hist_iY, hist_iX] = 0
        assert not s.data.any()
