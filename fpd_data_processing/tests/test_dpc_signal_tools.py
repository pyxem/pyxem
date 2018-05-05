import pytest
import numpy as np
import hyperspy.api as hs
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


class TestGetSignalMeanPositionAndValue:

    def test_simple(self):
        s = hs.signals.Signal2D(np.zeros((10, 10)))
        # s has the values 0 to 9, so middle position will be 4.5
        output = pst._get_signal_mean_position_and_value(s)
        assert len(output) == 3
        assert output[0] == 4.5  # x-position
        assert output[1] == 4.5  # y-position
        assert output[2] == 0.0  # Mean value

    def test_mean_value(self):
        s = hs.signals.Signal2D(np.ones((10, 10))*9)
        output = pst._get_signal_mean_position_and_value(s)
        assert output[2] == 9.0

    def test_non_square_shape(self):
        s = hs.signals.Signal2D(np.zeros((10, 5)))
        # s gets the shape 5, 10. Due to the axes being reversed
        output = pst._get_signal_mean_position_and_value(s)
        assert output[0] == 2.
        assert output[1] == 4.5

    def test_wrong_input_dimensions(self):
        s = hs.signals.Signal2D(np.ones((2, 10, 10)))
        with pytest.raises(ValueError):
            pst._get_signal_mean_position_and_value(s)
        s = hs.signals.Signal2D(np.ones((2, 2, 10, 10)))
        with pytest.raises(ValueError):
            pst._get_signal_mean_position_and_value(s)

    def test_origin(self):
        s = hs.signals.Signal2D(np.zeros((20, 10)))
        output = pst._get_signal_mean_position_and_value(s)
        assert output == (4.5, 9.5, 0)
        s.axes_manager[0].offset = 10
        output = pst._get_signal_mean_position_and_value(s)
        assert output == (14.5, 9.5, 0)
        s.axes_manager[1].offset = 6
        output = pst._get_signal_mean_position_and_value(s)
        assert output == (14.5, 15.5, 0)
        s.axes_manager[1].offset = -5
        output = pst._get_signal_mean_position_and_value(s)
        assert output == (14.5, 4.5, 0)
        s.axes_manager[1].offset = -50
        output = pst._get_signal_mean_position_and_value(s)
        assert output == (14.5, -40.5, 0)

    def test_scale(self):
        s = hs.signals.Signal2D(np.ones((20, 10)))
        output = pst._get_signal_mean_position_and_value(s)
        assert output == (4.5, 9.5, 1)
        s.axes_manager[0].scale = 2
        output = pst._get_signal_mean_position_and_value(s)
        assert output == (9, 9.5, 1)
        s.axes_manager[0].scale = 0.5
        output = pst._get_signal_mean_position_and_value(s)
        assert output == (2.25, 9.5, 1)
        s.axes_manager[1].scale = 10
        output = pst._get_signal_mean_position_and_value(s)
        assert output == (2.25, 95., 1)

    def test_origin_and_scale(self):
        s = hs.signals.Signal2D(np.zeros((30, 10)))
        output = pst._get_signal_mean_position_and_value(s)
        assert output == (4.5, 14.5, 0)
        s.axes_manager[0].offset = 10
        s.axes_manager[0].scale = 0.5
        output = pst._get_signal_mean_position_and_value(s)
        assert output == (12.25, 14.5, 0)
        s.axes_manager[1].offset = -50
        s.axes_manager[1].scale = 2
        output = pst._get_signal_mean_position_and_value(s)
        assert output == (12.25, -21., 0)
