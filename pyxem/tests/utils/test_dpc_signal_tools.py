# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
#
# This file is part of pyXem.
#
# pyXem is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyXem is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyXem.  If not, see <http://www.gnu.org/licenses/>.


import pytest
from pytest import approx
import numpy as np
from numpy.testing import assert_allclose

import hyperspy.api as hs

from pyxem.signals.differential_phase_contrast import make_bivariate_histogram
import pyxem.utils.pixelated_stem_tools as pst


class TestGetRgbPhaseMagnitudeArray:
    def test_simple(self):
        phase = np.zeros((50, 50))
        magnitude = np.zeros((50, 50))
        rgb_array = pst._get_rgb_phase_magnitude_array(phase, magnitude)
        assert (rgb_array == 0.0).all()

    def test_magnitude_zero(self):
        phase = np.random.random((50, 50))
        magnitude = np.zeros((50, 50))
        rgb_array = pst._get_rgb_phase_magnitude_array(phase, magnitude)
        assert (rgb_array == 0.0).all()

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
        assert (phase0 == 0.0).all()

    def test_rotation(self):
        phase = np.zeros((50, 50))
        phase0 = pst._find_phase(phase, rotation=90)
        assert (phase0 == np.pi / 2).all()
        phase1 = pst._find_phase(phase, rotation=45)
        assert (phase1 == np.pi / 4).all()
        phase2 = pst._find_phase(phase, rotation=180)
        assert (phase2 == np.pi).all()
        phase3 = pst._find_phase(phase, rotation=360)
        assert (phase3 == 0).all()
        phase4 = pst._find_phase(phase, rotation=-90)
        assert (phase4 == 3 * np.pi / 2).all()

    def test_max_phase(self):
        phase = (np.ones((50, 50)) * np.pi * 0.5) + np.pi
        phase0 = pst._find_phase(phase, max_phase=np.pi)
        assert (phase0 == np.pi / 2).all()


class TestMakeBivariateHistogram:
    def test_single_x(self):
        size = 100
        x, y = np.ones(size), np.zeros(size)
        s = make_bivariate_histogram(x, y)
        hist_iX = s.axes_manager[0].value2index(1.0)
        hist_iY = s.axes_manager[1].value2index(0.0)
        assert s.data[hist_iY, hist_iX] == size
        s.data[hist_iY, hist_iX] = 0
        assert not s.data.any()

    def test_single_negative_x(self):
        size = 100
        x, y = -np.ones(size), np.zeros(size)
        s = make_bivariate_histogram(x, y)
        hist_iX = s.axes_manager[0].value2index(-1)
        hist_iY = s.axes_manager[1].value2index(0)
        assert s.data[hist_iY, hist_iX] == size
        s.data[hist_iY, hist_iX] = 0
        assert not s.data.any()

    def test_single_negative_x_y(self):
        size = 100
        x, y = -np.ones(size), np.ones(size)
        s = make_bivariate_histogram(x, y)
        hist_iX = s.axes_manager[0].value2index(-1)
        hist_iY = s.axes_manager[1].value2index(1)
        assert s.data[hist_iY, hist_iX] == size
        s.data[hist_iY, hist_iX] = 0
        assert not s.data.any()


class TestGetCornerValues:
    def test_simple(self):
        s = hs.signals.Signal2D(np.zeros((101, 101)))
        corner_values = pst._get_corner_values(s)
        assert corner_values.shape == (3, 4)
        # tl: top left, bl: bottom right, tr: top right, br: bottom right
        corner_tl, corner_bl = corner_values[:, 0], corner_values[:, 1]
        corner_tr, corner_br = corner_values[:, 2], corner_values[:, 3]
        assert (corner_tl == (2.5, 2.5, 0.0)).all()
        assert (corner_bl == (2.5, 97.5, 0.0)).all()
        assert (corner_tr == (97.5, 2.5, 0.0)).all()
        assert (corner_br == (97.5, 97.5, 0.0)).all()

    def test_non_square_shape(self):
        s = hs.signals.Signal2D(np.zeros((201, 101)))
        corner_values = pst._get_corner_values(s)
        corner_tl, corner_bl = corner_values[:, 0], corner_values[:, 1]
        corner_tr, corner_br = corner_values[:, 2], corner_values[:, 3]
        assert (corner_tl == (2.5, 5.0, 0.0)).all()
        assert (corner_bl == (2.5, 195.0, 0.0)).all()
        assert (corner_tr == (97.5, 5.0, 0.0)).all()
        assert (corner_br == (97.5, 195.0, 0.0)).all()

    def test_corner_size_parameters(self):
        s = hs.signals.Signal2D(np.zeros((101, 101)))
        corner_values = pst._get_corner_values(s, corner_size=0.1)
        corner_tl, corner_bl = corner_values[:, 0], corner_values[:, 1]
        corner_tr, corner_br = corner_values[:, 2], corner_values[:, 3]
        assert (corner_tl == (5.0, 5.0, 0.0)).all()
        assert (corner_bl == (5.0, 95.0, 0.0)).all()
        assert (corner_tr == (95.0, 5.0, 0.0)).all()
        assert (corner_br == (95.0, 95.0, 0.0)).all()

    def test_different_corner_values(self):
        s = hs.signals.Signal2D(np.zeros((100, 100)))
        s.data[:5, :5] = 10
        s.data[-5:, :5] = 20
        s.data[:5, -5:] = 30
        s.data[-5:, -5:] = 40
        corner_values = pst._get_corner_values(s)
        assert (corner_values[2] == (10, 20, 30, 40)).all()
        corner_values = pst._get_corner_values(s, corner_size=0.1)
        assert (corner_values[2] == (2.5, 5.0, 7.5, 10.0)).all()

    def test_different_corner_values_non_square(self):
        s = hs.signals.Signal2D(np.zeros((200, 100)))
        s.data[:10, :5] = 10
        s.data[-10:, :5] = 20
        s.data[:10, -5:] = 30
        s.data[-10:, -5:] = 40
        corner_values = pst._get_corner_values(s)
        assert (corner_values[2] == (10, 20, 30, 40)).all()
        corner_values = pst._get_corner_values(s, corner_size=0.1)
        assert (corner_values[2] == (2.5, 5.0, 7.5, 10.0)).all()

    def test_offset(self):
        s = hs.signals.Signal2D(np.zeros((101, 101)))
        s.axes_manager[0].offset = 10
        s.axes_manager[1].offset = -10
        corner_values = pst._get_corner_values(s)
        corner_tl, corner_bl = corner_values[:, 0], corner_values[:, 1]
        corner_tr, corner_br = corner_values[:, 2], corner_values[:, 3]
        assert (corner_tl == (12.5, -7.5, 0.0)).all()
        assert (corner_bl == (12.5, 87.5, 0.0)).all()
        assert (corner_tr == (107.5, -7.5, 0.0)).all()
        assert (corner_br == (107.5, 87.5, 0.0)).all()

    def test_scale(self):
        s = hs.signals.Signal2D(np.zeros((101, 101)))
        s.axes_manager[0].scale = 0.5
        s.axes_manager[1].scale = 2
        corner_values = pst._get_corner_values(s)
        corner_tl, corner_bl = corner_values[:, 0], corner_values[:, 1]
        corner_tr, corner_br = corner_values[:, 2], corner_values[:, 3]
        assert (corner_tl == (1.25, 5.0, 0.0)).all()
        assert (corner_bl == (1.25, 195.0, 0.0)).all()
        assert (corner_tr == (48.75, 5.0, 0.0)).all()
        assert (corner_br == (48.75, 195.0, 0.0)).all()

    def test_crop_square(self):
        s = hs.signals.Signal2D(np.zeros((200, 200))).isig[50:150, 50:150]
        corner_values = pst._get_corner_values(s)
        corner_tl, corner_bl = corner_values[:, 0], corner_values[:, 1]
        corner_tr, corner_br = corner_values[:, 2], corner_values[:, 3]
        assert (corner_tl == (52.0, 52, 0.0)).all()
        assert (corner_bl == (52.0, 147.0, 0.0)).all()
        assert (corner_tr == (147.0, 52.0, 0.0)).all()
        assert (corner_br == (147.0, 147.0, 0.0)).all()

    def test_crop_square_values(self):
        s = hs.signals.Signal2D(np.zeros((200, 200)))
        s.data[50:60, 50:60] = 10
        s.data[140:150, 50:60] = 20
        s.data[50:60, 140:150] = 30
        s.data[140:150, 140:150] = 40
        s = s.isig[50:150, 50:150]
        corner_values = pst._get_corner_values(s)
        corner_tl, corner_bl = corner_values[:, 0], corner_values[:, 1]
        corner_tr, corner_br = corner_values[:, 2], corner_values[:, 3]
        assert (corner_tl == (52.0, 52, 10.0)).all()
        assert (corner_bl == (52.0, 147.0, 20.0)).all()
        assert (corner_tr == (147.0, 52.0, 30.0)).all()
        assert (corner_br == (147.0, 147.0, 40.0)).all()

    def test_wrong_input_dimensions(self):
        s = hs.signals.Signal2D(np.ones((2, 10, 10)))
        with pytest.raises(ValueError):
            pst._get_corner_values(s)
        s = hs.signals.Signal2D(np.ones((2, 2, 10, 10)))
        with pytest.raises(ValueError):
            pst._get_corner_values(s)
        s = hs.signals.Signal1D(np.ones(10))
        with pytest.raises(ValueError):
            pst._get_corner_values(s)


class TestFitRampToImage:
    def test_zero_values(self):
        data = np.zeros((100, 100))
        s = hs.signals.Signal2D(data)
        ramp = pst._fit_ramp_to_image(s, corner_size=0.05)
        assert_allclose(ramp, data, atol=1e-15)

    def test_ones_values(self):
        data = np.ones((100, 100))
        s = hs.signals.Signal2D(data)
        ramp = pst._fit_ramp_to_image(s, corner_size=0.05)
        assert_allclose(ramp, data, atol=1e-30)

    def test_negative_values(self):
        data = np.ones((100, 100)) * -10
        s = hs.signals.Signal2D(data)
        ramp = pst._fit_ramp_to_image(s, corner_size=0.05)
        assert_allclose(ramp, data, atol=1e-30)

    def test_large_values_in_middle(self):
        data = np.zeros((100, 100))
        data[5:95, :] = 10
        data[:, 5:95] = 10
        s = hs.signals.Signal2D(data)
        ramp05 = pst._fit_ramp_to_image(s, corner_size=0.05)
        assert_allclose(ramp05, np.zeros((100, 100)), atol=1e-15)
        ramp10 = pst._fit_ramp_to_image(s, corner_size=0.1)
        assert (ramp05 != ramp10).all()

    def test_different_corner_values(self):
        data = np.zeros((100, 100))
        data[:5, :5], data[:5, -5:] = -10, 10
        data[-5:, :5] = 10
        data[-5:, -5:] = 30
        s = hs.signals.Signal2D(data)
        ramp = pst._fit_ramp_to_image(s, corner_size=0.05)
        s.data = s.data - ramp
        assert approx(s.data[:5, :5].mean()) == 0.0
        assert approx(s.data[:5, -5:].mean()) == 0.0
        assert approx(s.data[-5:, :5].mean()) == 0.0
        assert approx(s.data[-5:, -5:].mean()) == 0.0
        assert s.data[5:95, 5:95].mean() != 0

    def test_wrong_input_dimensions(self):
        s = hs.signals.Signal2D(np.ones((2, 10, 10)))
        with pytest.raises(ValueError):
            pst._fit_ramp_to_image(s)
        s = hs.signals.Signal2D(np.ones((2, 2, 10, 10)))
        with pytest.raises(ValueError):
            pst._fit_ramp_to_image(s)
        s = hs.signals.Signal1D(np.ones(10))
        with pytest.raises(ValueError):
            pst._fit_ramp_to_image(s)


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
        s = hs.signals.Signal2D(np.ones((10, 10)) * 9)
        output = pst._get_signal_mean_position_and_value(s)
        assert output[2] == 9.0

    def test_non_square_shape(self):
        s = hs.signals.Signal2D(np.zeros((10, 5)))
        # s gets the shape 5, 10. Due to the axes being reversed
        output = pst._get_signal_mean_position_and_value(s)
        assert output[0] == 2.0
        assert output[1] == 4.5

    def test_wrong_input_dimensions(self):
        s = hs.signals.Signal2D(np.ones((2, 10, 10)))
        with pytest.raises(ValueError):
            pst._get_signal_mean_position_and_value(s)
        s = hs.signals.Signal2D(np.ones((2, 2, 10, 10)))
        with pytest.raises(ValueError):
            pst._get_signal_mean_position_and_value(s)
        s = hs.signals.Signal1D(np.ones(10))
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
        assert output == (2.25, 95.0, 1)

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
        assert output == (12.25, -21.0, 0)
