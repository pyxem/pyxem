# -*- coding: utf-8 -*-
# Copyright 2016-2024 The pyXem developers
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
import numpy as np
import hyperspy.api as hs

from hyperspy.signals import Signal1D
from pyxem.signals import InSituDiffraction2D


class TestTimeSeriesReconstruction:
    @pytest.fixture
    def insitu_data(self):
        dc = InSituDiffraction2D(data=np.ones(shape=(5, 2, 2, 25, 25)))
        dc.axes_manager.signal_axes[0].scale = 0.1
        dc.axes_manager.signal_axes[0].name = "kx"
        dc.axes_manager.signal_axes[1].scale = 0.1
        dc.axes_manager.signal_axes[1].name = "ky"
        return dc

    def test_roll_time_axis(self, insitu_data):
        rolled_data = insitu_data.roll_time_axis(0)
        assert (
            rolled_data.axes_manager.navigation_axes[2].size
            == insitu_data.axes_manager.navigation_axes[0].size
        )

    @pytest.mark.parametrize(
        "roi", [hs.roi.CircleROI(1, 1, 0.5), hs.roi.RectangularROI(0, 1, 1, 2), None]
    )
    def test_different_roi(self, insitu_data, roi):
        series = insitu_data.get_time_series(roi=roi)
        assert len(series.axes_manager.navigation_axes) == 1
        assert (
            series.axes_manager.navigation_axes[0].size
            == insitu_data.axes_manager.navigation_axes[2].size
        )
        assert (
            series.axes_manager.signal_axes[0].size
            == insitu_data.axes_manager.navigation_axes[0].size
        )
        assert (
            series.axes_manager.signal_axes[1].size
            == insitu_data.axes_manager.navigation_axes[1].size
        )

    def test_time_axis(self, insitu_data):
        series = insitu_data.get_time_series(
            roi=hs.roi.CircleROI(1, 1, 0.5), time_axis=0
        )
        assert (
            series.axes_manager.navigation_axes[0].size
            == insitu_data.axes_manager.navigation_axes[0].size
        )


class TestCorrelation:
    @pytest.fixture
    def insitu_data(self):
        dc = InSituDiffraction2D(data=np.random.rand(50, 10, 10, 4, 4))
        dc.axes_manager.signal_axes[0].scale = 0.1
        dc.axes_manager.signal_axes[0].name = "kx"
        dc.axes_manager.signal_axes[1].scale = 0.1
        dc.axes_manager.signal_axes[1].name = "ky"
        return dc

    def test_drift(self, insitu_data):
        shifts = insitu_data.get_drift_vectors()
        assert (
            shifts.axes_manager.navigation_axes[0].size
            == insitu_data.axes_manager.navigation_axes[2].size
        )
        assert shifts.axes_manager.signal_axes[0].size == 2

        assert isinstance(shifts, Signal1D)

    @pytest.mark.parametrize(
        "shifts",
        [
            Signal1D(np.zeros((50, 2))),
            Signal1D(
                np.repeat(np.linspace(0, 2, 50)[:, np.newaxis], repeats=2, axis=1)
            ),
            None,
        ],
    )
    def test_drift_corrected_g2_nonlazy(self, insitu_data, shifts):
        shifted_data = insitu_data.correct_real_space_drift(
            shifts=shifts, lazy_result=False
        )
        assert isinstance(shifted_data, InSituDiffraction2D)

        g2 = shifted_data.get_g2_2d_kresolved()
        assert g2.axes_manager.signal_axes[-1].size == 50
        mean_g2 = g2.isig[:, :, 1:].mean().data
        num_index = ~np.isreal(mean_g2)
        np.testing.assert_allclose(
            np.ones((49, 4, 4))[num_index], mean_g2[num_index], atol=0.1
        )

    def test_drift_corrected_g2_lazy(self, insitu_data):
        shifts = Signal1D(
            np.repeat(np.linspace(0, 2, 50)[:, np.newaxis], repeats=2, axis=1)
        )
        lazy_data = insitu_data.as_lazy()
        shifted_data = lazy_data.correct_real_space_drift(shifts=shifts)
        assert shifted_data._lazy
        assert isinstance(shifted_data, InSituDiffraction2D)
        g2_lazy = shifted_data.get_g2_2d_kresolved()
        assert g2_lazy.axes_manager.signal_axes[-1].size == 50
        g2_lazy.compute()
        mean_g2 = g2_lazy.isig[:, :, 1:].mean().data
        num_index = ~np.isreal(mean_g2)
        np.testing.assert_allclose(
            np.ones((49, 4, 4))[num_index], mean_g2[num_index], atol=0.1
        )

    @pytest.mark.parametrize(
        "shifts",
        [
            Signal1D(np.zeros((50, 2))),
            Signal1D(
                np.repeat(np.linspace(0, 2, 50)[:, np.newaxis], repeats=2, axis=1)
            ),
            None,
        ],
    )
    def test_fast_drift_corrected_g2_nonlazy(self, insitu_data, shifts):
        shifted_data = insitu_data.correct_real_space_drift_fast(shifts=shifts)
        assert isinstance(shifted_data, InSituDiffraction2D)

        g2 = shifted_data.get_g2_2d_kresolved()
        assert g2.axes_manager.signal_axes[-1].size == 50
        mean_g2 = g2.isig[:, :, 1:-1].mean(axis=[-1, -2, -3]).data
        num_index = ~np.isreal(mean_g2)
        np.testing.assert_allclose(
            np.ones((10, 10))[num_index], mean_g2[num_index], atol=0.1
        )

    def test_fast_drift_corrected_g2_lazy(self, insitu_data):
        shifts = Signal1D(
            np.repeat(np.linspace(0, 2, 50)[:, np.newaxis], repeats=2, axis=1)
        )
        lazy_data = insitu_data.as_lazy()
        lazy_data.rechunk((25, 5, 5, 4, 4))
        with pytest.raises(Exception) as exc_info:
            lazy_data.correct_real_space_drift_fast(shifts=shifts)

        assert exc_info.match(
            "Spatial axes are chunked. Please rechunk signal or use 'correct_real_space_drift' "
            "instead"
        )
        lazy_data.rechunk((1, 10, 10, 4, 4))
        shifted_data = lazy_data.correct_real_space_drift_fast(shifts=shifts)
        assert shifted_data._lazy
        assert isinstance(shifted_data, InSituDiffraction2D)

        g2_lazy = shifted_data.get_g2_2d_kresolved()
        assert g2_lazy.axes_manager.signal_axes[-1].size == 50
        g2_lazy.compute()
        mean_g2 = g2_lazy.isig[:, :, 1:-1].mean(axis=[-1, -2, -3]).data
        num_index = ~np.isreal(mean_g2)
        np.testing.assert_allclose(
            np.ones((10, 10))[num_index], mean_g2[num_index], atol=0.1
        )

    @pytest.mark.parametrize("normalization", ["self", "split"])
    def test_g2_normalization(self, insitu_data, normalization):
        g2 = insitu_data.get_g2_2d_kresolved(normalization=normalization)
        mean_g2 = g2.isig[:, :, 1:-1].mean(axis=[-1, -2, -3]).data
        num_index = ~np.isreal(mean_g2)
        np.testing.assert_allclose(
            np.ones((10, 10))[num_index], mean_g2[num_index], atol=0.1
        )

    @pytest.mark.parametrize("trs", [np.linspace(0, 10, 25), 10])
    @pytest.mark.parametrize("bins", [(2, 2, 5), (1, 4, 1)])
    def test_g2_bin_resample_time(self, insitu_data, trs, bins):
        g2 = insitu_data.get_g2_2d_kresolved(
            k1bin=bins[0], k2bin=bins[1], tbin=bins[2], resample_time=trs
        )
        assert g2.axes_manager.signal_axes[0].size == int(4 / bins[0])
        assert g2.axes_manager.signal_axes[1].size == int(4 / bins[1])
        mean_g2 = g2.isig[:, :, 1:-1].mean(axis=[-1, -2, -3]).data
        num_index = ~np.isreal(mean_g2)
        np.testing.assert_allclose(
            np.ones((10, 10))[num_index], mean_g2[num_index], atol=0.1
        )

    def test_unrolled_time_axes(self, insitu_data):
        rolled_data = insitu_data.roll_time_axis(0)
        shifted_data = rolled_data.correct_real_space_drift(
            shifts=Signal1D(np.zeros((50, 2))), time_axis=1
        )
        assert (
            shifted_data.axes_manager.navigation_axes[2].size
            == insitu_data.axes_manager.navigation_axes[2].size
        )
        shifted_data_fast = rolled_data.correct_real_space_drift_fast(
            shifts=Signal1D(np.zeros((50, 2))), time_axis=1
        )
        assert (
            shifted_data_fast.axes_manager.navigation_axes[2].size
            == insitu_data.axes_manager.navigation_axes[2].size
        )
        g2 = rolled_data.get_g2_2d_kresolved(time_axis=1)
        assert (
            g2.axes_manager.signal_axes[-1].size
            == insitu_data.axes_manager.navigation_axes[2].size
        )
