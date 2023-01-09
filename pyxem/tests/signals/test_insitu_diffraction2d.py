# -*- coding: utf-8 -*-
# Copyright 2016-2022 The pyXem developers
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
import dask.array as da
import hyperspy.api as hs

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

    @pytest.mark.parametrize("roi",
                             [hs.roi.CircleROI(1, 1, 0.5),
                              hs.roi.RectangularROI(0, 1, 1, 2),
                              None]
                             )
    def test_different_roi(self, insitu_data, roi):
        series = insitu_data.get_time_series(roi=roi)
        assert len(series.axes_manager.navigation_axes) == 1
        assert series.axes_manager.navigation_axes[0].size == insitu_data.axes_manager.navigation_axes[2].size
        assert series.axes_manager.signal_axes[0].size == insitu_data.axes_manager.navigation_axes[0].size
        assert series.axes_manager.signal_axes[1].size == insitu_data.axes_manager.navigation_axes[1].size

    def test_time_axis(self, insitu_data):
        series = insitu_data.get_time_series(roi=hs.roi.CircleROI(1, 1, 0.5), time_axis=0)
        assert series.axes_manager.navigation_axes[0].size == insitu_data.axes_manager.navigation_axes[0].size
