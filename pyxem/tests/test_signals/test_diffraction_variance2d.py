# -*- coding: utf-8 -*-
# Copyright 2017-2020 The pyXem developers
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

from pyxem.signals.diffraction_variance2d import DiffractionVariance2D
from pyxem.signals.diffraction_variance2d import ImageVariance
from pyxem.signals.diffraction_variance1d import DiffractionVariance1D
from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D


class TestDiffractionVariance:

    def test_get_diffraction_variance_signal(self,
                                             diffraction_pattern):
        difvar = DiffractionVariance2D(diffraction_pattern)
        assert isinstance(difvar, DiffractionVariance2D)

    def test_get_dif_var_radial_profile(self,
                                        diffraction_pattern):
        difvar = DiffractionVariance2D(diffraction_pattern)
        rp = difvar.get_radial_profile()
        assert isinstance(rp, DiffractionVariance1D)

    @pytest.fixture
    def axes_test_dp(self):
        dp_data = np.random.randint(0, 10, (2, 2, 10, 10))
        dp = ElectronDiffraction2D(dp_data)
        return dp

    def test_radial_profile_axes(self, axes_test_dp):
        n_scale = 0.5
        name = 'real_space'
        units = 'um'

        axes_test_dp.axes_manager.navigation_axes[0].scale = n_scale
        axes_test_dp.axes_manager.navigation_axes[0].name = name
        axes_test_dp.axes_manager.navigation_axes[0].units = units
        # name and units are flipped to make sure everything follows
        axes_test_dp.axes_manager.navigation_axes[1].scale = 2 * n_scale
        axes_test_dp.axes_manager.navigation_axes[1].name = units
        axes_test_dp.axes_manager.navigation_axes[1].units = name

        rp = axes_test_dp.get_radial_profile()
        rp_scale_x = rp.axes_manager.navigation_axes[0].scale
        rp_units_x = rp.axes_manager.navigation_axes[0].units
        rp_name_x = rp.axes_manager.navigation_axes[0].name

        rp_scale_y = rp.axes_manager.navigation_axes[1].scale
        rp_units_y = rp.axes_manager.navigation_axes[1].units
        rp_name_y = rp.axes_manager.navigation_axes[1].name

        assert n_scale == rp_scale_x
        assert 2 * n_scale == rp_scale_y
        assert units == rp_units_x
        assert name == rp_name_x
        assert name == rp_units_y
        assert units == rp_name_y


class TestImageVariance:

    def test_get_image_variance_signal(self,
                                       diffraction_pattern):
        imvar = ImageVariance(diffraction_pattern)
        assert isinstance(imvar, ImageVariance)
