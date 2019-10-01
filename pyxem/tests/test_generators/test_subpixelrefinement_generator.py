# -*- coding: utf-8 -*-
# Copyright 2017-2019 The pyXem developers
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

from pyxem.generators.subpixelrefinement_generator import SubpixelrefinementGenerator, get_simulated_disc
from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D
from skimage import draw

class Test_init():
    @pytest.mark.xfail(raises=ValueError)
    def test_out_of_range_vectors_numpy(self):
        """ Tests that putting vectors that lie outside of the diffraction patterns
        raise a ValueError"""
        vector = np.array([[1, -100]])
        dp = ElectronDiffraction2D(np.ones((20, 20)))
        sprg = SubpixelrefinementGenerator(dp, vector)

    @pytest.mark.xfail(raises=ValueError)
    def test_out_of_range_vectors_DiffractionVectors(self):
        vectors = DiffractionVectors(np.array([[1, -100]]))
        dp = ElectronDiffraction2D(np.ones((20, 20)))
        sprg = SubpixelrefinementGenerator(dp, vectors)

    @pytest.mark.xfail(raises=ValueError)
    def test_wrong_navigation_dimensions(self):
        dp = ElectronDiffraction2D(np.zeros((2, 2, 8, 8)))
        vectors = DiffractionVectors(np.zeros((1, 2)))
        dp.axes_manager.set_signal_dimension(2)
        vectors.axes_manager.set_signal_dimension(0)
        SPR_generator = SubpixelrefinementGenerator(dp, vectors)

class Test_finder():

    def create_spot(self):
        z1 = np.zeros((128, 128))
        z2 = np.zeros((128, 128))

        for r in [4, 3, 2]:
            c = 1 / r
            rr, cc = draw.circle(30, 90, radius=r, shape=z1.shape)  # 30 is y!
            z1[rr, cc] = c
            z2[rr, cc] = c
            rr2, cc2 = draw.circle(100, 60, radius=r, shape=z2.shape)
            z2[rr2, cc2] = c

        #TODO: mark the centers for the local gaussian method
        #TODO: add a slight shift to make z1a and z2a
        dp = ElectronDiffraction2D(np.asarray([[z1, z1], [z2, z2]]))  # this needs to be in 2x2
        return dp

    def create_Diffraction_vectors(self):
        v1 = np.array([[90 - 64, 30 - 64]])
        v2 = np.array([[90 - 64, 30 - 64], [100 - 64, 60 - 64]])
        vectors = DiffractionVectors(np.array([[v1, v1], [v2, v2]]))
        vectors.axes_manager.set_signal_dimension(0)
        return vectors

    def create_numpy_vectors(self):
        v = np.array([[90 - 64, 30 - 64]]
        return v

    def test_assertioned_xc():
        spr = SubpixelrefinementGenerator(dp, diffraction_vectors)
        s = spr.conventional_xc(12, 4, 8)
        error = s.data[0, 0] - np.asarray([[90 - 64, 30 - 64]])
        rms_error = np.sqrt(error[0, 0]**2 + error[0, 1]**2)
        assert rms_error < 0.2  # 1/5th a pixel

    def test_assertioned_com():
        # should be perfect detection for these cases
        spr = SubpixelrefinementGenerator(dp, diffraction_vectors)
        s = spr.center_of_mass_method(8)
        error = s.data[0, 0] - np.asarray([[90 - 64, 30 - 64]])
        rms_error = np.sqrt(error[0, 0]**2 + error[0, 1]**2)
        assert rms_error < 1e-5  # perfect detection for this trivial case

    def test_assertioned_log():
        #rename some of the internal bits of functionality
        pass


class Test_misc():
    @pytest.mark.parametrize('dp, diffraction_vectors',
                         [(create_spot_gaussian(), np.array([[55 - 64, 25 - 64]]))])
    @pytest.mark.filterwarnings('ignore::UserWarning')  # our warning
    def test_bad_square_size_local_gaussian_method(self,dp, diffraction_vectors):
        spr = SubpixelrefinementGenerator(dp, diffraction_vectors)
        s = spr.local_gaussian_method(2)

    def test_xy_errors_in_conventional_xc_method_as_per_issue_490(self):
        """ This was the MWE example code for the issue """
        dp = get_simulated_disc(100,20)
        # translate y by +4
        shifted = np.pad(dp, ((0,4),(0,0)), 'constant')[4:].reshape(1,1,*dp.shape)
        signal = ElectronDiffraction2D(shifted)
        spg = SubpixelrefinementGenerator(signal, np.array([[0,0]]))
        peaks = spg.conventional_xc(100,20,1).data[0,0,0] #as quoted in the issue
        np.testing.assert_allclose([0,-4],peaks)
        """ we also test com method for clarity """
        peaks = spg.center_of_mass_method(60).data[0,0,0]
        np.testing.assert_allclose([0,-4],peaks,atol=1.5)
