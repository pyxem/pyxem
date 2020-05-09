# -*- coding: utf-8 -*-
# Copyright 2016-2020 The pyXem developers
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

from pyxem.generators.subpixelrefinement_generator import (
    SubpixelrefinementGenerator,
    get_simulated_disc,
)
from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D
from skimage import draw


# @pytest.mark.xfail(raises=ValueError)
class Test_init_xfails:
    def test_out_of_range_vectors_numpy(self):
        """Test that putting vectors that lie outside of the
        diffraction patterns raises a ValueError"""
        vector = np.array([[1, -100]])
        dp = ElectronDiffraction2D(np.ones((20, 20)))

        with pytest.raises(
            ValueError,
            match="Some of your vectors do not lie within your diffraction pattern",
        ):
            sprg = SubpixelrefinementGenerator(dp, vector)

    def test_out_of_range_vectors_DiffractionVectors(self):
        """Test that putting vectors that lie outside of the
        diffraction patterns raises a ValueError"""
        vectors = DiffractionVectors(np.array([[1, -100]]))
        dp = ElectronDiffraction2D(np.ones((20, 20)))

        with pytest.raises(
            ValueError,
            match="Some of your vectors do not lie within your diffraction pattern",
        ):
            sprg = SubpixelrefinementGenerator(dp, vectors)

    def test_wrong_navigation_dimensions(self):
        """Tests that navigation dimensions must be appropriate too."""
        dp = ElectronDiffraction2D(np.zeros((2, 2, 8, 8)))
        vectors = DiffractionVectors(np.zeros((1, 2)))
        dp.axes_manager.set_signal_dimension(2)
        vectors.axes_manager.set_signal_dimension(0)

        # Note - uses regex via re.search()
        with pytest.raises(
            ValueError,
            match=r"Vectors with shape .* must have the same navigation shape as .*",
        ):
            sprg = SubpixelrefinementGenerator(dp, vectors)


class set_up_for_subpixelpeakfinders:
    def create_spot(self):
        z1, z1a = np.zeros((128, 128)), np.zeros((128, 128))
        z2, z2a = np.zeros((128, 128)), np.zeros((128, 128))

        rr, cc = draw.circle(30, 90, radius=4, shape=z1.shape)  # 30 is y!
        z1[rr, cc], z2[rr, cc] = 1, 1
        rr2, cc2 = draw.circle(100, 60, radius=4, shape=z2.shape)
        z2[rr2, cc2] = 1
        rr, cc = draw.circle(30, 90 + 3, radius=4, shape=z1.shape)  # 30 is y!
        z1a[rr, cc], z2a[rr, cc] = 1, 1
        rr2, cc2 = draw.circle(100 - 2, 60, radius=4, shape=z2.shape)
        z2a[rr2, cc2] = 1

        # marks centers for local com and local_gaussian_method
        z1[30, 90], z2[30, 90], z2[100, 60] = 2, 2, 2
        z1a[30, 93], z2a[30, 93], z2a[98, 60] = 10, 10, 10

        dp = ElectronDiffraction2D(
            np.asarray([[z1, z1a], [z2, z2a]])
        )  # this needs to be in 2x2
        return dp

    def create_Diffraction_vectors(self):
        v1 = np.array([[90 - 64, 30 - 64]])
        v2 = np.array([[90 - 64, 30 - 64], [100 - 64, 60 - 64]])
        vectors = DiffractionVectors(np.array([[v1, v1], [v2, v2]]))
        vectors.axes_manager.set_signal_dimension(0)
        return vectors


class Test_subpixelpeakfinders:
    """ Tests the various peak finders have the correct x,y conventions for
    both the vectors and the shifts, in both the numpy and the DiffractionVectors
    cases as well as confirming we have avoided 'off by one' errors """

    set_up = set_up_for_subpixelpeakfinders()

    @pytest.fixture(
        params=[set_up.create_Diffraction_vectors(), np.array([[90 - 64, 30 - 64]])]
    )
    def diffraction_vectors(self, request):
        # see https://bit.ly/2mXpSlD for an example of this architecture
        return request.param

    def get_spr(self, diffraction_vectors):
        dp = set_up_for_subpixelpeakfinders().create_spot()
        return SubpixelrefinementGenerator(dp, diffraction_vectors)

    def no_shift_case(self, s):
        error = s.data[0, 0] - np.asarray([[90 - 64, 30 - 64]])
        rms_error = np.sqrt(error[0, 0] ** 2 + error[0, 1] ** 2)
        assert rms_error < 1e-5  # perfect detection for this trivial case

    def x_shift_case(self, s):
        error = s.data[0, 1] - np.asarray([[93 - 64, 30 - 64]])
        rms_error = np.sqrt(error[0, 0] ** 2 + error[0, 1] ** 2)
        assert rms_error < 0.5  # correct to within a pixel

    def test_assertioned_xc(self, diffraction_vectors):
        subpixelsfound = self.get_spr(diffraction_vectors).conventional_xc(12, 4, 8)
        self.no_shift_case(subpixelsfound)
        self.x_shift_case(subpixelsfound)

    def test_assertioned_com(self, diffraction_vectors):
        subpixelsfound = self.get_spr(diffraction_vectors).center_of_mass_method(12)
        self.no_shift_case(subpixelsfound)
        self.x_shift_case(subpixelsfound)

    def test_assertioned_log(self, diffraction_vectors):
        with pytest.warns(
            UserWarning,
            match="peak in your pattern that lies on the edge of the square",
        ):
            subpixelsfound = self.get_spr(diffraction_vectors).local_gaussian_method(12)

        self.no_shift_case(subpixelsfound)
        self.x_shift_case(subpixelsfound)


# class Test_misc():
""" These tests will be removed for 0.11.0, but are needed for the log method &
security on the x/y conventions until then """


def create_spot_gaussian():
    z1 = np.zeros((128, 128))
    x = np.arange(0.0, 10, 1.0)
    y = x[:, np.newaxis]
    z1[20:30, 50:60] = np.exp(-((x - 5.1) ** 2 + (y - 5.3) ** 2) / 4)
    dp = ElectronDiffraction2D(
        np.asarray([[z1, z1], [z1, z1]])
    )  # this needs to be in 2x2
    return dp


@pytest.mark.parametrize(
    "dp, diffraction_vectors",
    [(create_spot_gaussian(), np.array([[55 - 64, 25 - 64]]))],
)
@pytest.mark.filterwarnings("ignore::UserWarning")  # our warning
def test_bad_square_size_local_gaussian_method(dp, diffraction_vectors):
    spr = SubpixelrefinementGenerator(dp, diffraction_vectors)
    s = spr.local_gaussian_method(2)


def test_xy_errors_in_conventional_xc_method_as_per_issue_490():
    """ This was the MWE example code for the issue """
    dp = get_simulated_disc(100, 20)
    # translate y by +4
    shifted = np.pad(dp, ((0, 4), (0, 0)), "constant")[4:].reshape(1, 1, *dp.shape)
    signal = ElectronDiffraction2D(shifted)
    spg = SubpixelrefinementGenerator(signal, np.array([[0, 0]]))
    peaks = spg.conventional_xc(100, 20, 1).data[0, 0, 0]  # as quoted in the issue
    np.testing.assert_allclose([0, -4], peaks)
    """ we also test com method for clarity """
    peaks = spg.center_of_mass_method(60).data[0, 0, 0]
    np.testing.assert_allclose([0, -4], peaks, atol=1.5)
    """ we also test reference_xc """
    peaks = spg.reference_xc(100, dp, 1).data[0, 0, 0]  # as quoted in the issue
    np.testing.assert_allclose([0, -4], peaks)
