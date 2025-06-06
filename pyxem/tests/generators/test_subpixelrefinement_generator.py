# -*- coding: utf-8 -*-
# Copyright 2016-2025 The pyXem developers
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

from skimage import draw

from pyxem.generators import SubpixelrefinementGenerator
from pyxem.generators.subpixelrefinement_generator import (
    get_simulated_disc,
    get_experimental_square,
)
from pyxem.signals import DiffractionVectors, ElectronDiffraction2D


@pytest.fixture()
def exp_disc():
    ss, disc_radius, upsample_factor = int(60), 6, 10

    arr = np.zeros((ss, ss))
    rr, cc = draw.disk(
        (int(ss / 2) + 20, int(ss / 2) - 10), radius=disc_radius, shape=arr.shape
    )
    arr[rr, cc] = 1
    return arr


@pytest.mark.filterwarnings("ignore::UserWarning")  # various skimage warnings
def test_experimental_square_size(exp_disc):
    square = get_experimental_square(exp_disc, [17, 19], 6)
    assert square.shape[0] == int(6)
    assert square.shape[1] == int(6)


def test_failure_for_non_even_entry_to_get_simulated_disc():
    with pytest.raises(ValueError, match="'square_size' must be an even number"):
        _ = get_simulated_disc(61, 5)


def test_failure_for_non_even_errors_get_experimental_square(exp_disc):
    with pytest.raises(ValueError, match="'square_size' must be an even number"):
        _ = get_experimental_square(exp_disc, [17, 19], 7)


class Test_init_xfails:
    def test_out_of_range_vectors_numpy(self):
        """Test that putting vectors that lie outside of the
        diffraction patterns raises a ValueError"""
        vector = np.array([[1, -100]])
        dp = ElectronDiffraction2D(np.ones((20, 20)))

        with pytest.raises(ValueError):
            _ = SubpixelrefinementGenerator(dp, vector)

    def test_out_of_range_vectors_DiffractionVectors(self):
        """Test that putting vectors that lie outside of the
        diffraction patterns raises a ValueError"""
        vectors = DiffractionVectors(np.array([[1, -100]]))
        dp = ElectronDiffraction2D(np.ones((20, 20)))

        with pytest.raises(ValueError):
            _ = SubpixelrefinementGenerator(dp, vectors)

    def test_wrong_navigation_dimensions(self):
        """Tests that navigation dimensions must be appropriate too."""
        dp = ElectronDiffraction2D(np.zeros((2, 2, 8, 8)))
        vectors = DiffractionVectors(np.zeros((1, 2)))
        # Note - uses regex via re.search()
        with pytest.raises(
            ValueError,
            match=r"Vectors with shape .* must have the same navigation shape as .*",
        ):
            _ = SubpixelrefinementGenerator(dp, vectors)


class set_up_for_subpixelpeakfinders:
    def create_spot(self):
        z1, z1a = np.zeros((128, 128)), np.zeros((128, 128))
        z2, z2a = np.zeros((128, 128)), np.zeros((128, 128))

        rr, cc = draw.disk((30, 90), radius=4, shape=z1.shape)  # 30 is y!
        z1[rr, cc], z2[rr, cc] = 1, 1
        rr2, cc2 = draw.disk((100, 60), radius=4, shape=z2.shape)
        z2[rr2, cc2] = 1
        rr, cc = draw.disk((30, 90 + 3), radius=4, shape=z1.shape)  # 30 is y!
        z1a[rr, cc], z2a[rr, cc] = 1, 1
        rr2, cc2 = draw.disk((100 - 2, 60), radius=4, shape=z2.shape)
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


class TestSubpixelPeakFinders:
    """Tests the various peak finders have the correct x,y conventions for
    both the vectors and the shifts, in both the numpy and the DiffractionVectors
    cases as well as confirming we have avoided 'off by one' errors"""

    @pytest.fixture
    def dp(self):
        z1, z1a = np.zeros((128, 128)), np.zeros((128, 128))
        z2, z2a = np.zeros((128, 128)), np.zeros((128, 128))

        rr, cc = draw.disk((30, 90), radius=4, shape=z1.shape)  # 30 is y!
        z1[rr, cc], z2[rr, cc] = 1, 1
        rr2, cc2 = draw.disk((100, 60), radius=4, shape=z2.shape)
        z2[rr2, cc2] = 1
        rr, cc = draw.disk((30, 90 + 3), radius=4, shape=z1.shape)  # 30 is y!
        z1a[rr, cc], z2a[rr, cc] = 1, 1
        rr2, cc2 = draw.disk((100 - 2, 60), radius=4, shape=z2.shape)
        z2a[rr2, cc2] = 1

        # marks centers for local com and local_gaussian_method
        z1[30, 90], z2[30, 90], z2[100, 60] = 2, 2, 2
        z1a[30, 93], z2a[30, 93], z2a[98, 60] = 10, 10, 10

        return ElectronDiffraction2D(
            np.asarray([[z1, z1a], [z2, z2a]])
        )  # this needs to be in 2x2

    @pytest.fixture
    def diffraction_vectors(self):
        v1 = np.array([[90 - 64, 30 - 64]])
        v2 = np.array([[90 - 64, 30 - 64], [100 - 64, 60 - 64]])
        return DiffractionVectors(
            np.array([[v1, v1], [v2, v2]], dtype=object),
            ragged=True,
        )

    @pytest.fixture
    def diffraction_vectors_1_empty(self):
        v1 = np.empty((0, 2))
        v2 = np.array([[90 - 64, 30 - 64], [100 - 64, 60 - 64]])
        return DiffractionVectors(
            np.array([[v1, v1], [v2, v2]], dtype=object),
            ragged=True,
        )

    @pytest.fixture
    def sprg(self, dp, diffraction_vectors):
        return SubpixelrefinementGenerator(dp, diffraction_vectors)

    @pytest.fixture
    def sprg_empty(self, dp, diffraction_vectors_1_empty):
        return SubpixelrefinementGenerator(dp, diffraction_vectors_1_empty)

    def no_shift_case(self, s):
        error = s.data[0, 0] - np.asarray([[90 - 64, 30 - 64]])
        rms_error = np.sqrt(error[0, 0] ** 2 + error[0, 1] ** 2)
        assert rms_error < 1e-5  # perfect detection for this trivial case

    def x_shift_case(self, s):
        error = s.data[0, 1] - np.asarray([[93 - 64, 30 - 64]])
        rms_error = np.sqrt(error[0, 0] ** 2 + error[0, 1] ** 2)
        assert rms_error < 0.5  # correct to within a pixel

    @pytest.mark.skip(reason="Broken downstream in scikit image")
    def test_assertioned_xc(self, sprg):
        subpixelsfound = sprg.conventional_xc(12, 4, 8)
        self.no_shift_case(subpixelsfound)
        self.x_shift_case(subpixelsfound)

    def test_assertioned_com(self, sprg):
        vector_refine = sprg.center_of_mass_method(12)
        self.no_shift_case(vector_refine)
        self.x_shift_case(vector_refine)

    def test_assertioned_com_empty(self, sprg_empty):
        vector_refine = sprg_empty.center_of_mass_method(12)
        assert isinstance(vector_refine, DiffractionVectors)

    def test_failure_out_of_bounds(self, dp):
        v = [[1000, 100]]
        vectors = np.array([[v, v], [v, v]])
        with pytest.raises(ValueError):
            SubpixelrefinementGenerator(dp, vectors)

    def test_log(self, sprg):
        with pytest.raises(
            NotImplementedError,
            match="This functionality was removed in v.0.13.0",
        ):
            _ = sprg.local_gaussian_method(12)


def test_xy_errors_in_conventional_xc_method_as_per_issue_490():
    """This was the MWE example code for the issue"""
    dp = get_simulated_disc(100, 20)
    # translate y by +4
    shifted = np.pad(dp, ((0, 4), (0, 0)), "constant")[4:].reshape(1, 1, *dp.shape)
    signal = ElectronDiffraction2D(shifted)
    spg = SubpixelrefinementGenerator(signal, np.array([[0, 0]]))
    peaks = spg.conventional_xc(100, 20, 1).data[0][0][0]  # as quoted in the issue
    np.testing.assert_allclose([0, -4], peaks)
    """ we also test com method for clarity """
    peaks = spg.center_of_mass_method(60).data[0][0][0]
    np.testing.assert_allclose([0, -4], peaks, atol=1.5)
    """ we also test reference_xc """
    peaks = spg.reference_xc(100, dp, 1).data[0][0][0]  # as quoted in the issue
    np.testing.assert_allclose([0, -4], peaks)
