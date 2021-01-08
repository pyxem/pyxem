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
import numpy as np

from pyxem.signals import ElectronDiffraction1D, ReducedIntensity1D
from pyxem.generators import ReducedIntensityGenerator1D


@pytest.fixture
def red_int_generator():
    data = np.arange(10, 0, -1).reshape(1, 10) * np.arange(1, 5).reshape(4, 1)
    data = data.reshape(2, 2, 10)
    rp = ElectronDiffraction1D(data)
    rigen = ReducedIntensityGenerator1D(rp)
    return rigen


def test_init(red_int_generator):
    assert isinstance(red_int_generator, ReducedIntensityGenerator1D)


def test_scattering_calibration(red_int_generator):
    calib = 0.1
    red_int_generator.set_diffraction_calibration(calibration=calib)
    s_scale = red_int_generator.signal.axes_manager.signal_axes[0].scale
    assert s_scale == calib
    offset = 0.05
    red_int_generator.set_diffraction_offset(offset=offset)
    s_offset = red_int_generator.signal.axes_manager.signal_axes[0].offset
    assert s_offset == offset


def test_fit_atomic_scattering(red_int_generator):
    calib = 0.1
    red_int_generator.set_diffraction_calibration(calibration=calib)
    elements = ["Cu"]
    fracs = [1]
    assert red_int_generator.background_fit is None
    assert red_int_generator.normalisation is None
    red_int_generator.fit_atomic_scattering(elements=elements, fracs=fracs)

    assert red_int_generator.background_fit.data.shape == (2, 2, 10)
    assert red_int_generator.normalisation.data.shape == (2, 2, 10)


def test_set_cutoff(red_int_generator):
    s_min, s_max = 0, 8
    red_int_generator.set_s_cutoff(s_min, s_max)
    assert red_int_generator.cutoff == [s_min, s_max]


def test_subtract_bkgd(red_int_generator):
    bkgd_pattern = np.arange(10, 0, -1).reshape(10)
    red_int_generator.subtract_bkgd_pattern(bkgd_pattern)

    expected = np.arange(10, 0, -1).reshape(1, 10) * np.arange(0, 4).reshape(4, 1)
    expected = expected.reshape(2, 2, 10)

    assert np.array_equal(red_int_generator.signal.data, expected)


def test_mask_from_bkgd(red_int_generator):
    mask_pattern = np.arange(10, 0, -1).reshape(10)
    mask_threshold = 6.5
    red_int_generator.mask_from_bkgd_pattern(
        mask_pattern, mask_threshold=mask_threshold
    )

    expected = np.arange(10, 0, -1).reshape(1, 10) * np.arange(1, 5).reshape(4, 1)
    expected = expected.reshape(2, 2, 10)
    expected[:, :, :4] = 0

    assert np.array_equal(red_int_generator.signal.data, expected)


def test_mask_reduced_intensity(red_int_generator):
    mask_pattern = np.ones(10)
    mask_pattern[:4] = 0
    red_int_generator.mask_reduced_intensity(mask_pattern)

    expected = np.arange(10, 0, -1).reshape(1, 10) * np.arange(1, 5).reshape(4, 1)
    expected = expected.reshape(2, 2, 10)
    expected[:, :, :4] = 0

    assert np.array_equal(red_int_generator.signal.data, expected)


def test_incorrect_mask(red_int_generator):
    mask_pattern = np.ones(10)
    mask_pattern[:4] = 0
    mask_pattern[8] = 2
    with pytest.raises(
        ValueError, match="Masking array does not consist of zeroes and ones"
    ):
        red_int_generator.mask_reduced_intensity(mask_pattern)


def test_get_reduced_intensity(red_int_generator):
    red_int_generator.set_diffraction_calibration(calibration=0.1)
    red_int_generator.fit_atomic_scattering(elements=["Cu"], fracs=[1])
    ri = red_int_generator.get_reduced_intensity()
    assert isinstance(ri, ReducedIntensity1D)
