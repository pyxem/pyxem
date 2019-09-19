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

from pyxem.signals.electron_diffraction1d import ElectronDiffraction1D
from pyxem.signals.reduced_intensity1d import ReducedIntensity1D
from pyxem.generators.red_intensity_generator1d import ReducedIntensityGenerator1D


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


def test_fit_atomic_scattering(red_int_generator):
    calib = 0.1
    red_int_generator.set_diffraction_calibration(calibration=calib)
    elements = ['Cu']
    fracs = [1]
    assert red_int_generator.background_fit is None
    assert red_int_generator.normalisation is None
    red_int_generator.fit_atomic_scattering(elements=elements, fracs=fracs)

    fit_expected = np.array([[[11.0464083, 8.76653449, 6.44758362, 5.39985641,
                               4.87205424, 4.5707376, 4.39213254, 4.28497237,
                               4.21972043, 4.17912948],
                              [22.09281664, 17.533069, 12.89516724, 10.7997128,
                               9.74410846, 9.14147517, 8.78426507, 8.56994472,
                               8.43944083, 8.35825893]],

                             [[33.13922486, 26.29960343, 19.34275083, 16.19956918,
                               14.61616267, 13.71221275, 13.17639759, 12.85491707,
                               12.65916124, 12.53738839],
                              [44.18563327, 35.06613799, 25.7903345, 21.59942561,
                               19.48821693, 18.28295037, 17.56853016, 17.13988946,
                               16.87888168, 16.71651788]]])
    norm_expected = np.array([[[ 6.95347555,  4.67360174,  2.35465087,  1.30692365,
                                0.77912148,  0.47780484,  0.29919979,  0.19203962,
                                0.12678767,  0.08619672],
                                [13.9069511 ,  9.34720348,  4.70930174,  2.61384731,
                                1.55824297,  0.95560969,  0.59839959,  0.38407924,
                                0.25357535,  0.17239345]],

                                [[20.86042659, 14.02080517,  7.06395259,  3.92077095,
                                2.33736445,  1.43341453,  0.89759938,  0.57611885,
                                0.38036302,  0.25859017],
                                [27.81390212, 18.6944069 ,  9.41860346,  5.2276946 ,
                                3.11648593,  1.91121937,  1.19679917,  0.76815847,
                                0.5071507 ,  0.3447869 ]]])

    assert np.allclose(red_int_generator.background_fit, fit_expected)
    assert np.allclose(red_int_generator.normalisation, norm_expected)


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
    red_int_generator.mask_from_bkgd_pattern(mask_pattern,mask_threshold=mask_threshold)

    expected = np.arange(10, 0, -1).reshape(1, 10) * np.arange(1, 5).reshape(4, 1)
    expected = expected.reshape(2, 2, 10)
    expected[:,:,:4] = 0

    assert np.array_equal(red_int_generator.signal.data, expected)

def test_mask_reduced_intensity(red_int_generator):
    mask_pattern = np.ones(10)
    mask_pattern[:4] = 0
    red_int_generator.mask_reduced_intensity(mask_pattern)

    expected = np.arange(10, 0, -1).reshape(1, 10) * np.arange(1, 5).reshape(4, 1)
    expected = expected.reshape(2, 2, 10)
    expected[:,:,:4] = 0

    assert np.array_equal(red_int_generator.signal.data, expected)

@pytest.mark.xfail(raises=ValueError)
def test_incorrect_mask(red_int_generator):
    mask_pattern = np.ones(10)
    mask_pattern[:4] = 0
    mask_pattern[8] = 2
    red_int_generator.mask_reduced_intensity(mask_pattern)


def test_get_reduced_intensity(red_int_generator):
    red_int_generator.set_diffraction_calibration(calibration=0.1)
    red_int_generator.fit_atomic_scattering(elements=['Cu'], fracs=[1])
    ri = red_int_generator.get_reduced_intensity()
    assert isinstance(ri, ReducedIntensity1D)

    ri_expected_single = np.array([-0.00000000e+00, 3.13870790e-02,
                                    8.28498177e-01, 2.30786214e+00,
                                    3.63850434e+00, 2.82242349e+00,
                                    -4.94086192e+00, -2.94293632e+01,
                                    -8.80017073e+01, -2.08564231e+02])
    ri_expected_array = np.vstack((ri_expected_single,ri_expected_single,
                                    ri_expected_single,ri_expected_single)).reshape(2,2,10)
    ri_expected = ReducedIntensity1D(ri_expected_array)

    assert np.allclose(ri.data, ri_expected.data)
