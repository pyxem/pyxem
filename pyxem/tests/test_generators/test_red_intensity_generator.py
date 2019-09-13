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
from pyxem.signals.reduced_intensity_profile import ReducedIntensityProfile
from pyxem.generators.red_intensity_generator import ReducedIntensityGenerator


@pytest.fixture
def red_int_generator():
    data = np.arange(10, 0, -1).reshape(1, 10) * np.arange(1, 5).reshape(4, 1)
    data = data.reshape(2, 2, 10)
    rp = ElectronDiffraction1D(data)
    rigen = ReducedIntensityGenerator(rp)
    return rigen


def test_init(red_int_generator):
    assert isinstance(red_int_generator, ReducedIntensityGenerator)


def test_scattering_calibration(red_int_generator):
    calib = 0.1
    red_int_generator.specify_scattering_calibration(calibration=calib)
    s_scale = red_int_generator.signal.axes_manager.signal_axes[0].scale
    assert s_scale == calib


def test_fit_atomic_scattering(red_int_generator):
    calib = 0.1
    red_int_generator.specify_scattering_calibration(calibration=calib)
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
    norm_expected = np.array([[[1.24147037, 1.01779718, 0.72243479, 0.53822083, 0.41556406,
                                0.32543231, 0.25752301, 0.206315, 0.16763856, 0.13822315],
                               [2.48294075, 2.03559437, 1.44486959, 1.07644167, 0.83112813,
                                0.65086462, 0.51504602, 0.41263001, 0.33527711, 0.27644629]],

                              [[3.72441111, 3.05339154, 2.16730438, 1.61466249, 1.24669219,
                                0.97629693, 0.77256903, 0.61894501, 0.50291567, 0.41466944],
                               [4.9658815, 4.07118874, 2.88973919, 2.15288333, 1.66225626,
                                1.30172924, 1.03009204, 0.82526001, 0.67055423, 0.55289259]]])

    assert np.allclose(red_int_generator.background_fit, fit_expected)
    assert np.allclose(red_int_generator.normalisation, norm_expected)


def test_specify_cutoff(red_int_generator):
    s_min, s_max = 0, 8
    red_int_generator.specify_cutoff_vector(s_min, s_max)
    assert red_int_generator.cutoff == [s_min, s_max]
    return


def test_subtract_bkgd(red_int_generator):
    bkgd_pattern = np.arange(10, 0, -1).reshape(10)
    red_int_generator.subtract_bkgd_pattern(bkgd_pattern)

    expected = np.arange(10, 0, -1).reshape(1, 10) * np.arange(0, 4).reshape(4, 1)
    expected = expected.reshape(2, 2, 10)

    assert np.array_equal(red_int_generator.signal.data, expected)
    return


def test_get_reduced_intensity(red_int_generator):

    elements = ['Cu']
    fracs = [1]
    red_int_generator.fit_atomic_scattering(elements=elements, fracs=fracs)
    ri = red_int_generator.get_reduced_intensity()
    assert isinstance(ri, ReducedIntensityProfile)
    ri = red_int_generator.get_reduced_intensity(cutoff=[0, 8])
    assert isinstance(ri, ReducedIntensityProfile)
    # better test needed here, an assert allclose or such at least
    return
