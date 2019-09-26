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

from pyxem.signals.reduced_intensity1d import ReducedIntensity1D


@pytest.fixture
def RedIntData():
    data = np.ones((1, 10)) * np.arange(4).reshape(4, 1)
    data = data.reshape(2, 2, 10)
    return data


def test_reduced_intensity1d_init(RedIntData):
    ri = ReducedIntensity1D(RedIntData)
    assert isinstance(ri, ReducedIntensity1D)


def test_damp_exponential(RedIntData):
    ri = ReducedIntensity1D(RedIntData)
    ri.damp_exponential(b=1)
    compare = np.array([[[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                          0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                          0.00000000e+00, 0.00000000e+00],
                         [1.00000000e+00, 3.67879441e-01, 1.83156389e-02, 1.23409804e-04,
                          1.12535175e-07, 1.38879439e-11, 2.31952283e-16, 5.24288566e-22,
                          1.60381089e-28, 6.63967720e-36]],

                        [[2.00000000e+00, 7.35758882e-01, 3.66312778e-02, 2.46819608e-04,
                          2.25070349e-07, 2.77758877e-11, 4.63904566e-16, 1.04857713e-21,
                          3.20762178e-28, 1.32793544e-35],
                         [3.00000000e+00, 1.10363832e+00, 5.49469167e-02, 3.70229412e-04,
                          3.37605524e-07, 4.16638316e-11, 6.95856849e-16, 1.57286570e-21,
                          4.81143267e-28, 1.99190316e-35]]])
    assert np.allclose(ri, compare)


def test_damp_lorch(RedIntData):
    ri = ReducedIntensity1D(RedIntData)
    ri.damp_lorch(s_max=10)
    compare = np.array([[[0., 0., 0., 0., 0.,
                          0., 0., 0., 0., 0.],
                         [0., 0.98363164, 0.93548928, 0.85839369, 0.75682673,
                          0.63661977, 0.50455115, 0.36788301, 0.23387232, 0.1092924]],

                        [[0., 1.96726329, 1.87097857, 1.71678738, 1.51365346,
                          1.27323954, 1.0091023, 0.73576602, 0.46774464, 0.21858481],
                         [0., 2.95089493, 2.80646785, 2.57518107, 2.27048019,
                          1.90985932, 1.51365346, 1.10364903, 0.70161696, 0.32787721]]])
    assert np.allclose(ri, compare)


def test_damp_updated_lorch(RedIntData):
    ri = ReducedIntensity1D(RedIntData)
    ri.damp_updated_lorch(s_max=10)
    compare = np.array([[[0., 0., 0., 0., 0.,
                          0., 0., 0., 0., 0.],
                         [0., 0.99016512, 0.96107415, 0.91394558, 0.85073648,
                          0.77403683, 0.68693073, 0.5928333, 0.49531303, 0.3979104]],

                        [[0., 1.98033024, 1.92214831, 1.82789116, 1.70147296,
                          1.54807365, 1.37386146, 1.18566661, 0.99062606, 0.7958208],
                         [0., 2.97049536, 2.88322246, 2.74183673, 2.55220944,
                          2.32211048, 2.06079219, 1.77849991, 1.48593909, 1.1937312]]])
    assert np.allclose(ri, compare)


def test_damp_low_q_region_erfc(RedIntData):
    ri = ReducedIntensity1D(RedIntData)
    ri.damp_low_q_region_erfc(scale=20, offset=1.3)
    compare = np.array([[[0., 0., 0., 0., 0.,
                          0., 0., 0., 0., 0.],
                         [0.03299603, 1., 1., 1., 1.,
                          1., 1., 1., 1., 1.]],

                        [[0.06599206, 2., 2., 2., 2.,
                          2., 2., 2., 2., 2.],
                         [0.09898808, 3., 3., 3., 3.,
                          3., 3., 3., 3., 3.]]])
    assert np.allclose(ri, compare)


def test_multiple_scatter_correction(RedIntData):
    ri = ReducedIntensity1D(RedIntData)
    ri.fit_thermal_multiple_scattering_correction()
    compare = np.array([[[0., 0., 0., 0.,
                          0., 0., 0., 0.,
                          0., 0.],
                         [1., 0.14216084, -0.10154346, -0.06092608,
                          0.03655565, 0.06580016, 0.00406174, -0.06904955,
                          0.02843217, 0.58082859]],

                        [[2., 0.28432171, -0.20308691, -0.12185215,
                          0.07311128, 0.13160032, 0.00812348, -0.1380991,
                          0.05686434, 1.16165714],
                         [3., 0.42648258, -0.30463036, -0.18277824,
                          0.10966691, 0.19740047, 0.01218523, -0.20714863,
                          0.08529649, 1.74248558]]])
    assert np.allclose(ri, compare)


def test_s_max_statements(RedIntData):
    ri = ReducedIntensity1D(RedIntData)
    ri.damp_lorch()
    ri.damp_updated_lorch()
    ri.fit_thermal_multiple_scattering_correction(s_max=5, plot=True)
    assert isinstance(ri, ReducedIntensity1D)
