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

from pyxem.signals import ReducedIntensity1D


@pytest.fixture
def RedIntData():
    data = np.ones((1, 10)) * np.arange(1, 5).reshape(4, 1)
    data = data.reshape(2, 2, 10)
    return data


def test_reduced_intensity1d_init(RedIntData):
    ri = ReducedIntensity1D(RedIntData)
    assert isinstance(ri, ReducedIntensity1D)


def test_damp_exponential(RedIntData):
    ri = ReducedIntensity1D(RedIntData)
    ri.axes_manager.signal_axes[0].scale = 1
    ri.axes_manager.signal_axes[0].offset = 0.5

    ri.damp_exponential(b=1)
    compare = np.array(
        [
            [
                [
                    7.78800783e-01,
                    1.05399225e-01,
                    1.93045414e-03,
                    4.78511739e-06,
                    1.60522806e-09,
                    7.28772410e-14,
                    4.47773244e-19,
                    3.72336312e-25,
                    4.19009319e-32,
                    6.38150345e-40,
                ],
                [
                    1.55760157e00,
                    2.10798449e-01,
                    3.86090827e-03,
                    9.57023478e-06,
                    3.21045611e-09,
                    1.45754482e-13,
                    8.95546488e-19,
                    7.44672624e-25,
                    8.38018639e-32,
                    1.27630069e-39,
                ],
            ],
            [
                [
                    2.33640235e00,
                    3.16197674e-01,
                    5.79136241e-03,
                    1.43553522e-05,
                    4.81568417e-09,
                    2.18631723e-13,
                    1.34331973e-18,
                    1.11700894e-24,
                    1.25702796e-31,
                    1.91445103e-39,
                ],
                [
                    3.11520313e00,
                    4.21596898e-01,
                    7.72181654e-03,
                    1.91404696e-05,
                    6.42091222e-09,
                    2.91508964e-13,
                    1.79109298e-18,
                    1.48934525e-24,
                    1.67603728e-31,
                    2.55260138e-39,
                ],
            ],
        ]
    )
    assert np.allclose(ri, compare)


def test_damp_lorch(RedIntData):
    ri = ReducedIntensity1D(RedIntData)
    ri.axes_manager.signal_axes[0].scale = 1
    ri.axes_manager.signal_axes[0].offset = 0.5

    ri.damp_lorch(s_max=10)
    compare = np.array(
        [
            [
                [
                    0.99589274,
                    0.96339776,
                    0.90031632,
                    0.81033196,
                    0.69864659,
                    0.57161993,
                    0.43633259,
                    0.30010544,
                    0.17001137,
                    0.05241541,
                ],
                [
                    1.99178547,
                    1.92679552,
                    1.80063263,
                    1.62066392,
                    1.39729317,
                    1.14323987,
                    0.87266519,
                    0.60021088,
                    0.34002274,
                    0.10483081,
                ],
            ],
            [
                [
                    2.98767821,
                    2.89019329,
                    2.70094895,
                    2.43099587,
                    2.09593976,
                    1.7148598,
                    1.30899778,
                    0.90031632,
                    0.51003411,
                    0.15724622,
                ],
                [
                    3.98357094,
                    3.85359105,
                    3.60126526,
                    3.24132783,
                    2.79458634,
                    2.28647973,
                    1.74533037,
                    1.20042175,
                    0.68004548,
                    0.20966163,
                ],
            ],
        ]
    )
    assert np.allclose(ri, compare)


def test_damp_updated_lorch(RedIntData):
    ri = ReducedIntensity1D(RedIntData)
    ri.axes_manager.signal_axes[0].scale = 1
    ri.axes_manager.signal_axes[0].offset = 0.5

    ri.damp_updated_lorch(s_max=10)
    compare = np.array(
        [
            [
                [
                    0.99753477,
                    0.97796879,
                    0.9396585,
                    0.88420257,
                    0.81388998,
                    0.73157686,
                    0.64053436,
                    0.54427698,
                    0.44638168,
                    0.35030873,
                ],
                [
                    1.99506954,
                    1.95593757,
                    1.879317,
                    1.76840514,
                    1.62777996,
                    1.46315372,
                    1.28106873,
                    1.08855397,
                    0.89276337,
                    0.70061746,
                ],
            ],
            [
                [
                    2.99260432,
                    2.93390636,
                    2.8189755,
                    2.65260771,
                    2.44166995,
                    2.19473058,
                    1.92160309,
                    1.63283095,
                    1.33914505,
                    1.05092619,
                ],
                [
                    3.99013909,
                    3.91187515,
                    3.758634,
                    3.53681029,
                    3.25555993,
                    2.92630744,
                    2.56213745,
                    2.17710793,
                    1.78552674,
                    1.40123492,
                ],
            ],
        ]
    )
    assert np.allclose(ri, compare)


def test_damp_low_q_region_erfc(RedIntData):
    ri = ReducedIntensity1D(RedIntData)
    ri.axes_manager.signal_axes[0].scale = 1
    ri.axes_manager.signal_axes[0].offset = 0.5

    ri.damp_low_q_region_erfc(scale=20, offset=20)
    compare = np.array(
        [
            [
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            ],
            [
                [0.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                [0.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            ],
        ]
    )
    assert np.allclose(ri, compare)


def test_multiple_scatter_correction(RedIntData):
    ri = ReducedIntensity1D(RedIntData)
    ri.axes_manager.signal_axes[0].scale = 1
    ri.axes_manager.signal_axes[0].offset = 0.5

    ri.fit_thermal_multiple_scattering_correction()

    assert ri.data.shape == (2, 2, 10)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_s_max_statements(RedIntData):
    ri = ReducedIntensity1D(RedIntData)
    ri.axes_manager.signal_axes[0].scale = 1
    ri.axes_manager.signal_axes[0].offset = 0.5

    ri.damp_lorch()
    ri.damp_updated_lorch()
    ri.fit_thermal_multiple_scattering_correction(s_max=5, plot=True)
    assert isinstance(ri, ReducedIntensity1D)
