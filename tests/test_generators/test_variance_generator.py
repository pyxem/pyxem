# -*- coding: utf-8 -*-
# Copyright 2017-2018 The pyXem developers
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

from pyxem.generators.variance_generator import VarianceGenerator
from pyxem.signals.electron_diffraction import ElectronDiffraction

from pyxem.signals.diffraction_variance import DiffractionVariance

@pytest.fixture(params=[
    np.array([[[0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0., 0., 0.],
               [0., 0., 1., 2., 1., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.]],
              [[0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 1., 0., 0., 0.],
               [0., 0., 0., 1., 2., 1., 0., 0.],
               [0., 0., 0., 0., 1., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.]],
              [[0., 0., 0., 0., 0., 0., 0., 2.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0., 0., 0.],
               [0., 0., 1., 2., 1., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.]],
              [[0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 2., 0., 0., 0.],
               [0., 0., 0., 2., 2., 2., 0., 0.],
               [0., 0., 0., 0., 2., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.]]]).reshape(2,2,8,8)
])

def electron_diffraction(request):
    return ElectronDiffraction(request.param)

@pytest.fixture
def variance_generator(electron_diffraction):
    return VarianceGenerator(electron_diffraction)

class TestVarianceGenerator:

    @pytest.mark.parametrize('dqe', [
        0.5,
        0.6
    ])
    def test_get_diffraction_variance(
            self,
            variance_generator: VarianceGenerator,
            dqe
            ):
        vardps = variance_generator.get_diffraction_variance(dqe)
        assert isinstance(vardps, DiffractionVariance)

        # TEST answers as well
        mean_dp = np.array(
        [[0., 0., 0., 0., 0., 0., 0., 0.5],
         [0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0.5, 0., 0., 0., 0.],
         [0., 0., 0.5, 1., 1.25, 0., 0., 0.],
         [0., 0., 0., 1.25, 1., 0.75, 0., 0.],
         [0., 0., 0., 0., 0.75, 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0.]]).reshape(8,8)
        meansq_dp = np.array(
        [[0., 0., 0., 0., 0., 0., 0., 1.],
         [0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0.5, 0., 0., 0., 0.],
         [0., 0., 0.5, 2., 1.75, 0., 0., 0.],
         [0., 0., 0., 1.75, 2., 1.25, 0., 0.],
         [0., 0., 0., 0., 1.25, 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0.]]).reshape(8,8)
        var_dp = np.array(
         [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3.],
          [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
          [np.nan, np.nan, np.nan, 1., np.nan, np.nan, np.nan, np.nan],
          [np.nan, np.nan, 1., 1., 0.12, np.nan, np.nan, np.nan],
          [np.nan, np.nan, np.nan, 0.12, 1., 0.6875/0.75/0.75 , np.nan, np.nan],
          [np.nan, np.nan, np.nan, np.nan,  0.6875/0.75/0.75, np.nan, np.nan, np.nan],
          [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
          [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]).reshape(8,8)
        corr_var_dp = var_dp-np.divide(dqe,mean_dp)

        assert np.array_equal(vardps.data[0,0], mean_dp)
        assert np.array_equal(vardps.data[0,1], meansq_dp)
        assert np.allclose(vardps.data[1,0], var_dp,atol=1e-14,equal_nan=True)
        assert np.allclose(vardps.data[1,1], corr_var_dp,atol=1e-14,equal_nan=True)

    # Here's the non-normalised variance matrix to test against
    # [0.,0.,0.,0.,0.,0.,0.,0.75]
    # [0.,0.,0.,0.,0.,0.,0.,0.]
    # [0.,0.,0.,0.25,0.1875,0.,0.,0.]
    # [0.,0.,0.25,1.,0.1875,0.,0.,0.]
    # [0.,0.,0.,0.1875,1.,0.6875,0.,0.]
    # [0.,0.,0.,0.,0.6875,0.,0.,0.]
    # [0.,0.,0.,0.,0.,0.,0.,0.]
    # [0.,0.,0.,0.,0.,0.,0.,0.]
