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
from pytest import approx
import numpy as np
import scipy.constants as sc
import pyxem.utils.dpc_utils as dpct


class TestBetaToBst:
    def test_zero(self):
        data = np.zeros((100, 100))
        bst = dpct.beta_to_bst(data, 200000)
        assert data.shape == bst.shape
        assert (data == 0.0).all()

    def test_ones(self):
        data = np.ones((100, 100)) * 10
        bst = dpct.beta_to_bst(data, 200000)
        assert data.shape == bst.shape
        assert (data != 0.0).all()

    def test_beta_to_bst_to_beta(self):
        beta = 2e-6
        output = dpct.bst_to_beta(dpct.beta_to_bst(beta, 200000), 200000)
        assert beta == output

    def test_known_value(self):
        # From https://dx.doi.org/10.1016/j.ultramic.2016.03.006
        bst = 10e-9 * 1  # 10 nm, 1 Tesla
        av = 200000  # 200 kV
        beta = dpct.bst_to_beta(bst, av)
        assert approx(beta, rel=1e-4) == 6.064e-6


class TestBstToBeta:
    def test_zero(self):
        data = np.zeros((100, 100))
        beta = dpct.bst_to_beta(data, 200000)
        assert data.shape == beta.shape
        assert (data == 0.0).all()

    def test_ones(self):
        data = np.ones((100, 100)) * 10
        beta = dpct.bst_to_beta(data, 200000)
        assert data.shape == beta.shape
        assert (data != 0.0).all()

    def test_bst_to_beta_to_bst(self):
        bst = 10e-6
        output = dpct.beta_to_bst(dpct.bst_to_beta(bst, 200000), 200000)
        assert bst == output


class TestEtToBeta:
    def test_zero(self):
        data = np.zeros((100, 100))
        beta = dpct.et_to_beta(data, 200000)
        assert data.shape == beta.shape
        assert (data == 0.0).all()

    def test_ones(self):
        data = np.ones((100, 100)) * 10
        beta = dpct.bst_to_beta(data, 200000)
        assert data.shape == beta.shape
        assert (data != 0.0).all()


class TestAccelerationVoltageToVelocity:
    def test_zero(self):
        assert dpct.acceleration_voltage_to_velocity(0) == 0.0

    @pytest.mark.parametrize(
        "av,vel", [(100000, 1.6434e8), (200000, 2.0844e8), (300000, 2.3279e8)]
    )  # V, m/s
    def test_values(self, av, vel):
        v = dpct.acceleration_voltage_to_velocity(av)
        assert approx(v, rel=0.001) == vel


class TestAccelerationVoltageToRelativisticMass:
    def test_zero(self):
        mr = dpct.acceleration_voltage_to_relativistic_mass(0.0)
        assert approx(mr) == sc.electron_mass

    def test_200kv(self):
        mr = dpct.acceleration_voltage_to_relativistic_mass(200000)
        assert approx(mr) == 1.268e-30
