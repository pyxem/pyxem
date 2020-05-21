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
import pyxem.utils.diffraction_tools as dt


@pytest.mark.parametrize(
    "av,wl", [(100000, 3.701e-12), (200000, 2.507e-12), (300000, 1.968e-12)]
)  # V, pm
def test_acceleration_voltage_to_wavelength(av, wl):
    wavelength = dt.acceleration_voltage_to_wavelength(av)
    assert approx(wavelength, rel=0.001, abs=0.0) == wl


def test_acceleration_voltage_to_wavelength_array():
    av = np.array([100000, 200000, 300000])  # In Volt
    wavelength = dt.acceleration_voltage_to_wavelength(av)
    wl = np.array([3.701e-12, 2.507e-12, 1.968e-12])  # In pm
    assert len(wl) == 3
    assert approx(wavelength, rel=0.001, abs=0.0) == wl


class TestDiffractionScatteringAngle:
    def test_simple(self):
        # This should give ~9.84e-3 radians
        acceleration_voltage = 300000
        lattice_size = 2e-10  # 2 Ångstrøm (in meters)
        miller_index = (1, 0, 0)
        scattering_angle = dt.diffraction_scattering_angle(
            acceleration_voltage, lattice_size, miller_index
        )
        assert approx(scattering_angle, rel=0.001) == 9.84e-3

    @pytest.mark.parametrize(
        "mi,sa",
        [
            ((1, 0, 0), 9.84e-3),
            ((0, 1, 0), 9.84e-3),
            ((0, 0, 1), 9.84e-3),
            ((2, 0, 0), 19.68e-3),
            ((0, 2, 0), 19.68e-3),
            ((0, 0, 2), 19.68e-3),
        ],
    )
    def test_miller_index(self, mi, sa):
        acceleration_voltage = 300000
        lattice_size = 2e-10  # 2 Ångstrøm (in meters)
        scattering_angle = dt.diffraction_scattering_angle(
            acceleration_voltage, lattice_size, mi
        )
        assert approx(scattering_angle, rel=0.001) == sa

    def test_array_like(self):
        # This should give ~9.84e-3 radians
        acceleration_voltage = 300000
        lattice_size = np.array([2e-10, 2e-10])
        miller_index = (1, 0, 0)
        scattering_angle = dt.diffraction_scattering_angle(
            acceleration_voltage, lattice_size, miller_index
        )
        assert len(scattering_angle) == 2
        sa_known = np.array([9.84e-3, 9.84e-3])
        assert approx(scattering_angle, rel=0.001) == sa_known
