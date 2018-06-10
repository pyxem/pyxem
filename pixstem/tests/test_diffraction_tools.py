import pytest
from pytest import approx
import numpy as np
import pixstem.diffraction_tools as dt


@pytest.mark.parametrize("av,wl", [
    (100000, 3.701e-12), (200000, 2.507e-12), (300000, 1.968e-12)])  # V, pm
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
                acceleration_voltage, lattice_size, miller_index)
        assert approx(scattering_angle, rel=0.001) == 9.84e-3

    @pytest.mark.parametrize("mi,sa", [
        ((1, 0, 0), 9.84e-3), ((0, 1, 0), 9.84e-3), ((0, 0, 1), 9.84e-3),
        ((2, 0, 0), 19.68e-3), ((0, 2, 0), 19.68e-3), ((0, 0, 2), 19.68e-3)])
    def test_miller_index(self, mi, sa):
        acceleration_voltage = 300000
        lattice_size = 2e-10  # 2 Ångstrøm (in meters)
        scattering_angle = dt.diffraction_scattering_angle(
                acceleration_voltage, lattice_size, mi)
        assert approx(scattering_angle, rel=0.001) == sa

    def test_array_like(self):
        # This should give ~9.84e-3 radians
        acceleration_voltage = 300000
        lattice_size = np.array([2e-10, 2e-10])
        miller_index = (1, 0, 0)
        scattering_angle = dt.diffraction_scattering_angle(
                acceleration_voltage, lattice_size, miller_index)
        assert len(scattering_angle) == 2
        sa_known = np.array([9.84e-3, 9.84e-3])
        assert approx(scattering_angle, rel=0.001) == sa_known
