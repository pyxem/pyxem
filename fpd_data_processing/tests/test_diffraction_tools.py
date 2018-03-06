import pytest
from pytest import approx
import numpy as np
import fpd_data_processing.diffraction_tools as dt


@pytest.mark.parametrize("av,wl", [
    (100000, 3.701), (200000, 2.507), (300000, 1.968)])
def test_acceleration_voltage_to_wavelength(av, wl):
    wavelength = dt.acceleration_voltage_to_wavelength(av)
    wavelength_pm = wavelength * 10**12
    assert approx(wavelength_pm, abs=0.001) == wl


def test_acceleration_voltage_to_wavelength_array():
    av = np.array([100000, 200000, 300000])
    wavelength = dt.acceleration_voltage_to_wavelength(av)
    wavelength_pm = wavelength * 10**12
    wl = np.array([3.701, 2.507, 1.968])
    assert len(wl) == 3
    assert approx(wavelength_pm, abs=0.001) == wl


class TestDiffractionScatteringAngle:

    def test_simple(self):
        # This should give ~9.84 mrad
        acceleration_voltage = 300000
        lattice_size = 0.2
        miller_index = (1, 0, 0)
        scattering_angle = dt.diffraction_scattering_angle(
                acceleration_voltage, lattice_size, miller_index)
        assert approx(scattering_angle, abs=0.01) == 9.84

    @pytest.mark.parametrize("mi,sa", [
        ((1, 0, 0), 9.84), ((0, 1, 0), 9.84), ((0, 0, 1), 9.84),
        ((2, 0, 0), 19.68), ((0, 2, 0), 19.68), ((0, 0, 2), 19.68)])
    def test_miller_index(self, mi, sa):
        acceleration_voltage = 300000
        lattice_size = 0.2
        scattering_angle = dt.diffraction_scattering_angle(
                acceleration_voltage, lattice_size, mi)
        assert approx(scattering_angle, abs=0.01) == sa

    def test_array_like(self):
        # This should give ~9.84 mrad
        acceleration_voltage = 300000
        lattice_size = np.array([0.2, 0.2])
        miller_index = (1, 0, 0)
        scattering_angle = dt.diffraction_scattering_angle(
                acceleration_voltage, lattice_size, miller_index)
        assert len(scattering_angle) == 2
        assert approx(scattering_angle, abs=0.01) == np.array([9.84, 9.84])
