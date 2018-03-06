import pytest
import fpd_data_processing.diffraction_tools as dt


@pytest.mark.parametrize("av,wl", [
    (100000, 3.701), (200000, 2.507), (300000, 1.968)])
def test_acceleration_voltage_to_wavelength(av, wl):
    wavelength = dt.acceleration_voltage_to_wavelength(av)
    wavelength_pm = wavelength * 10**12
    assert pytest.approx(wavelength_pm, abs=0.001) == wl
