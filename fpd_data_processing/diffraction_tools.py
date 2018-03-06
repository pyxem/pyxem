import numpy as np
import scipy.constants as sc


def acceleration_voltage_to_wavelength(acceleration_voltage):
    """Get electron wavelength from the acceleration voltage.

    Parameters
    ----------
    acceleration_voltage : float or array-like
        In Volt

    Returns
    -------
    wavelength : float or array-like
        In meters

    Examples
    --------
    >>> import fpd_data_processing.diffraction_tools as dt
    >>> wavelength = dt.acceleration_voltage_to_wavelength(200000)
    >>> wavelength_picometer = wavelength*10**12

    """
    E = acceleration_voltage*sc.elementary_charge
    h = sc.Planck
    m0 = sc.electron_mass
    c = sc.speed_of_light
    wavelength = h/(2*m0*E*(1 + (E/(2*m0*c**2))))**0.5
    return wavelength


def diffraction_scattering_angle(
        acceleration_voltage, lattice_size, miller_index):
    """Get electron scattering angle from a crystal lattice.

    Returns the total scattering angle, as measured from the middle of the
    direct beam (0, 0, 0) to the given Miller index.

    Miller index: h, k, l = miller_index
    Interplanar distance: d = a / (h**2 + k**2 + l**2)**0.5
    Bragg's law: theta = arcsin(electron_wavelength / (2 * d))
    Total scattering angle (phi):  phi = 2 * theta

    Parameters
    ----------
    acceleration_voltage : float
        In Volt
    lattice_size : float or array-like
        In nanometer
    miller_index : tuple
        (h, k, l)

    Returns
    -------
    angle : float
        Scattering angle in mrad.

    """
    wavelength = acceleration_voltage_to_wavelength(acceleration_voltage)
    wavelength_nm = wavelength*10**9
    H, K, L = miller_index
    a = lattice_size
    d = a/(H**2 + K**2 + L**2)**0.5
    scattering_angle = 2 * np.arcsin(wavelength_nm / (2 * d))
    scattering_angle_mrad = scattering_angle*1000
    return scattering_angle_mrad
