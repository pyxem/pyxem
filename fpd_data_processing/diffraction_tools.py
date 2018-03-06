import scipy.constants as sc


def acceleration_voltage_to_wavelength(acceleration_voltage):
    """Get electron wavelength from the acceleration voltage.

    Parameters
    ----------
    acceleration_voltage : float
        In Volt

    Returns
    -------
    wavelength : float
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
